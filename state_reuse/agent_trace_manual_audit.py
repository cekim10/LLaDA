"""Prepare and analyze a blinded, stratified manual audit of agent turns.

Typical workflow:
  python3 state_reuse/agent_trace_manual_audit.py prepare \
    --turns state_reuse/results/swe_agent_audit_random100/turns.jsonl \
    --sample-size 400 --overlap-size 100 \
    --out state_reuse/results/swe_agent_manual400

  python3 state_reuse/agent_trace_manual_audit.py label \
    --packet state_reuse/results/swe_agent_manual400/labeler_a.csv \
    --labels state_reuse/results/swe_agent_manual400/labels_a.csv

  python3 state_reuse/agent_trace_manual_audit.py analyze \
    --manifest state_reuse/results/swe_agent_manual400/sample_manifest.jsonl \
    --labels-a state_reuse/results/swe_agent_manual400/labels_a.csv \
    --labels-b state_reuse/results/swe_agent_manual400/labels_b.csv \
    --out state_reuse/results/swe_agent_manual400/analysis
"""

import argparse
import collections
import csv
import hashlib
import json
import math
import os
import random
import sys


LABELS = ("safe", "history_dependent", "ambiguous", "bookkeeping")
LABEL_KEYS = {"s": "safe", "h": "history_dependent", "a": "ambiguous", "b": "bookkeeping"}
PACKET_FIELDS = (
    "item_id",
    "manual_label",
    "confidence",
    "dependency_evidence",
    "notes",
    "current_task",
    "old_history",
    "latest_observation",
    "current_generation",
)


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_target(value):
    if value is True:
        return "resolved"
    if value is False:
        return "unresolved"
    return "unknown"


def stratum_for(turn):
    return f"{turn.get('command_family', 'unknown')}|{normalize_target(turn.get('target'))}"


def proportional_allocation(group_sizes, total):
    population = sum(group_sizes.values())
    if not 0 <= total <= population:
        raise ValueError(f"requested {total} items from a population of {population}")
    raw = {key: total * size / population for key, size in group_sizes.items()}
    allocation = {key: min(group_sizes[key], math.floor(value)) for key, value in raw.items()}
    remaining = total - sum(allocation.values())
    order = sorted(group_sizes, key=lambda key: (raw[key] - allocation[key], group_sizes[key], key), reverse=True)
    for key in order:
        if not remaining:
            break
        if allocation[key] < group_sizes[key]:
            allocation[key] += 1
            remaining -= 1
    if remaining:
        raise RuntimeError("could not complete proportional allocation")
    return allocation


def blinded_id(key, seed):
    return hashlib.sha256(f"swe-agent-manual-audit:{seed}:{key}".encode()).hexdigest()[:16]


def packet_row(turn, item_id):
    return {
        "item_id": item_id,
        "manual_label": "",
        "confidence": "",
        "dependency_evidence": "",
        "notes": "",
        "current_task": turn.get("task_text", ""),
        "old_history": turn.get("old_history_for_labeling", turn.get("old_history", "")),
        "latest_observation": turn.get("latest_observation", ""),
        "current_generation": turn.get("ai_text", ""),
    }


def write_packet(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PACKET_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_rubric(path):
    text = """# SWE-agent turn labeling rubric

## Question

Would an agent need the semantic content of `old_history` to perform `current_generation`
correctly, given only `current_task` and `latest_observation`?

Judge necessity, not whether the action is vaguely predictable or commonly follows the latest
observation. Treat this as a counterfactual: if old history were unavailable, would information
needed to justify or execute this particular generation be missing?

## Labels

- `safe`: The current task and latest observation contain all information materially needed for
  this generation. Old history is provenance or redundant context.
- `history_dependent`: The generation requires a fact, constraint, diagnosis, code content,
  prior result, or plan state found only in old history.
- `ambiguous`: The trace does not support a confident necessity judgment. Do not force a binary
  label or assume an unstated workflow.
- `bookkeeping`: The generation is malformed, formatting-only, a wrapper retry, or otherwise
  non-substantive.

## Strict decisions

- A latest `test failed` observation does not by itself make the next action safe. If the agent
  chooses a file or action using a plan formed earlier, label `history_dependent`.
- Opening or searching a file is safe only when the target and reason follow from the current task
  or latest observation. Navigation that continues an earlier plan is `history_dependent`.
- Editing is safe only when the required code and intended change are present in the current task
  or latest observation. Code inspected earlier counts as old-history dependence.
- Re-running a command can be safe when the latest observation explicitly requests correction or
  retry and contains the needed details.
- Do not label an incorrect or inexplicable action safe merely because it could be emitted without
  context. The question is whether this generation can be performed correctly.

For `history_dependent`, record the required old-history fact in `dependency_evidence`. Use
`confidence=high` only when the necessity judgment is explicit; otherwise use `low` or
`ambiguous`.

Packets intentionally omit heuristic labels, trajectory outcome, original turn keys, and the
other annotator's order and labels.
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def prepare(args):
    turns = read_jsonl(args.turns)
    if not turns:
        raise ValueError("turn file is empty")
    missing = [turn["key"] for turn in turns if "old_history" not in turn]
    if missing:
        raise ValueError(
            "turns.jsonl lacks full old_history; regenerate it with the updated "
            f"agent_trace_audit.py (first missing key: {missing[0]})"
        )
    if args.overlap_size > args.sample_size:
        raise ValueError("--overlap-size cannot exceed --sample-size")

    rng = random.Random(args.seed)
    groups = collections.defaultdict(list)
    for turn in turns:
        groups[stratum_for(turn)].append(turn)
    population_sizes = {key: len(rows) for key, rows in groups.items()}
    sample_allocation = proportional_allocation(population_sizes, args.sample_size)

    selected_by_stratum = {}
    for key in sorted(groups):
        selected_by_stratum[key] = rng.sample(groups[key], sample_allocation[key])
    overlap_allocation = proportional_allocation(
        {key: len(rows) for key, rows in selected_by_stratum.items()}, args.overlap_size
    )
    overlap_keys = set()
    for key in sorted(selected_by_stratum):
        overlap_keys.update(turn["key"] for turn in rng.sample(selected_by_stratum[key], overlap_allocation[key]))

    selected = [turn for key in sorted(selected_by_stratum) for turn in selected_by_stratum[key]]
    non_overlap = [turn for turn in selected if turn["key"] not in overlap_keys]
    rng.shuffle(non_overlap)
    a_only_count = (len(non_overlap) + 1) // 2
    a_keys = overlap_keys | {turn["key"] for turn in non_overlap[:a_only_count]}
    b_keys = overlap_keys | {turn["key"] for turn in non_overlap[a_only_count:]}

    manifest = []
    for turn in selected:
        stratum = stratum_for(turn)
        item_id = blinded_id(turn["key"], args.seed)
        manifest.append(
            {
                "item_id": item_id,
                "key": turn["key"],
                "row_idx": turn["row_idx"],
                "instance_id": turn.get("instance_id", ""),
                "turn_index": turn.get("turn_index"),
                "stratum": stratum,
                "population_n": population_sizes[stratum],
                "sample_n": sample_allocation[stratum],
                "sampling_weight": population_sizes[stratum] / sample_allocation[stratum],
                "command_family": turn.get("command_family", "unknown"),
                "target": turn.get("target"),
                "context_tokens_est": turn.get("context_tokens_est", 0),
                "old_history_tokens_est": turn.get("old_history_tokens_est", 0),
                "in_labeler_a": turn["key"] in a_keys,
                "in_labeler_b": turn["key"] in b_keys,
                "is_overlap": turn["key"] in overlap_keys,
                "packet": packet_row(turn, item_id),
            }
        )

    packet_a = [row["packet"] for row in manifest if row["in_labeler_a"]]
    packet_b = [row["packet"] for row in manifest if row["in_labeler_b"]]
    random.Random(args.seed + 1).shuffle(packet_a)
    random.Random(args.seed + 2).shuffle(packet_b)
    os.makedirs(args.out, exist_ok=True)
    write_jsonl(os.path.join(args.out, "sample_manifest.jsonl"), manifest)
    write_packet(os.path.join(args.out, "labeler_a.csv"), packet_a)
    write_packet(os.path.join(args.out, "labeler_b.csv"), packet_b)
    write_rubric(os.path.join(args.out, "LABELING_RUBRIC.md"))
    design = {
        "turns_file": os.path.abspath(args.turns),
        "seed": args.seed,
        "population_turns": len(turns),
        "population_trajectories": len({turn["row_idx"] for turn in turns}),
        "sample_size": len(manifest),
        "sample_trajectories": len({row["row_idx"] for row in manifest}),
        "overlap_size": len(overlap_keys),
        "labeler_a_items": len(packet_a),
        "labeler_b_items": len(packet_b),
        "strata": {
            key: {
                "population_n": population_sizes[key],
                "sample_n": sample_allocation[key],
                "overlap_n": overlap_allocation[key],
            }
            for key in sorted(groups)
        },
    }
    with open(os.path.join(args.out, "design.json"), "w", encoding="utf-8") as f:
        json.dump(design, f, indent=2)
    print(json.dumps(design, indent=2))


def read_packet_labels(path, require_history_evidence=False):
    if not path or not os.path.exists(path):
        return {}
    labels = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            item_id = row.get("item_id", "").strip()
            if not item_id:
                raise ValueError(f"missing item_id in {path}")
            label = (row.get("manual_label") or row.get("label") or "").strip().lower()
            if not label:
                continue
            if label in LABEL_KEYS:
                label = LABEL_KEYS[label]
            if label not in LABELS:
                raise ValueError(f"invalid label {label!r} for item {item_id}")
            if require_history_evidence and label == "history_dependent":
                evidence = (row.get("dependency_evidence") or "").strip()
                if not evidence:
                    raise ValueError(
                        f"history_dependent item {item_id} is missing dependency_evidence"
                    )
            if item_id in labels:
                raise ValueError(f"duplicate labeled item_id {item_id} in {path}")
            labels[item_id] = label
    return labels


def label_interactively(args):
    csv.field_size_limit(sys.maxsize)
    with open(args.packet, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    existing = read_packet_labels(args.labels)
    for row in rows:
        if row["item_id"] in existing:
            row["manual_label"] = existing[row["item_id"]]
    remaining = [row for row in rows if not row.get("manual_label", "").strip()]
    for position, row in enumerate(remaining, 1):
        print("\n" + "=" * 88)
        print(f"ITEM {position}/{len(remaining)}  id={row['item_id']}")
        for title, field in (
            ("CURRENT TASK", "current_task"),
            ("OLD HISTORY", "old_history"),
            ("LATEST OBSERVATION", "latest_observation"),
            ("CURRENT GENERATION", "current_generation"),
        ):
            print(f"\n--- {title} ---\n{row[field]}")
        while True:
            choice = input("\nlabel [s=safe, h=history, a=ambiguous, b=bookkeeping, q=quit]: ").strip().lower()
            if choice == "q":
                write_packet(args.labels, rows)
                print(f"saved progress to {args.labels}")
                return
            if choice in LABEL_KEYS:
                row["manual_label"] = LABEL_KEYS[choice]
                break
            print("invalid choice")
        row["confidence"] = input("confidence [high/low]: ").strip().lower()
        if row["manual_label"] == "history_dependent":
            while not row["dependency_evidence"]:
                row["dependency_evidence"] = input("required old-history fact: ").strip()
                if not row["dependency_evidence"]:
                    print("dependency evidence is required for history_dependent")
        row["notes"] = input("notes (optional): ").strip()
        write_packet(args.labels, rows)
    print(f"complete; labels saved to {args.labels}")


def cohen_kappa(pairs):
    if not pairs:
        return float("nan")
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    a_counts = collections.Counter(a for a, _ in pairs)
    b_counts = collections.Counter(b for _, b in pairs)
    expected = sum((a_counts[label] / n) * (b_counts[label] / n) for label in LABELS)
    return (observed - expected) / (1 - expected) if expected < 1 else float("nan")


def metric_bounds(rows):
    substantive = [row for row in rows if row["final_label"] != "bookkeeping"]
    denominator = sum(row["sampling_weight"] for row in substantive)
    context_total = sum(row["sampling_weight"] * row["context_tokens_est"] for row in substantive)
    safe_n = sum(row["sampling_weight"] for row in substantive if row["final_label"] == "safe")
    ambiguous_n = sum(row["sampling_weight"] for row in substantive if row["final_label"] == "ambiguous")
    safe_cold = sum(
        row["sampling_weight"] * row["old_history_tokens_est"]
        for row in substantive
        if row["final_label"] == "safe"
    )
    ambiguous_cold = sum(
        row["sampling_weight"] * row["old_history_tokens_est"]
        for row in substantive
        if row["final_label"] == "ambiguous"
    )
    return {
        "f_safe_lower": safe_n / denominator if denominator else float("nan"),
        "f_safe_upper": (safe_n + ambiguous_n) / denominator if denominator else float("nan"),
        "oracle_saving_lower": safe_cold / context_total if context_total else float("nan"),
        "oracle_saving_upper": (safe_cold + ambiguous_cold) / context_total if context_total else float("nan"),
    }


def percentile(values, q):
    values = sorted(values)
    if not values:
        return float("nan")
    position = (len(values) - 1) * q
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - position) + values[hi] * (position - lo)


def cluster_bootstrap(rows, trials, seed):
    clusters = collections.defaultdict(list)
    for row in rows:
        clusters[row["row_idx"]].append(row)
    cluster_ids = sorted(clusters)
    rng = random.Random(seed)
    draws = collections.defaultdict(list)
    for _ in range(trials):
        sampled = []
        for cluster_id in rng.choices(cluster_ids, k=len(cluster_ids)):
            sampled.extend(clusters[cluster_id])
        metrics = metric_bounds(sampled)
        for key, value in metrics.items():
            draws[key].append(value)
    return {key: [percentile(values, 0.025), percentile(values, 0.975)] for key, values in draws.items()}


def fmt_pct(value):
    return "nan" if value != value else f"{100 * value:.1f}%"


def write_adjudication(path, disagreements, manifest_by_id):
    rows = []
    for item_id, label_a, label_b in disagreements:
        row = dict(manifest_by_id[item_id]["packet"])
        row["manual_label"] = ""
        row["notes"] = f"labeler_a={label_a}; labeler_b={label_b}"
        rows.append(row)
    write_packet(path, rows)


def analyze(args):
    csv.field_size_limit(sys.maxsize)
    manifest = read_jsonl(args.manifest)
    by_id = {row["item_id"]: row for row in manifest}
    labels_a = read_packet_labels(args.labels_a, require_history_evidence=True)
    labels_b = read_packet_labels(args.labels_b, require_history_evidence=True)
    adjudicated = read_packet_labels(args.adjudication, require_history_evidence=True)
    expected_a = {row["item_id"] for row in manifest if row["in_labeler_a"]}
    expected_b = {row["item_id"] for row in manifest if row["in_labeler_b"]}
    unknown_a = set(labels_a) - expected_a
    unknown_b = set(labels_b) - expected_b
    unknown_adjudication = set(adjudicated) - set(by_id)
    if unknown_a:
        raise ValueError(f"labels A contains {len(unknown_a)} item(s) outside packet A")
    if unknown_b:
        raise ValueError(f"labels B contains {len(unknown_b)} item(s) outside packet B")
    if unknown_adjudication:
        raise ValueError(f"adjudication contains {len(unknown_adjudication)} unknown item(s)")
    overlap = [row for row in manifest if row["is_overlap"]]
    pairs = [
        (labels_a[row["item_id"]], labels_b[row["item_id"]])
        for row in overlap
        if row["item_id"] in labels_a and row["item_id"] in labels_b
    ]
    disagreements = [
        (row["item_id"], labels_a[row["item_id"]], labels_b[row["item_id"]])
        for row in overlap
        if row["item_id"] in labels_a
        and row["item_id"] in labels_b
        and labels_a[row["item_id"]] != labels_b[row["item_id"]]
        and row["item_id"] not in adjudicated
    ]

    resolved = []
    unresolved = []
    for row in manifest:
        item_id = row["item_id"]
        if item_id in adjudicated:
            label = adjudicated[item_id]
        elif row["is_overlap"]:
            if item_id in labels_a and item_id in labels_b and labels_a[item_id] == labels_b[item_id]:
                label = labels_a[item_id]
            else:
                unresolved.append(item_id)
                continue
        else:
            source = labels_a if row["in_labeler_a"] else labels_b
            if item_id not in source:
                unresolved.append(item_id)
                continue
            label = source[item_id]
        item = {key: value for key, value in row.items() if key != "packet"}
        item["final_label"] = label
        resolved.append(item)

    os.makedirs(args.out, exist_ok=True)
    adjudication_queue = os.path.join(args.out, "adjudication.csv")
    if args.adjudication and os.path.abspath(args.adjudication) == os.path.abspath(adjudication_queue):
        if disagreements:
            write_adjudication(os.path.join(args.out, "adjudication_remaining.csv"), disagreements, by_id)
    else:
        write_adjudication(adjudication_queue, disagreements, by_id)
    write_jsonl(os.path.join(args.out, "resolved_labels.jsonl"), resolved)
    counts = collections.Counter(row["final_label"] for row in resolved)
    weighted_counts = collections.Counter()
    for row in resolved:
        weighted_counts[row["final_label"]] += row["sampling_weight"]
    agreement = sum(a == b for a, b in pairs) / len(pairs) if pairs else float("nan")
    summary = {
        "sample_items": len(manifest),
        "resolved_items": len(resolved),
        "unresolved_items": len(unresolved),
        "overlap_planned": len(overlap),
        "overlap_double_labeled": len(pairs),
        "raw_agreement": agreement,
        "cohen_kappa": cohen_kappa(pairs),
        "agreement_confusion": {
            label_a: {
                label_b: sum(a == label_a and b == label_b for a, b in pairs)
                for label_b in LABELS
            }
            for label_a in LABELS
        },
        "unadjudicated_disagreements": len(disagreements),
        "counts": dict(counts),
        "weighted_population_counts_est": dict(weighted_counts),
    }
    if not unresolved:
        bounds = metric_bounds(resolved)
        summary.update(bounds)
        summary["cluster_bootstrap_95ci"] = cluster_bootstrap(resolved, args.bootstrap_trials, args.seed)
        lower_ci = summary["cluster_bootstrap_95ci"]["oracle_saving_lower"]
        upper_ci = summary["cluster_bootstrap_95ci"]["oracle_saving_upper"]
        if lower_ci[0] > 0.30:
            summary["decision"] = "GO"
        elif upper_ci[1] < 0.20:
            summary["decision"] = "STOP"
        else:
            summary["decision"] = "INCONCLUSIVE"
    else:
        summary["decision"] = "INCOMPLETE"

    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    lines = [
        "# Manual agent trace audit",
        "",
        f"- resolved: {len(resolved)}/{len(manifest)}",
        f"- overlap double-labeled: {len(pairs)}/{len(overlap)}",
        f"- raw agreement: {fmt_pct(agreement)}",
        f"- Cohen's kappa: {summary['cohen_kappa']:.3f}" if summary["cohen_kappa"] == summary["cohen_kappa"] else "- Cohen's kappa: nan",
        f"- unadjudicated disagreements: {len(disagreements)}",
        f"- decision: **{summary['decision']}**",
    ]
    if not unresolved:
        ci = summary["cluster_bootstrap_95ci"]
        lines.extend(
            [
                "",
                "## Ambiguity Bounds",
                "",
                "| metric | lower | upper | approximate cluster-bootstrap 95% CI |",
                "|---|---:|---:|---:|",
                f"| safe-turn share | {fmt_pct(summary['f_safe_lower'])} | {fmt_pct(summary['f_safe_upper'])} | "
                f"lower [{fmt_pct(ci['f_safe_lower'][0])}, {fmt_pct(ci['f_safe_lower'][1])}]; "
                f"upper [{fmt_pct(ci['f_safe_upper'][0])}, {fmt_pct(ci['f_safe_upper'][1])}] |",
                f"| oracle saving | {fmt_pct(summary['oracle_saving_lower'])} | {fmt_pct(summary['oracle_saving_upper'])} | "
                f"lower [{fmt_pct(ci['oracle_saving_lower'][0])}, {fmt_pct(ci['oracle_saving_lower'][1])}]; "
                f"upper [{fmt_pct(ci['oracle_saving_upper'][0])}, {fmt_pct(ci['oracle_saving_upper'][1])}] |",
                "",
                "Lower treats every ambiguous label as history-dependent; upper treats every ambiguous label as safe.",
            ]
        )
    else:
        lines.extend(["", "Metrics are withheld until all sampled items and overlap disagreements are resolved."])
    with open(os.path.join(args.out, "SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    csv.field_size_limit(sys.maxsize)
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare")
    prep.add_argument("--turns", required=True)
    prep.add_argument("--sample-size", type=int, default=400)
    prep.add_argument("--overlap-size", type=int, default=100)
    prep.add_argument("--seed", type=int, default=0)
    prep.add_argument("--out", required=True)
    prep.set_defaults(func=prepare)

    labeling = sub.add_parser("label")
    labeling.add_argument("--packet", required=True)
    labeling.add_argument("--labels", required=True)
    labeling.set_defaults(func=label_interactively)

    analysis = sub.add_parser("analyze")
    analysis.add_argument("--manifest", required=True)
    analysis.add_argument("--labels-a", required=True)
    analysis.add_argument("--labels-b", required=True)
    analysis.add_argument("--adjudication")
    analysis.add_argument("--bootstrap-trials", type=int, default=10000)
    analysis.add_argument("--seed", type=int, default=0)
    analysis.add_argument("--out", required=True)
    analysis.set_defaults(func=analyze)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
