"""
Turn-level audit helper for public agent trajectories.

The script fetches or reads SWE-agent-style trajectories, extracts each model generation turn, writes
a labeling queue, and summarizes either manual labels or heuristic triage labels.

Examples:
  python3 state_reuse/agent_trace_audit.py --source hf --n-trajectories 3 --out state_reuse/results/swe_audit_smoke
  python3 state_reuse/agent_trace_audit.py --source hf --sample-mode random --n-trajectories 100 --seed 0 --out state_reuse/results/swe_audit_main
  python3 state_reuse/agent_trace_audit.py --source local --input trajectories.jsonl --out state_reuse/results/swe_audit_local
  python3 state_reuse/agent_trace_audit.py --source local --input trajectories.jsonl --labels labels.csv --out state_reuse/results/swe_audit_labeled
"""
import argparse
import collections
import csv
import glob
import json
import math
import os
import random
import re
import socket
import sys
import time
import urllib.parse
import urllib.request
import urllib.error


HF_ROWS_API = "https://datasets-server.huggingface.co/rows"
DEFAULT_DATASET = "ElenaFu/SWE-agent-trajectories"


def estimate_tokens(text):
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def compact(text, limit=500):
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_local_rows(paths):
    for path in paths:
        if os.path.isdir(path):
            files = []
            for pat in ("**/*.traj", "**/*.json", "**/*.jsonl"):
                files.extend(glob.glob(os.path.join(path, pat), recursive=True))
            for row in iter_local_rows(sorted(files)):
                yield row
            continue

        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if line:
                        yield {"row_idx": i, "row": json.loads(line)}
            continue

        obj = read_json(path)
        if isinstance(obj, list):
            for i, row in enumerate(obj):
                yield {"row_idx": i, "row": row}
        else:
            yield {"row_idx": obj.get("row_idx", 0), "row": obj.get("row", obj)}


def fetch_hf_rows(dataset, config, split, offset, length, timeout, retries=3, retry_sleep=2.0):
    query = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    url = f"{HF_ROWS_API}?{query}"
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            break
        except (TimeoutError, socket.timeout, urllib.error.HTTPError, urllib.error.URLError):
            if attempt == retries:
                raise
            time.sleep(retry_sleep * (2 ** attempt))
    if "error" in payload:
        raise RuntimeError(payload["error"])
    return payload


def iter_hf_rows(args):
    if args.sample_mode == "random":
        rng = random.Random(args.seed)
        offsets = sorted(rng.sample(range(args.hf_total_rows), args.n_trajectories))
        for n, offset in enumerate(offsets, 1):
            try:
                payload = fetch_hf_rows(args.dataset, args.config, args.split, offset, 1, args.timeout, args.retries, args.retry_sleep)
            except Exception as exc:
                if not args.skip_fetch_errors:
                    raise
                print(f"skip offset {offset}: {exc}", file=sys.stderr)
                continue
            for item in payload["rows"]:
                yield item
            if args.sleep and n < len(offsets):
                time.sleep(args.sleep)
        return

    remaining = args.n_trajectories
    offset = args.offset
    while remaining > 0:
        length = min(args.page_size, remaining)
        try:
            payload = fetch_hf_rows(args.dataset, args.config, args.split, offset, length, args.timeout, args.retries, args.retry_sleep)
        except Exception as exc:
            if not args.skip_fetch_errors:
                raise
            print(f"skip offset {offset} length {length}: {exc}", file=sys.stderr)
            offset += length
            remaining -= length
            continue
        rows = payload["rows"]
        if not rows:
            break
        for item in rows:
            yield item
        got = len(rows)
        remaining -= got
        offset += got
        if got < length:
            break
        if args.sleep and remaining > 0:
            time.sleep(args.sleep)


def normalize_messages(row):
    traj = row.get("trajectory", [])
    if isinstance(traj, str):
        traj = json.loads(traj)
    if not isinstance(traj, list):
        return []

    messages = []
    if traj and isinstance(traj[0], dict) and "role" in traj[0]:
        for msg in traj:
            role = msg.get("role")
            text = msg.get("text")
            if text is None and role == "system":
                text = msg.get("system_prompt")
            messages.append({"role": role, "text": text or ""})
        return messages

    for step in traj:
        if not isinstance(step, dict):
            continue
        response = step.get("response") or step.get("output")
        if response is None:
            thought = step.get("thought", "")
            action = step.get("action", "")
            response = f"{thought}\n\n```\n{action}\n```".strip()
        messages.append({"role": "ai", "text": response or ""})
        if step.get("observation") is not None:
            messages.append({"role": "user", "text": step.get("observation") or ""})
    return messages


def parse_command(ai_text):
    blocks = re.findall(r"```(?:[A-Za-z0-9_+-]+)?\n(.*?)```", ai_text or "", flags=re.S)
    command = blocks[-1].strip() if blocks else ""
    if not command:
        m = re.search(r"\bCOMMAND\s*\n(.*)", ai_text or "", flags=re.S | re.I)
        if m:
            command = m.group(1).strip()
    first = command.splitlines()[0].strip() if command else ""
    kind = first.split()[0] if first else ""
    return command, kind


def command_family(kind):
    if not kind:
        return "none"
    if kind in {"open", "goto", "scroll_down", "scroll_up", "find_file", "search_dir", "search_file", "ls", "pwd"}:
        return "inspect"
    if kind in {"edit", "create"}:
        return "edit"
    if kind in {"pytest", "tox", "python", "python3", "npm", "cargo", "go", "mvn"}:
        return "test"
    if kind in {"submit"}:
        return "submit"
    return "shell"


def heuristic_label(ai_text, latest_observation, old_history, kind):
    text = (ai_text or "").lower()
    obs = latest_observation or ""
    family = command_family(kind)
    evidence = []

    if not (ai_text or "").strip():
        return "bookkeeping", "empty_ai_text"

    if re.search(r"malformat|syntax error|try again|same failed edit|proposed edit", obs, re.I):
        if family in {"edit", "inspect"}:
            return "safe_candidate", "latest_observation_is_repair_feedback"

    history_cues = [
        "earlier",
        "previously",
        "previous output",
        "previous command",
        "we saw",
        "we found",
        "we already",
        "as noted",
        "from before",
        "original issue",
        "the issue says",
        "the user reported",
        "combine",
        "now that we know",
    ]
    for cue in history_cues:
        if cue in text:
            evidence.append(cue)

    if family == "inspect":
        return "safe_candidate", "inspection_or_navigation_action"
    if family == "test" and re.search(r"test|failed|passed|error|traceback|syntax", obs, re.I):
        return "safe_candidate", "test_action_after_current_feedback"
    if family == "submit":
        if evidence:
            return "history_candidate", ",".join(evidence[:3])
        return "ambiguous", "submit_requires_solution_state"
    if family == "edit":
        if evidence or estimate_tokens(old_history) > 1000:
            return "history_candidate", ",".join(evidence[:3]) or "edit_after_long_history"
        return "ambiguous", "edit_without_history_cue"
    if evidence:
        return "history_candidate", ",".join(evidence[:3])
    return "ambiguous", f"command_family={family}"


def first_user_message_index(messages):
    for i, msg in enumerate(messages):
        if msg.get("role") == "user":
            return i
    return None


def extract_turns(row_idx, row, max_turns_per_trajectory=None):
    messages = normalize_messages(row)
    task_idx = first_user_message_index(messages)
    task_text = messages[task_idx]["text"] if task_idx is not None else ""
    ai_seen = 0
    for i, msg in enumerate(messages):
        if msg.get("role") not in {"ai", "assistant"}:
            continue
        ai_seen += 1
        if max_turns_per_trajectory and ai_seen > max_turns_per_trajectory:
            break

        latest_obs_idx = None
        for j in range(i - 1, -1, -1):
            if messages[j].get("role") == "user":
                latest_obs_idx = j
                break
        latest_observation = messages[latest_obs_idx]["text"] if latest_obs_idx is not None else ""
        old_parts = []
        old_parts_for_labeling = []
        for j, prev in enumerate(messages[:i]):
            if j == task_idx or j == latest_obs_idx or prev.get("role") == "system":
                continue
            text = prev.get("text", "")
            old_parts.append(text)
            role = "ASSISTANT" if prev.get("role") in {"ai", "assistant"} else "ENVIRONMENT OBSERVATION"
            old_parts_for_labeling.append(f"--- {role} ---\n{text}")
        old_history = "\n\n".join(old_parts)
        old_history_for_labeling = "\n\n".join(old_parts_for_labeling)
        context = "\n\n".join(m.get("text", "") for m in messages[:i])
        command, kind = parse_command(msg.get("text", ""))
        label, evidence = heuristic_label(msg.get("text", ""), latest_observation, old_history, kind)
        key = f"{row_idx}:{ai_seen}"
        yield {
            "key": key,
            "row_idx": row_idx,
            "instance_id": row.get("instance_id", ""),
            "model_name": row.get("model_name", ""),
            "target": row.get("target"),
            "exit_status": row.get("exit_status", ""),
            "turn_index": ai_seen,
            "message_index": i,
            "command_kind": kind,
            "command_family": command_family(kind),
            "heuristic_label": label,
            "heuristic_evidence": evidence,
            "context_tokens_est": estimate_tokens(context),
            "task_tokens_est": estimate_tokens(task_text),
            "latest_observation_tokens_est": estimate_tokens(latest_observation),
            "old_history_tokens_est": estimate_tokens(old_history),
            "cold_share_est": (estimate_tokens(old_history) / estimate_tokens(context)) if context else 0.0,
            "command": command,
            "ai_text": msg.get("text", ""),
            "latest_observation": latest_observation,
            "task_text": task_text,
            "old_history": old_history,
            "old_history_for_labeling": old_history_for_labeling,
            "old_history_preview": compact(old_history, 1000),
        }


def read_labels(path):
    if not path:
        return {}
    labels = {}
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                labels[row["key"]] = row.get("label") or row.get("manual_label")
        return labels
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = (row.get("label") or row.get("manual_label") or "").strip()
            if label:
                labels[row["key"]] = label
    return labels


def unwrap_dataset_item(item, fallback_idx):
    row_idx = item.get("row_idx", fallback_idx) if isinstance(item, dict) else fallback_idx
    row = item.get("row", item) if isinstance(item, dict) else item
    while isinstance(row, dict) and "trajectory" not in row and "row" in row:
        row_idx = row.get("row_idx", row_idx)
        row = row["row"]
    return row_idx, row


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_label_csv(path, turns, labels):
    fields = [
        "key",
        "manual_label",
        "heuristic_label",
        "heuristic_evidence",
        "instance_id",
        "target",
        "turn_index",
        "command_family",
        "command_kind",
        "context_tokens_est",
        "old_history_tokens_est",
        "cold_share_est",
        "command",
        "ai_text_snippet",
        "latest_observation_snippet",
        "old_history_preview",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for t in turns:
            w.writerow(
                {
                    "key": t["key"],
                    "manual_label": labels.get(t["key"], ""),
                    "heuristic_label": t["heuristic_label"],
                    "heuristic_evidence": t["heuristic_evidence"],
                    "instance_id": t["instance_id"],
                    "target": t["target"],
                    "turn_index": t["turn_index"],
                    "command_family": t["command_family"],
                    "command_kind": t["command_kind"],
                    "context_tokens_est": t["context_tokens_est"],
                    "old_history_tokens_est": t["old_history_tokens_est"],
                    "cold_share_est": f"{t['cold_share_est']:.3f}",
                    "command": compact(t["command"], 300),
                    "ai_text_snippet": compact(t["ai_text"], 700),
                    "latest_observation_snippet": compact(t["latest_observation"], 700),
                    "old_history_preview": t["old_history_preview"],
                }
            )


def summarize(turns, labels, n_trajectories):
    label_source = "manual" if labels else "heuristic"
    resolved = []
    for t in turns:
        label = labels.get(t["key"], t["heuristic_label"])
        if label == "safe_candidate":
            label = "safe" if not labels else label
        elif label == "history_candidate":
            label = "history_dependent" if not labels else label
        item = dict(t)
        item["label"] = label
        resolved.append(item)

    counts = collections.Counter(t["label"] for t in resolved)
    denom_labels = {"safe", "history_dependent", "ambiguous"}
    denom = sum(counts[k] for k in denom_labels)
    safe = counts["safe"]
    f_safe = safe / denom if denom else float("nan")
    total_ctx = sum(t["context_tokens_est"] for t in resolved if t["label"] in denom_labels)
    safe_cold = sum(t["old_history_tokens_est"] for t in resolved if t["label"] == "safe")
    saving = safe_cold / total_ctx if total_ctx else float("nan")

    by_family = collections.defaultdict(collections.Counter)
    by_target = collections.defaultdict(collections.Counter)
    for t in resolved:
        by_family[t["command_family"]][t["label"]] += 1
        by_target[str(t["target"])][t["label"]] += 1

    return {
        "label_source": label_source,
        "n_trajectories": n_trajectories,
        "n_turns": len(turns),
        "counts": dict(counts),
        "denominator": denom,
        "f_safe": f_safe,
        "oracle_compute_saving_est": saving,
        "total_context_tokens_est": total_ctx,
        "safe_cold_history_tokens_est": safe_cold,
        "by_family": {k: dict(v) for k, v in sorted(by_family.items())},
        "by_target": {k: dict(v) for k, v in sorted(by_target.items())},
    }


def fmt_pct(x):
    if x != x:
        return "nan"
    return f"{100 * x:.1f}%"


def write_summary_md(path, summary, args):
    lines = [
        "# Agent trace audit summary",
        "",
        f"- source: `{args.source}`",
        f"- label source: `{summary['label_source']}`",
        f"- trajectories fetched/read: {summary['n_trajectories']}",
        f"- generation turns: {summary['n_turns']}",
        f"- denominator excluding bookkeeping: {summary['denominator']}",
        f"- f_safe: **{fmt_pct(summary['f_safe'])}**",
        f"- oracle compute saving estimate: **{fmt_pct(summary['oracle_compute_saving_est'])}**",
        f"- context tokens estimate in denominator: {summary['total_context_tokens_est']}",
        f"- safe cold-history tokens estimate: {summary['safe_cold_history_tokens_est']}",
        "",
        "Labels are primary only when `label source` is `manual`. Heuristic output is triage.",
        "",
        "## Counts",
        "",
        "| label | count |",
        "|---|---:|",
    ]
    if args.source == "hf":
        lines.insert(5, f"- requested trajectories: {args.n_trajectories}")
        lines.insert(6, f"- sample mode: `{args.sample_mode}`")
    else:
        lines.insert(5, f"- local input files/directories: {len(args.input)}")
    for label, count in sorted(summary["counts"].items()):
        lines.append(f"| {label} | {count} |")

    lines.extend(["", "## By Command Family", "", "| command_family | safe | history_dependent | ambiguous | bookkeeping | other |", "|---|---:|---:|---:|---:|---:|"])
    for family, ctr in summary["by_family"].items():
        other = sum(v for k, v in ctr.items() if k not in {"safe", "history_dependent", "ambiguous", "bookkeeping"})
        lines.append(
            f"| {family} | {ctr.get('safe', 0)} | {ctr.get('history_dependent', 0)} | "
            f"{ctr.get('ambiguous', 0)} | {ctr.get('bookkeeping', 0)} | {other} |"
        )

    lines.extend(["", "## By Target", "", "| target | safe | history_dependent | ambiguous | bookkeeping | other |", "|---|---:|---:|---:|---:|---:|"])
    for target, ctr in summary["by_target"].items():
        other = sum(v for k, v in ctr.items() if k not in {"safe", "history_dependent", "ambiguous", "bookkeeping"})
        lines.append(
            f"| {target} | {ctr.get('safe', 0)} | {ctr.get('history_dependent', 0)} | "
            f"{ctr.get('ambiguous', 0)} | {ctr.get('bookkeeping', 0)} | {other} |"
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["hf", "local"], default="hf")
    ap.add_argument("--input", nargs="*", default=[], help="local .json/.jsonl/.traj files or directories")
    ap.add_argument("--labels", default=None, help="optional CSV/JSONL with key,label or key,manual_label")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results", "swe_agent_audit"))
    ap.add_argument("--max-turns-per-trajectory", type=int, default=0)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--config", default="default")
    ap.add_argument("--split", default="train")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--n-trajectories", type=int, default=20)
    ap.add_argument("--sample-mode", choices=["contiguous", "random"], default="contiguous")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hf-total-rows", type=int, default=80036)
    ap.add_argument("--page-size", type=int, default=10)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--retry-sleep", type=float, default=2.0)
    ap.add_argument("--skip-fetch-errors", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.0)
    args = ap.parse_args()

    if args.source == "local" and not args.input:
        ap.error("--source local requires --input")

    os.makedirs(args.out, exist_ok=True)
    labels = read_labels(args.labels)
    max_turns = args.max_turns_per_trajectory or None
    row_iter = iter_hf_rows(args) if args.source == "hf" else iter_local_rows(args.input)

    raw_rows = []
    turns = []
    for item in row_iter:
        row_idx, row = unwrap_dataset_item(item, len(raw_rows))
        raw_rows.append({"row_idx": row_idx, "row": row})
        turns.extend(extract_turns(row_idx, row, max_turns_per_trajectory=max_turns))

    write_jsonl(os.path.join(args.out, "trajectories_sample.jsonl"), raw_rows)
    write_jsonl(os.path.join(args.out, "turns.jsonl"), turns)
    write_label_csv(os.path.join(args.out, "turns_for_labeling.csv"), turns, labels)
    summary = summarize(turns, labels, len(raw_rows))
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    write_summary_md(os.path.join(args.out, "SUMMARY.md"), summary, args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
