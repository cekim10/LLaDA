"""
Request-level reuse gate evaluation (GATE_PREREG.md).

Offline, zero GPU. Uses the 400 adjudicated SWE-agent turns: extracts the frozen causal feature
set from (turn_index, task_text, old_history, latest_observation), fits an L2 logistic regression
under trajectory-grouped cross-validation, picks the operating point on training folds only, and
reports out-of-fold S_gate / unsafe_reuse_share with a cluster bootstrap, against the mandatory
baselines.

  python gate_eval.py --turns results/swe_agent_audit_random100/turns.jsonl \
      --manifest results/swe_agent_manual400/sample_manifest.jsonl \
      --resolved results/swe_agent_manual400/analysis/resolved_labels.jsonl \
      --out results/swe_agent_gate
"""
import argparse, collections, json, math, os, random, re, sys

import numpy as np

# ---------------------------------------------------------------- features
ERROR_RE = re.compile(r"command not found|Traceback|No such file|not recognized|error:|Error:|ERROR|failed|Failed", re.I)
EMPTY_OK_RE = re.compile(r"ran successfully and did not produce any output", re.I)
FILE_VIEW_RE = re.compile(r"\[File: ")
EDIT_REJECT_RE = re.compile(r"Your changes have NOT been applied")
WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
PATH_RE = re.compile(r"[\w/\.-]+\.(?:py|txt|yaml|yml|json|cfg|toml|md|rst|sh)\b")
OBS_SPLIT_RE = re.compile(r"--- ENVIRONMENT OBSERVATION ---")
ASSISTANT_RE = re.compile(r"--- ASSISTANT ---")

FEATURES = [
    "turn_index", "log_hist_tokens", "log_obs_tokens", "cold_share",
    "obs_is_error", "obs_is_empty_success", "obs_is_file_view", "obs_edit_rejected",
    "jaccard_obs_task", "jaccard_obs_hist", "obs_novel_rate", "obs_is_repeat",
    "n_prior_assistant_turns", "obs_new_path_rate",
]


def _norm(text):
    return re.sub(r"\s+", " ", text).strip()


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def extract_features(turn):
    """Causal features only: turn_index, task_text, old_history, latest_observation and counts.
    Never ai_text / command* / heuristic* / exit_status / target (see GATE_PREREG.md)."""
    obs = turn.get("latest_observation", "") or ""
    task = turn.get("task_text", "") or ""
    hist = turn.get("old_history", "") or ""
    obs_w = set(WORD_RE.findall(obs.lower()))
    task_w = set(WORD_RE.findall(task.lower()))
    hist_w = set(WORD_RE.findall(hist.lower()))
    obs_paths = set(PATH_RE.findall(obs))
    task_paths = set(PATH_RE.findall(task))
    prior_obs = {_norm(chunk)[:400] for chunk in OBS_SPLIT_RE.split(hist)[1:]}
    ctx = max(1, turn.get("context_tokens_est", 1))
    return {
        "turn_index": float(turn.get("turn_index", 0)),
        "log_hist_tokens": math.log1p(turn.get("old_history_tokens_est", 0)),
        "log_obs_tokens": math.log1p(turn.get("latest_observation_tokens_est", 0)),
        "cold_share": turn.get("old_history_tokens_est", 0) / ctx,
        "obs_is_error": float(bool(ERROR_RE.search(obs))),
        "obs_is_empty_success": float(bool(EMPTY_OK_RE.search(obs))),
        "obs_is_file_view": float(bool(FILE_VIEW_RE.search(obs))),
        "obs_edit_rejected": float(bool(EDIT_REJECT_RE.search(obs))),
        "jaccard_obs_task": jaccard(obs_w, task_w),
        "jaccard_obs_hist": jaccard(obs_w, hist_w),
        "obs_novel_rate": (len(obs_w - task_w - hist_w) / len(obs_w)) if obs_w else 0.0,
        "obs_is_repeat": float(_norm(obs)[:400] in prior_obs),
        "n_prior_assistant_turns": float(len(ASSISTANT_RE.findall(hist))),
        "obs_new_path_rate": (len(obs_paths - task_paths) / len(obs_paths)) if obs_paths else 0.0,
    }


# ---------------------------------------------------------------- model
def fit_logreg(X, y, w, l2=1.0, iters=400, lr=0.5):
    """Class-balanced L2 logistic regression by full-batch gradient ascent on the weighted
    log-likelihood. Small deterministic model; no external dependency."""
    n, d = X.shape
    pos, neg = w[y == 1].sum(), w[y == 0].sum()
    cw = np.where(y == 1, 0.5 / max(pos, 1e-9), 0.5 / max(neg, 1e-9)) * w
    cw = cw / cw.sum() * n
    beta = np.zeros(d + 1)
    Xb = np.hstack([np.ones((n, 1)), X])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ beta, -30, 30)))
        g = Xb.T @ (cw * (y - p)) / n
        g[1:] -= l2 * beta[1:] / n
        beta += lr * g
    return beta


def predict(beta, X):
    Xb = np.hstack([np.ones((X.shape[0], 1)), X])
    return 1.0 / (1.0 + np.exp(-np.clip(Xb @ beta, -30, 30)))


# ---------------------------------------------------------------- metrics
def gate_metrics(reuse, safe, w, cold, ctx):
    """S_gate and unsafe_reuse_share per GATE_PREREG.md, over the rows passed in
    (callers pass non-bookkeeping rows only)."""
    denom = float((w * ctx).sum())
    total_w = float(w.sum())
    s = float((w * cold * reuse).sum()) / denom if denom else float("nan")
    unsafe = float((w * reuse * (~safe)).sum()) / total_w if total_w else float("nan")
    return s, unsafe


def pick_threshold(scores, safe, w, budget):
    """Most permissive threshold (lowest cutoff) whose unsafe-reuse share stays within budget.
    Evaluated on training rows only."""
    total_w = w.sum()
    order = np.argsort(-scores)
    best = 1.01
    cum_unsafe = 0.0
    for i in order:
        cum_unsafe += w[i] * (not safe[i])
        if cum_unsafe / total_w > budget:
            break
        best = scores[i]
    return best


def cluster_bootstrap(groups, fn, trials, seed):
    uniq = sorted(set(groups))
    idx_by_g = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = random.Random(seed)
    out = collections.defaultdict(list)
    for _ in range(trials):
        take = np.concatenate([idx_by_g[g] for g in rng.choices(uniq, k=len(uniq))])
        for k, v in fn(take).items():
            out[k].append(v)
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in out.items()}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--resolved", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=float, default=0.10)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap-trials", type=int, default=10000)
    ap.add_argument("--ambiguous-safe", action="store_true", help="optimistic bound")
    args = ap.parse_args()

    manifest = {json.loads(l)["item_id"]: json.loads(l) for l in open(args.manifest)}
    labels = {json.loads(l)["item_id"]: json.loads(l)["final_label"] for l in open(args.resolved)}
    want_keys = {m["key"]: i for i, m in manifest.items()}

    rows = []
    with open(args.turns) as f:
        for line in f:
            t = json.loads(line)
            item = want_keys.get(t["key"])
            if item is None:
                continue
            m = manifest[item]
            lab = labels[item]
            feats = extract_features(t)
            rows.append(dict(
                item_id=item, group=m["row_idx"], label=lab, w=m["sampling_weight"],
                cold=m["old_history_tokens_est"], ctx=m["context_tokens_est"],
                heuristic=t.get("heuristic_label", ""), **feats))
    assert len(rows) == len(manifest), f"matched {len(rows)} of {len(manifest)} sampled turns"

    sub = [r for r in rows if r["label"] != "bookkeeping"]
    X = np.array([[r[f] for f in FEATURES] for r in sub], float)
    safe = np.array([r["label"] == "safe" or (args.ambiguous_safe and r["label"] == "ambiguous") for r in sub])
    w = np.array([r["w"] for r in sub], float)
    cold = np.array([r["cold"] for r in sub], float)
    ctx = np.array([r["ctx"] for r in sub], float)
    groups = np.array([r["group"] for r in sub])
    n = len(sub)
    print(f"non-bookkeeping turns: {n} (safe {int(safe.sum())}), trajectories: {len(set(groups))}", flush=True)

    # ---- out-of-fold predictions, grouped CV, threshold chosen on training folds only
    uniq = sorted(set(groups))
    oof_score = np.zeros(n)
    oof_reuse = np.zeros(n, bool)
    counts = np.zeros(n)
    reuse_votes = np.zeros(n)
    for rep in range(args.repeats):
        rng = random.Random(args.seed + rep)
        shuffled = uniq[:]
        rng.shuffle(shuffled)
        fold_of = {g: i % args.folds for i, g in enumerate(shuffled)}
        assign = np.array([fold_of[g] for g in groups])
        for k in range(args.folds):
            tr, te = assign != k, assign == k
            if te.sum() == 0 or safe[tr].sum() in (0, tr.sum()):
                continue
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            beta = fit_logreg((X[tr] - mu) / sd, safe[tr].astype(float), w[tr])
            s_tr = predict(beta, (X[tr] - mu) / sd)
            thr = pick_threshold(s_tr, safe[tr], w[tr], args.budget)
            s_te = predict(beta, (X[te] - mu) / sd)
            oof_score[te] += s_te
            reuse_votes[te] += (s_te >= thr)
            counts[te] += 1
    oof_score /= np.maximum(counts, 1)
    oof_reuse = (reuse_votes / np.maximum(counts, 1)) >= 0.5

    def metrics_on(idx, reuse):
        s, u = gate_metrics(reuse[idx], safe[idx], w[idx], cold[idx], ctx[idx])
        return {"S_gate": s, "unsafe_reuse_share": u}

    results = {}
    results["gate"] = metrics_on(np.arange(n), oof_reuse)
    ci = cluster_bootstrap(groups, lambda idx: metrics_on(idx, oof_reuse), args.bootstrap_trials, args.seed)
    results["gate"].update({"S_gate_ci": ci["S_gate"], "unsafe_ci": ci["unsafe_reuse_share"]})
    results["gate"]["reuse_rate"] = float((w * oof_reuse).sum() / w.sum())

    # ---- oracle and trivial baselines
    results["oracle"] = metrics_on(np.arange(n), safe)
    results["always_reuse"] = metrics_on(np.arange(n), np.ones(n, bool))
    results["never_reuse"] = metrics_on(np.arange(n), np.zeros(n, bool))
    rng = np.random.default_rng(args.seed)
    rr = results["gate"]["reuse_rate"]
    rnd = [metrics_on(np.arange(n), rng.random(n) < rr) for _ in range(200)]
    results["random_matched"] = {k: float(np.mean([r[k] for r in rnd])) for k in ("S_gate", "unsafe_reuse_share")}

    # ---- single-feature threshold rules, same budget, same grouped CV discipline
    singles = {}
    for j, fname in enumerate(FEATURES):
        for sign in (1.0, -1.0):
            v = sign * X[:, j]
            votes = np.zeros(n); cnt = np.zeros(n)
            for rep in range(args.repeats):
                rng2 = random.Random(args.seed + rep)
                sh = uniq[:]; rng2.shuffle(sh)
                fold_of = {g: i % args.folds for i, g in enumerate(sh)}
                assign = np.array([fold_of[g] for g in groups])
                for k in range(args.folds):
                    tr, te = assign != k, assign == k
                    if te.sum() == 0:
                        continue
                    thr = pick_threshold(v[tr], safe[tr], w[tr], args.budget)
                    votes[te] += (v[te] >= thr); cnt[te] += 1
            reuse = (votes / np.maximum(cnt, 1)) >= 0.5
            m = metrics_on(np.arange(n), reuse)
            key = f"{'+' if sign > 0 else '-'}{fname}"
            singles[key] = m
    results["single_feature"] = singles

    # ---- value-only baseline: rank by cold tokens alone, no safety model at all.
    # If this matches the D1 gate, the safety model contributes nothing and the "gate" is just
    # "reuse the turns with the most history".
    votes = np.zeros(n); cnt = np.zeros(n)
    for rep in range(args.repeats):
        rng2 = random.Random(args.seed + rep)
        sh = uniq[:]; rng2.shuffle(sh)
        fold_of = {g_: i % args.folds for i, g_ in enumerate(sh)}
        assign = np.array([fold_of[g_] for g_ in groups])
        for k in range(args.folds):
            tr, te = assign != k, assign == k
            if te.sum() == 0:
                continue
            thr = pick_threshold(cold[tr], safe[tr], w[tr], args.budget)
            votes[te] += (cold[te] >= thr); cnt[te] += 1
    cold_only = (votes / np.maximum(cnt, 1)) >= 0.5
    results["value_only_cold_rank"] = metrics_on(np.arange(n), cold_only)
    results["value_only_cold_rank"]["reuse_rate"] = float((w * cold_only).sum() / w.sum())

    # ---- illegal reference: heuristic label reads the generation
    heur = np.array([r["heuristic"] == "safe_candidate" for r in sub])
    results["ILLEGAL_heuristic_label"] = metrics_on(np.arange(n), heur)

    # ---- deviation D1: value/risk ranked gate (see GATE_PREREG.md deviations log).
    # The preregistered gate thresholds P(safe) alone, which ignores that a reused turn is worth
    # its cold-history tokens. The system-correct ranking for "maximise saving subject to an
    # unsafe budget" is a fractional knapsack on cold_i / (1 - p_i). Same grouped CV, same
    # training-only threshold selection, same budget.
    votes = np.zeros(n); cnt = np.zeros(n)
    for rep in range(args.repeats):
        rng2 = random.Random(args.seed + rep)
        sh = uniq[:]; rng2.shuffle(sh)
        fold_of = {g_: i % args.folds for i, g_ in enumerate(sh)}
        assign = np.array([fold_of[g_] for g_ in groups])
        for k in range(args.folds):
            tr, te = assign != k, assign == k
            if te.sum() == 0 or safe[tr].sum() in (0, tr.sum()):
                continue
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            beta = fit_logreg((X[tr] - mu) / sd, safe[tr].astype(float), w[tr])
            p_tr = predict(beta, (X[tr] - mu) / sd)
            p_te = predict(beta, (X[te] - mu) / sd)
            r_tr = cold[tr] / np.maximum(1.0 - p_tr, 1e-3)
            r_te = cold[te] / np.maximum(1.0 - p_te, 1e-3)
            thr = pick_threshold(r_tr, safe[tr], w[tr], args.budget)
            votes[te] += (r_te >= thr); cnt[te] += 1
    vr_reuse = (votes / np.maximum(cnt, 1)) >= 0.5
    results["gate_value_risk_D1"] = metrics_on(np.arange(n), vr_reuse)
    ci2 = cluster_bootstrap(groups, lambda idx: metrics_on(idx, vr_reuse), args.bootstrap_trials, args.seed)
    results["gate_value_risk_D1"].update({"S_gate_ci": ci2["S_gate"], "unsafe_ci": ci2["unsafe_reuse_share"]})
    results["gate_value_risk_D1"]["reuse_rate"] = float((w * vr_reuse).sum() / w.sum())

    # ---- feasible frontier: best S reachable on each out-of-fold ranking within budget
    def sweep_best(rank):
        order_ = np.argsort(-rank)
        cum_u, best = 0.0, 0.0
        mask = np.zeros(n, bool)
        for i in order_:
            cum_u += w[i] * (not safe[i])
            if cum_u / w.sum() > args.budget:
                break
            mask[i] = True
            best = max(best, gate_metrics(mask, safe, w, cold, ctx)[0])
        return best
    results["best_feasible_S_pfe_ranking"] = sweep_best(oof_score)
    results["best_feasible_S_value_risk_ranking"] = sweep_best(cold / np.maximum(1.0 - oof_score, 1e-3))
    results["best_feasible_S_oracle_ranking"] = sweep_best(np.where(safe, 1e9, 0) + cold)

    # ---- decision
    g = results["gate"]
    go = g["S_gate"] >= 0.20 and g["unsafe_reuse_share"] <= args.budget and g["S_gate_ci"][0] >= 0.15
    best_s = max(results["best_feasible_S_pfe_ranking"], results["best_feasible_S_value_risk_ranking"])
    results["best_feasible_S_on_oof_sweep"] = best_s
    results["decision"] = "GO" if go else ("STOP" if best_s < 0.15 else "INCONCLUSIVE")

    os.makedirs(args.out, exist_ok=True)
    tag = "optimistic" if args.ambiguous_safe else "conservative"
    with open(os.path.join(args.out, f"gate_{tag}.json"), "w") as f:
        json.dump(results, f, indent=2)

    def pct(x):
        return "nan" if x != x else f"{100 * x:.1f}%"
    print(f"\n=== {tag} bound (ambiguous counted {'safe' if args.ambiguous_safe else 'unsafe'}) ===")
    print(f"{'rule':34s} {'S_gate':>9s} {'unsafe':>8s}")
    print(f"{'oracle (knows the labels)':34s} {pct(results['oracle']['S_gate']):>9s} {pct(results['oracle']['unsafe_reuse_share']):>8s}")
    print(f"{'always reuse':34s} {pct(results['always_reuse']['S_gate']):>9s} {pct(results['always_reuse']['unsafe_reuse_share']):>8s}")
    print(f"{'random @ matched rate':34s} {pct(results['random_matched']['S_gate']):>9s} {pct(results['random_matched']['unsafe_reuse_share']):>8s}")
    print(f"{'LEARNED GATE (out-of-fold)':34s} {pct(g['S_gate']):>9s} {pct(g['unsafe_reuse_share']):>8s}"
          f"   CI [{pct(g['S_gate_ci'][0])}, {pct(g['S_gate_ci'][1])}]  reuse rate {pct(g['reuse_rate'])}")
    v = results["gate_value_risk_D1"]
    print(f"{'D1 value/risk gate (out-of-fold)':34s} {pct(v['S_gate']):>9s} {pct(v['unsafe_reuse_share']):>8s}"
          f"   CI [{pct(v['S_gate_ci'][0])}, {pct(v['S_gate_ci'][1])}]  reuse rate {pct(v['reuse_rate'])}")
    c_ = results["value_only_cold_rank"]
    print(f"{'value-only (rank by cold tokens)':34s} {pct(c_['S_gate']):>9s} {pct(c_['unsafe_reuse_share']):>8s}"
          f"   reuse rate {pct(c_['reuse_rate'])}")
    print(f"{'(illegal) heuristic on generation':34s} {pct(results['ILLEGAL_heuristic_label']['S_gate']):>9s} {pct(results['ILLEGAL_heuristic_label']['unsafe_reuse_share']):>8s}")
    print("\ntop single-feature rules by S_gate within budget:")
    ok = [(k, v) for k, v in singles.items() if v["unsafe_reuse_share"] <= args.budget + 1e-9]
    for k, v in sorted(ok, key=lambda kv: -kv[1]["S_gate"])[:8]:
        print(f"  {k:32s} {pct(v['S_gate']):>9s} {pct(v['unsafe_reuse_share']):>8s}")
    print(f"\nbest feasible S within budget, by ranking:  P(safe) {pct(results['best_feasible_S_pfe_ranking'])}"
          f" | value/risk {pct(results['best_feasible_S_value_risk_ranking'])}"
          f" | oracle ranking {pct(results['best_feasible_S_oracle_ranking'])}")
    print(f"DECISION: {results['decision']}")


if __name__ == "__main__":
    main()
