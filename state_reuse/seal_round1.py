"""
Round-1 seal analysis: bootstrap CIs over prompts for the preregistered primary
metrics (answer accuracy vs gold, answer agreement with full recompute) and the
secondary metric (generated-token agreement); locality slope; step convergence.
Writes <out>/ROUND1_SEAL.md.  Analysis only - does not touch pilot code.
"""
import sys, os, json, glob, collections
import numpy as np

out = sys.argv[1] if len(sys.argv) > 1 else "results/pilot_gpu"
rng = np.random.default_rng(0)
recs = [json.loads(l) for l in open(os.path.join(out, "records.jsonl"))]
old = {r["qid"]: r for r in recs if r["kind"] == "old"}
full = {(r["qid"], r["delta"]): r for r in recs if r["kind"] == "new_full"}
deltas = ["irrelevant_append", "hint_append", "name_edit", "multiturn_append", "number_edit"]
PRESERVING = deltas[:4]


def boot_ci(x, n=2000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    m = x.mean()
    bs = rng.choice(x, (n, len(x)), replace=True).mean(1)
    return m, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def fmt(ci):
    m, lo, hi = ci
    return f"{m:.2f} [{lo:.2f},{hi:.2f}]" if not np.isnan(m) else "nan"


L = []
L.append("# Round 1 seal (n=50 GSM8K prompts, LLaDA-8B-Instruct, gen 128, 64 steps)\n")
L.append(f"records: {collections.Counter(r['kind'] for r in recs)}\n")
acc_old = boot_ci([r["correct"] for r in old.values()])
L.append(f"old-request accuracy: {fmt(acc_old)}\n")

# ---------------- full recompute of P+D ------------------------------------
L.append("## Full recompute of P+D (reference)\n")
L.append("| delta | n | acc | answer same as old |")
L.append("|---|---|---|---|")
for d in deltas:
    rs = [r for (q, dd), r in full.items() if dd == d]
    acc = boot_ci([r["correct"] for r in rs if r["correct"] is not None])
    same = boot_ci([r["same_as_old"] for r in rs])
    L.append(f"| {d} | {len(rs)} | {fmt(acc)} | {fmt(same)} |")

# ---------------- Exp1/3/4 from sims files ---------------------------------
L.append("\n## Exp1: hidden-state drift at shared positions (mean cos-sim)\n")
L.append("| delta | n | L0 | L8 | L16 | L24 | L32 | L32 first-step | L32 last-step |")
L.append("|---|---|---|---|---|---|---|---|---|")
loc = {}
for d in deltas:
    files = sorted(glob.glob(os.path.join(out, f"sims_q*_{d}.npz")))
    per_layer, first, last, slopes = [], [], [], []
    for f in files:
        z = np.load(f)
        s = z["sims"].astype(np.float32)  # [T, L+1, n]
        per_layer.append(s.mean((0, 2)))
        first.append(s[0, -1].mean()); last.append(s[-1, -1].mean())
        dist = z["dist"].astype(float)
        drift_deep = 1 - s[:, 20:, :].mean((0, 1))  # per position
        if len(dist) > 5 and dist.std() > 0:
            slopes.append(np.corrcoef(dist, drift_deep)[0, 1])
    pl = np.array(per_layer)
    loc[d] = slopes
    L.append(f"| {d} | {len(files)} | " + " | ".join(f"{pl[:, l].mean():.3f}" for l in (0, 8, 16, 24, 32))
             + f" | {np.mean(first):.3f} | {np.mean(last):.3f} |")

L.append("\n## Exp3: spatial locality (Pearson r between distance-to-change and deep-layer drift, per prompt)\n")
L.append("| delta | mean r [95% CI] | frac prompts with r < -0.3 |")
L.append("|---|---|---|")
for d in deltas:
    s = np.array(loc[d])
    if len(s):
        L.append(f"| {d} | {fmt(boot_ci(s))} | {(s < -0.3).mean():.2f} |")
L.append("\nNegative r = drift decays with distance from the change (locality). r near 0 = no spatial locality.")

# ---------------- Exp2 -------------------------------------------------------
L.append("\n## Exp2: layer-state reuse, primary metrics with 95% bootstrap CI over prompts\n")
L.append("Quality loss = acc(k) - acc(full) in percentage points, paired per prompt (gold known only).\n")
reuse = collections.defaultdict(list)
for r in recs:
    if r["kind"] == "reuse":
        reuse[(r["delta"], r["source"], r["k"])].append(r)
ks = sorted({k for (_, _, k) in reuse})
sources = sorted({s for (_, s, _) in reuse})
for src in sources:
    L.append(f"\n### source = {src}\n")
    L.append("| delta | k | saved | acc(k) | quality loss (pp) | answer==full | tok agree |")
    L.append("|---|---|---|---|---|---|---|")
    for d in deltas:
        for k in ks:
            rs = reuse.get((d, src, k)) or reuse.get((d, sources[0], k)) if k == 0 else reuse.get((d, src, k))
            if not rs:
                continue
            gk = [r for r in rs if r["correct"] is not None]
            acc = boot_ci([r["correct"] for r in gk])
            loss = boot_ci([100 * (int(r["correct"]) - int(full[(r["qid"], d)]["correct"])) for r in gk]) if gk else (np.nan,) * 3
            agree = boot_ci([r["agree_full"] for r in rs])
            tok = boot_ci([r["tok_agree"] for r in rs])
            saved = np.mean([r["compute_saved"] for r in rs])
            L.append(f"| {d} | {k} | {saved:.2f} | {fmt(acc)} | {loss[0]:+.1f} [{loss[1]:+.1f},{loss[2]:+.1f}] | {fmt(agree)} | {fmt(tok)} |")

# pooled over preserving deltas
L.append("\n### Pooled over answer-preserving deltas (irrelevant, hint, name, multiturn)\n")
L.append("| source | k | saved | acc(k) | quality loss (pp) | answer==full | tok agree | n |")
L.append("|---|---|---|---|---|---|---|---|")
for src in sources:
    for k in ks:
        rs = []
        for d in PRESERVING:
            rs += reuse.get((d, src, k)) or (reuse.get((d, sources[0], k)) if k == 0 else []) or []
        if not rs:
            continue
        gk = [r for r in rs if r["correct"] is not None]
        acc = boot_ci([r["correct"] for r in gk])
        loss = boot_ci([100 * (int(r["correct"]) - int(full[(r["qid"], r["delta"])]["correct"])) for r in gk])
        agree = boot_ci([r["agree_full"] for r in rs]); tok = boot_ci([r["tok_agree"] for r in rs])
        L.append(f"| {src} | {k} | {np.mean([r['compute_saved'] for r in rs]):.2f} | {fmt(acc)} | {loss[0]:+.1f} [{loss[1]:+.1f},{loss[2]:+.1f}] | {fmt(agree)} | {fmt(tok)} | {len(rs)} |")

# ---------------- Exp5 -------------------------------------------------------
L.append("\n## Exp5: canvas resume from old request's partial canvas\n")
L.append("| delta | t0 | forwards | acc | quality loss (pp) | answer==full | tok agree | n |")
L.append("|---|---|---|---|---|---|---|---|")
res = collections.defaultdict(list)
for r in recs:
    if r["kind"] == "resume":
        res[(r["delta"], r["t0"])].append(r)
for d in deltas:
    for t0 in sorted({t for (_, t) in res}):
        rs = res.get((d, t0))
        if not rs:
            continue
        gk = [r for r in rs if r["correct"] is not None]
        acc = boot_ci([r["correct"] for r in gk])
        loss = boot_ci([100 * (int(r["correct"]) - int(full[(r["qid"], d)]["correct"])) for r in gk]) if gk else (np.nan,) * 3
        agree = boot_ci([r["agree_full"] for r in rs]); tok = boot_ci([r["tok_agree"] for r in rs])
        L.append(f"| {d} | {t0} | {np.mean([r['n_forward'] for r in rs]):.0f} | {fmt(acc)} | {loss[0]:+.1f} [{loss[1]:+.1f},{loss[2]:+.1f}] | {fmt(agree)} | {fmt(tok)} | {len(rs)} |")

# does the resumed answer just equal the OLD answer?
L.append("\nResume output vs OLD request's answer (is resume just keeping the old answer?):\n")
L.append("| delta | t0 | answer==old | answer==full | full==old |")
L.append("|---|---|---|---|---|")
for d in deltas:
    for t0 in sorted({t for (_, t) in res}):
        rs = res.get((d, t0))
        if not rs:
            continue
        eq_old = np.mean([r["pred"] == old[r["qid"]]["pred"] for r in rs])
        eq_full = np.mean([r["agree_full"] for r in rs])
        f_old = np.mean([full[(r["qid"], d)]["same_as_old"] for r in rs])
        L.append(f"| {d} | {t0} | {eq_old:.2f} | {eq_full:.2f} | {f_old:.2f} |")

open(os.path.join(out, "ROUND1_SEAL.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
