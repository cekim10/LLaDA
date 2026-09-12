"""
Round 4a'' analysis (ROUND4APP_PREREG.md). Primary: D = Q(all_but_dep) - Q(identity) on the new
prompts. Writes <out>/ROUND4APP_RESULTS.md and fig_r4app.png. Optional second arg: path of the
Round 4a' results dir for the labeled pooled secondary estimate.
"""
import sys, os, json, collections, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1] if len(sys.argv) > 1 else "results/round4app"
prev = sys.argv[2] if len(sys.argv) > 2 else None
rng = np.random.default_rng(0)


def load(d):
    recs = [json.loads(l) for l in open(os.path.join(d, "records.jsonl"))]
    full = {(r["qid"], r["p"]): r for r in recs if r["kind"] == "full"}
    cond = collections.defaultdict(dict)
    for r in recs:
        if r["kind"] in ("identity", "all_but_dep"):
            cond[(r["qid"], r["p"])][r["kind"]] = r
    keys = [k for k in full if "identity" in cond.get(k, {}) and "all_but_dep" in cond.get(k, {})]
    return recs, full, cond, sorted(keys)


def boot(x, n=4000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    bs = rng.choice(x, (n, len(x))).mean(1)
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def fmt(ci):
    m, lo, hi = ci
    return "nan" if np.isnan(m) else f"{m:+.1f} [{lo:+.1f}, {hi:+.1f}]"


def D_stats(full, cond, keys):
    d = boot([100 * (int(cond[k]["all_but_dep"]["correct"]) - int(cond[k]["identity"]["correct"])) for k in keys])
    li = boot([100 * (int(cond[k]["identity"]["correct"]) - int(full[k]["correct"])) for k in keys])
    la = boot([100 * (int(cond[k]["all_but_dep"]["correct"]) - int(full[k]["correct"])) for k in keys])
    return d, li, la


recs, full, cond, keys = load(out)
positions = sorted({k[1] for k in keys})
L = [f"# Round 4a'' results ({len(recs)} records, {len(keys)} complete cells, prompts {min(k[0] for k in keys)}-{max(k[0] for k in keys)})\n"]
L.append(f"full-recompute accuracy {np.mean([full[k]['correct'] for k in keys]):.2f}; identity {np.mean([cond[k]['identity']['correct'] for k in keys]):.2f}; "
         f"all_but_dep {np.mean([cond[k]['all_but_dep']['correct'] for k in keys]):.2f}\n")

d, li, la = D_stats(full, cond, keys)
L.append("## Primary: D = Q(all_but_dep) - Q(identity), paired, new prompts only\n")
L.append(f"- D = **{fmt(d)} pp** (n={len(keys)})")
confirmed = d[0] >= 5 and d[1] > 0
kill = (d[1] <= 0 <= d[2] and d[0] < 5)
verdict = "CONFIRMED (block liveness: GO)" if confirmed else ("KILL (localization unstable)" if kill else "NOT CONFIRMED")
L.append(f"- **{verdict}**")
L.append("\n## Secondary\n")
L.append(f"- Q(identity) - Q(full) = {fmt(li)} pp  (freezing cost; legacy -15 criterion {'met' if li[0] <= -15 else 'not met'})")
L.append(f"- Q(all_but_dep) - Q(full) = {fmt(la)} pp  (3% live block vs full recompute)")
for p in positions:
    kp = [k for k in keys if k[1] == p]
    dp, lip, lap = D_stats(full, cond, kp)
    L.append(f"- p={p}%: D = {fmt(dp)}, identity-full = {fmt(lip)}, all_but_dep-full = {fmt(lap)} (n={len(kp)})")
if prev and os.path.exists(os.path.join(prev, "records.jsonl")):
    _, f2, c2, k2 = load(prev)
    k2 = [k for k in k2 if k[1] in positions]  # same positions only
    fullP = {**{("prev",) + k: f2[k] for k in k2}, **{("new",) + k: full[k] for k in keys}}
    condP = {**{("prev",) + k: c2[k] for k in k2}, **{("new",) + k: cond[k] for k in keys}}
    dP, liP, laP = D_stats(fullP, condP, list(fullP))
    L.append(f"\nPooled 4a' (p in {positions}) + 4a'' (secondary, labeled): D = {fmt(dP)}, identity-full = {fmt(liP)}, all_but_dep-full = {fmt(laP)} (n={len(fullP)})")

# attention logging summary (descriptive only)
att = [r for r in recs if r["kind"] == "attn0"]
if att:
    L.append("\n## Passive attention log (descriptive; no decision uses this)\n")
    layers = att[0]["layers"]
    share = np.array([r["dep_share_of_ctx"] for r in att])
    frac = np.mean([r["dep_frac_tokens"] for r in att])
    L.append("| layer | " + " | ".join(str(l) for l in layers) + " |")
    L.append("|---|" + "---|" * len(layers))
    L.append("| problem block share of canvas->context attention (mean) | " + " | ".join(f"{v:.3f}" for v in share.mean(0)) + " |")
    L.append(f"\nproblem block = {frac:.3f} of context tokens; share/frac ratio by layer: " + ", ".join(f"{v/frac:.1f}x" for v in share.mean(0)))

fig, ax = plt.subplots(figsize=(6, 4))
vals = [li, la, d]
ax.bar(range(3), [v[0] for v in vals], yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]], capsize=4)
ax.set_xticks(range(3)); ax.set_xticklabels(["identity - full", "all_but_dep - full", "D = all_but_dep - identity"], rotation=10)
ax.axhline(0, color="k", lw=0.8); ax.axhline(5, color="g", ls="--"); ax.set_ylabel("pp, 95% CI"); ax.set_title("Round 4a'' (unseen prompts)")
fig.tight_layout(); fig.savefig(os.path.join(out, "fig_r4app.png"), dpi=130)
open(os.path.join(out, "ROUND4APP_RESULTS.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
