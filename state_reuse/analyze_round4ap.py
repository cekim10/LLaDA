"""
Round 4a' analysis (ROUND4AP_PREREG.md). Writes <out>/ROUND4AP_RESULTS.md and fig_r4ap.png.
"""
import sys, os, json, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1] if len(sys.argv) > 1 else "results/round4ap"
rng = np.random.default_rng(0)
recs = [json.loads(l) for l in open(os.path.join(out, "records.jsonl"))]
CONDS = ["identity", "all_but_dep", "dep_only", "dep_fresh_num"]
full = {(r["qid"], r["p"]): r for r in recs if r["kind"] == "full"}
cond = collections.defaultdict(dict)
for r in recs:
    if r["kind"] in CONDS:
        cond[(r["qid"], r["p"])][r["kind"]] = r
positions = sorted({r["p"] for r in recs})


def boot(x, n=3000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    bs = rng.choice(x, (n, len(x))).mean(1)
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def fmt(ci):
    m, lo, hi = ci
    return "nan" if np.isnan(m) else f"{m:+.1f} [{lo:+.1f}, {hi:+.1f}]"


def keys(c, ps):
    return [k for k in full if k[1] in ps and c in cond.get(k, {})]


def loss(c, ps):
    ks = keys(c, ps)
    return boot([100 * (int(cond[k][c]["correct"]) - int(full[k]["correct"])) for k in ks]), ks


L = [f"# Round 4a' results ({len(recs)} records; no mutation, P' = P)\n"]
L.append(f"full-recompute accuracy: {np.mean([r['correct'] for r in full.values()]):.2f} (n={len(full)}); "
         f"mean generated tokens full: {np.mean([r['n_gen'] for r in full.values()]):.0f} / 256\n")
for ps, label in ([positions, "pooled over p"],) + tuple(([p], f"p = {p}%") for p in positions):
    L.append(f"\n## {label}\n")
    L.append("| condition | n | frozen frac | acc full | acc cond | loss (pp) | agree full | tok agree | gen tokens |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for c in CONDS:
        l_, ks = loss(c, ps)
        if not ks:
            continue
        L.append(f"| {c} | {len(ks)} | {np.mean([cond[k][c]['frozen_frac'] for k in ks]):.2f} | "
                 f"{np.mean([full[k]['correct'] for k in ks]):.2f} | {np.mean([cond[k][c]['correct'] for k in ks]):.2f} | {fmt(l_)} | "
                 f"{np.mean([cond[k][c]['agree_full'] for k in ks]):.2f} | {np.mean([cond[k][c]['tok_agree'] for k in ks]):.2f} | "
                 f"{np.mean([cond[k][c]['n_gen'] for k in ks]):.0f} |")

L.append("\n## Preregistered gate (identity, pooled)\n")
li, ks = loss("identity", positions)
if ks:
    if li[0] <= -15:
        verdict = "mutation-independent freezing failure exists (H2 supported)"
    elif li[1] > -5:
        verdict = "pure freezing does not explain Round 4a (H1 side)"
    else:
        verdict = "INCONCLUSIVE"
    L.append(f"- loss(identity) pooled = {fmt(li)} pp (n={len(ks)}) -> **{verdict}**")
    l10, _ = loss("identity", [10]); l90, _ = loss("identity", [90])
    L.append(f"- replication cell p=10: {fmt(l10)}; p=90: {fmt(l90)}")
    if verdict == "INCONCLUSIVE" and l10[0] <= -15 and l90[1] > -5:
        L.append("- recorded as: distance-implicated, inconclusive on the pooled gate")
    la, _ = loss("all_but_dep", positions); ld, _ = loss("dep_only", positions); lf, _ = loss("dep_fresh_num", positions)
    localized = (la[0] >= -5 and la[1] >= -10) and (ld[0] <= -10)
    L.append(f"\n## Localization (secondary)\n")
    L.append(f"- loss(all_but_dep) = {fmt(la)}; loss(dep_only) = {fmt(ld)} -> **{'localized' if localized else 'not localized'}**"
             + ("" if verdict != "INCONCLUSIVE" else " (primary gate inconclusive: not interpreted)"))
    L.append(f"- diagnostic dep_fresh_num = {fmt(lf)} (not used in decisions)")

# paired differences between conditions (same cells), pooled
L.append("\n## Paired differences between conditions (pp, pooled)\n")
ks_all = [k for k in full if all(c in cond.get(k, {}) for c in CONDS)]
for a, b in (("identity", "all_but_dep"), ("identity", "dep_only"), ("dep_only", "all_but_dep"), ("identity", "dep_fresh_num")):
    d = boot([100 * (int(cond[k][a]["correct"]) - int(cond[k][b]["correct"])) for k in ks_all])
    L.append(f"- {a} - {b}: {fmt(d)}")
L.append(f"\nFormat diagnostics: outputs at >= 250 gen tokens: full {np.mean([r['n_gen'] >= 250 for r in full.values()]):.2f}; "
         + ", ".join(f"{c} {np.mean([cond[k][c]['n_gen'] >= 250 for k in ks_all]):.2f}" for c in CONDS))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for i, p in enumerate(positions):
    vals = [loss(c, [p])[0][0] for c in CONDS]
    ax[0].bar(np.arange(len(CONDS)) + (i - (len(positions) - 1) / 2) * 0.25, vals, 0.25, label=f"p={p}%")
ax[0].set_xticks(range(len(CONDS))); ax[0].set_xticklabels(CONDS, rotation=15); ax[0].axhline(-5, color="r", ls="--"); ax[0].axhline(-15, color="k", ls=":")
ax[0].set_ylabel("paired accuracy loss vs full (pp)"); ax[0].legend()
vals = [loss(c, positions)[0] for c in CONDS]
ax[1].bar(range(len(CONDS)), [v[0] for v in vals], yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]], capsize=4)
ax[1].set_xticks(range(len(CONDS))); ax[1].set_xticklabels(CONDS, rotation=15); ax[1].axhline(-5, color="r", ls="--"); ax[1].axhline(-15, color="k", ls=":")
ax[1].set_ylabel("pooled loss (pp), 95% CI")
fig.suptitle("Round 4a': freezing without mutation"); fig.tight_layout(); fig.savefig(os.path.join(out, "fig_r4ap.png"), dpi=130)
open(os.path.join(out, "ROUND4AP_RESULTS.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
