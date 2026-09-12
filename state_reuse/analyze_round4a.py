"""
Round 4a analysis (ROUND4A_PREREG.md): safe rate, paired loss, stale rate per condition;
A/B/C reading; drift at dependency-turn tokens. Writes <out>/ROUND4A_RESULTS.md.
"""
import sys, os, json, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1] if len(sys.argv) > 1 else "results/round4a"
rng = np.random.default_rng(0)
recs = [json.loads(l) for l in open(os.path.join(out, "records.jsonl"))]
CONDS = ["ctrl_other", "ctrl_name", "dep_number", "dep_move"]
full = {(r["qid"], r["p"], r["cond"]): r for r in recs if r["kind"] == "full"}
cond = collections.defaultdict(dict)
for r in recs:
    if r["kind"] in ("ours", "prefix"):
        cond[(r["qid"], r["p"], r["cond"])][r["kind"]] = r
positions = sorted({r["p"] for r in recs})


def boot(x, n=3000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    bs = rng.choice(x, (n, len(x))).mean(1)
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def fmt(ci, pp=False):
    m, lo, hi = ci
    if np.isnan(m):
        return "nan"
    return f"{m:+.1f} [{lo:+.1f}, {hi:+.1f}]" if pp else f"{m:.2f} [{lo:.2f}, {hi:.2f}]"


def keys_for(c, ps):
    return [k for k in full if k[2] == c and k[1] in ps and "ours" in cond.get(k, {})]


def stats(keys, which):
    safe = boot([cond[k][which]["agree_full"] for k in keys])
    gk = [k for k in keys if full[k]["correct"] is not None and cond[k][which]["correct"] is not None]
    loss = boot([100 * (int(cond[k][which]["correct"]) - int(full[k]["correct"])) for k in gk])
    accf = np.mean([full[k]["correct"] for k in gk]) if gk else np.nan
    accw = np.mean([cond[k][which]["correct"] for k in gk]) if gk else np.nan
    stale_w = np.mean([cond[k][which]["stale"] for k in keys])
    stale_f = np.mean([full[k]["same_as_old"] for k in keys])  # full == old (answer did not move)
    S = np.mean([cond[k][which]["compute_saved"] for k in keys])
    return safe, loss, accf, accw, len(gk), stale_w, stale_f, S


L = [f"# Round 4a results ({len(recs)} records)\n"]
L.append("safe = P(answer_ours == answer_full). stale(ours) = P(ours == old and full != old). "
         "full==old = share of cells where full recompute kept the pre-mutation answer.\n")
for ps, label in ([positions, "pooled over p"],) + tuple(([p], f"p = {p}%") for p in positions):
    L.append(f"\n## {label}\n")
    L.append("| condition | n | safe ours | safe prefix | acc full | acc ours | loss ours (pp) | n gold | stale ours | full==old | S_ours | S_prefix |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in CONDS:
        ks = keys_for(c, ps)
        if not ks:
            continue
        safe, loss, accf, accw, ng, st, sf, S = stats(ks, "ours")
        kp = [k for k in ks if "prefix" in cond[k]]  # a cell can lack prefix while the run is still writing
        safe_p = stats(kp, "prefix")[0] if kp else (np.nan, np.nan, np.nan)
        Sp = np.mean([cond[k]["prefix"]["compute_saved"] for k in kp]) if kp else np.nan
        L.append(f"| {c} | {len(ks)} | {fmt(safe)} | {fmt(safe_p)} | {accf:.2f} | {accw:.2f} | {fmt(loss, True)} | {ng} | {st:.2f} | {sf:.2f} | {S:.2f} | {Sp:.2f} |")

# ---- readings
L.append("\n## Preregistered reading\n")
kd = keys_for("dep_number", positions); ko = keys_for("ctrl_other", positions)
if kd and ko:
    sd = np.array([cond[k]["ours"]["agree_full"] for k in kd], float)
    so = np.array([cond[k]["ours"]["agree_full"] for k in ko], float)
    diff = boot(sd) [0] - boot(so)[0]
    bs = [rng.choice(sd, len(sd)).mean() - rng.choice(so, len(so)).mean() for _ in range(3000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    stale_o = np.mean([cond[k]["ours"]["stale"] for k in kd]); stale_f = np.mean([full[k]["same_as_old"] for k in kd])
    gk = [k for k in kd if full[k]["correct"] is not None]
    loss_dep = boot([100 * (int(cond[k]["ours"]["correct"]) - int(full[k]["correct"])) for k in gk])
    L.append(f"- safe(dep_number) - safe(ctrl_other) = {100*diff:+.1f} pp [{100*lo:+.1f}, {100*hi:+.1f}]")
    L.append(f"- stale(ours, dep_number) = {stale_o:.2f}; full kept old answer in {stale_f:.2f} of cells")
    L.append(f"- loss(dep_number, gold-eligible n={len(gk)}) = {fmt(loss_dep, True)} pp")
    ctrl_ok = {}
    for c in ("ctrl_other", "ctrl_name", "dep_move"):
        ks = keys_for(c, positions)
        if ks:
            _, l_, *_r = stats(ks, "ours")
            ctrl_ok[c] = (l_[0] >= -5 and l_[1] >= -10, l_)
            L.append(f"- loss({c}) = {fmt(l_, True)} pp -> {'ok' if ctrl_ok[c][0] else 'FAIL'}")
    A = (diff >= -0.10) and (stale_o <= stale_f + 0.10) and (np.isnan(loss_dep[0]) or loss_dep[0] >= -5)
    B = (diff <= -0.20) and (hi < 0) and all(v[0] for v in ctrl_ok.values())
    move_fail = ("dep_move" in ctrl_ok) and (not ctrl_ok["dep_move"][0]) and ctrl_ok.get("ctrl_other", (False,))[0]
    reading = "A (broadly tolerant)" if A else "B (dependency-sensitive boundary)" if B else "C (unstructured)"
    L.append(f"\n**Reading: {reading}**" + ("  -- dep_move FAILED while ctrl_other passed: Round 3 promotion retracted" if move_fail else ""))

# ---- drift at dependency tokens
L.append("\n## Step-0 drift (1 - cos) at mapped tokens: dependency-turn tokens vs other tokens\n")
L.append("| condition | cells | dep L16 | dep L32 | non-dep L16 | non-dep L32 |")
L.append("|---|---|---|---|---|---|")
for c in CONDS:
    ks = keys_for(c, positions)
    dep = [full[k]["drift0"]["dep"] for k in ks if full[k]["drift0"]["dep"]]
    nd = [full[k]["drift0"]["nondep"] for k in ks if full[k]["drift0"]["nondep"]]
    if ks:
        d = np.mean(dep, 0) if dep else np.full(33, np.nan); n_ = np.mean(nd, 0) if nd else np.full(33, np.nan)
        L.append(f"| {c} | {len(ks)} | {d[16]:.4f} | {d[32]:.4f} | {n_[16]:.4f} | {n_[32]:.4f} |")

# ---- figure
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
xs = np.arange(len(CONDS))
for i, which in enumerate(("ours", "prefix")):
    vals = []
    for c in CONDS:
        ks = keys_for(c, positions)
        v_ = [cond[k][which]["agree_full"] for k in ks if which in cond[k]]
        vals.append(np.mean(v_) if v_ else np.nan)
    ax[0].bar(xs + (i - 0.5) * 0.35, vals, 0.35, label=which)
ax[0].set_xticks(xs); ax[0].set_xticklabels(CONDS, rotation=15); ax[0].set_ylabel("safe rate (answer == full)"); ax[0].legend(); ax[0].set_ylim(0, 1)
vals = []
for c in CONDS:
    ks = keys_for(c, positions)
    gk = [k for k in ks if full[k]["correct"] is not None]
    vals.append(np.mean([100 * (int(cond[k]["ours"]["correct"]) - int(full[k]["correct"])) for k in gk]) if gk else np.nan)
ax[1].bar(xs, vals); ax[1].axhline(-5, color="r", ls="--"); ax[1].set_xticks(xs); ax[1].set_xticklabels(CONDS, rotation=15); ax[1].set_ylabel("paired accuracy loss ours vs full (pp)")
fig.suptitle("Round 4a: dependency vs control mutations"); fig.tight_layout()
fig.savefig(os.path.join(out, "fig_r4a.png"), dpi=130)

open(os.path.join(out, "ROUND4A_RESULTS.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
