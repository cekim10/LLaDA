"""
Round 3 analysis (ROUND3_PREREG.md): mutation/relocation tolerance of step-0 state reuse.
Tables per (mutation, p): acc_full / acc_ours / acc_prefix, paired loss with 95% CI,
agreement, S_ours, S_prefix, gap; gate on pooled {edit,insert,delete} at p<=50.
Figures: compute saved vs p (ours vs prefix); loss vs p; step-0 drift per layer pre vs post.
Writes <out>/ROUND3_RESULTS.md.
"""
import sys, os, json, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1] if len(sys.argv) > 1 else "results/round3"
rng = np.random.default_rng(0)
recs = [json.loads(l) for l in open(os.path.join(out, "records.jsonl"))]
TYPES = ["edit", "insert", "delete", "reorder"]
GATE_TYPES = ["edit", "insert", "delete"]

full = {(r["qid"], r["ctx"], r["mtype"], r["p"]): r for r in recs if r["kind"] == "full"}
cond = collections.defaultdict(dict)
for r in recs:
    if r["kind"] in ("ours", "prefix"):
        cond[(r["qid"], r["ctx"], r["mtype"], r["p"])][r["kind"]] = r
ctxs = sorted({r["ctx"] for r in recs})
positions = sorted({r["p"] for r in recs if "p" in r})


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


def cells(ctx, types, ps):
    return [key for key in full if key[1] == ctx and key[2] in types and key[3] in ps and "ours" in cond.get(key, {})]


def loss_list(keys, which):
    return [100 * (int(cond[k][which]["correct"]) - int(full[k]["correct"])) for k in keys if which in cond[k]]


L = [f"# Round 3 results ({len(recs)} records)\n"]
for ctx in ctxs:
    L.append(f"\n## ctx {ctx}\n")
    L.append("| mutation | p% | n | acc full | acc ours | acc prefix | loss ours (pp) | loss prefix (pp) | agree ours | S_ours | S_prefix | gap | shifted tokens |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    curve = collections.defaultdict(dict)
    for t in TYPES:
        for p in positions:
            keys = cells(ctx, [t], [p])
            if not keys:
                continue
            af = np.mean([full[k]["correct"] for k in keys])
            ao = np.mean([cond[k]["ours"]["correct"] for k in keys])
            apf = np.mean([cond[k]["prefix"]["correct"] for k in keys if "prefix" in cond[k]]) if any("prefix" in cond[k] for k in keys) else np.nan
            lo_ = boot(loss_list(keys, "ours")); lp_ = boot(loss_list(keys, "prefix"))
            ag = np.mean([cond[k]["ours"]["agree_full"] for k in keys])
            So = np.mean([full[k]["S_ours"] for k in keys]); Sp = np.mean([full[k]["S_prefix"] for k in keys])
            nsh = np.mean([full[k]["n_shifted"] for k in keys])
            curve[t][p] = (So, Sp, lo_[0], lp_[0])
            L.append(f"| {t} | {p} | {len(keys)} | {af:.2f} | {ao:.2f} | {apf:.2f} | {fmt(lo_, True)} | {fmt(lp_, True)} | {ag:.2f} | {So:.2f} | {Sp:.2f} | {So-Sp:+.2f} | {nsh:.0f} |")

    # gate
    gk = cells(ctx, GATE_TYPES, [p for p in positions if p <= 50])
    if gk:
        lo_ = boot(loss_list(gk, "ours"))
        So = np.mean([full[k]["S_ours"] for k in gk]); Sp = np.mean([full[k]["S_prefix"] for k in gk])
        q_ok = (lo_[0] >= -5.0) and (lo_[1] >= -10.0)
        h_ok = (So - Sp) >= 0.30
        L.append(f"\n### Gate (pooled edit+insert+delete, p<=50, n={len(gk)})\n")
        L.append(f"- quality: loss ours = {fmt(lo_, True)} pp -> {'PASS' if q_ok else 'FAIL'} (need point >= -5, CI low >= -10)")
        L.append(f"- headroom: S_ours - S_prefix = {So:.2f} - {Sp:.2f} = {So-Sp:+.2f} -> {'PASS' if h_ok else 'FAIL'} (need >= +0.30)")
        L.append(f"- **{'PASS' if (q_ok and h_ok) else 'FAIL'}**")
        for t in GATE_TYPES:
            kk = cells(ctx, [t], [p for p in positions if p <= 50])
            if kk:
                L.append(f"  - {t}: loss {fmt(boot(loss_list(kk, 'ours')), True)}, gap {np.mean([full[k]['S_ours']-full[k]['S_prefix'] for k in kk]):+.2f}")
        # stuck on old answer
        L.append("\nStuck-on-old-answer rate (ours vs full): " + ", ".join(
            f"{t} {np.mean([cond[k]['ours']['same_as_old'] for k in cells(ctx,[t],positions)]):.2f}/"
            f"{np.mean([full[k]['same_as_old'] for k in cells(ctx,[t],positions)]):.2f}" for t in TYPES if cells(ctx, [t], positions)))

    # drift: relocated (shifted) vs unshifted mapped tokens, from the sims files (exact per-token split)
    import glob, re
    L.append("\n### Step-0 drift (1 - cos) at mapped positions: relocated (old pos != new pos) vs unshifted tokens, cells averaged\n")
    L.append("| mutation | set | cells | L8 | L16 | L24 | L32 |")
    L.append("|---|---|---|---|---|---|---|")
    drift_fig = {}
    for t in TYPES:
        acc = {"unshifted": [], "relocated": []}
        for f in glob.glob(os.path.join(out, f"sims0_q*_c{ctx}_{t}_p*.npz")):
            z = np.load(f); s_ = z["sims0"].astype(np.float32); sh = z["old_idx"] != z["new_idx"]
            if (~sh).any(): acc["unshifted"].append(1 - s_[:, ~sh].mean(1))
            if sh.any(): acc["relocated"].append(1 - s_[:, sh].mean(1))
        for name, v in acc.items():
            if v:
                m = np.mean(v, 0); drift_fig[(t, name)] = m
                L.append(f"| {t} | {name} | {len(v)} | {m[8]:.4f} | {m[16]:.4f} | {m[24]:.4f} | {m[32]:.4f} |")

    # figures
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    for t in TYPES:
        if t not in curve:
            continue
        ps = sorted(curve[t]); So = [curve[t][p][0] for p in ps]; Sp = [curve[t][p][1] for p in ps]
        ax[0].plot(ps, [100 * s for s in So], "-o", label=f"ours {t}")
        ax[0].plot(ps, [100 * s for s in Sp], "--x", label=f"prefix {t}")
        ax[1].plot(ps, [curve[t][p][2] for p in ps], "-o", label=f"ours {t}")
        ax[1].plot(ps, [curve[t][p][3] for p in ps], "--x", label=f"prefix {t}")
    ax[0].set_xlabel("mutation position (% of history)"); ax[0].set_ylabel("% compute saved (FLOP proxy)"); ax[0].set_title("reuse vs mutation position"); ax[0].legend(fontsize=7)
    ax[1].set_xlabel("mutation position (% of history)"); ax[1].set_ylabel("accuracy loss vs full (pp)"); ax[1].axhline(-5, color="r", ls="--"); ax[1].set_title("quality vs mutation position"); ax[1].legend(fontsize=7)
    for (t, name), m in drift_fig.items():
        ax[2].plot(range(33), m, "-" if name == "unshifted" else "--", label=f"{t} {name}")
    ax[2].set_xlabel("layer"); ax[2].set_ylabel("1 - cos (step 0)"); ax[2].set_title("state drift: unshifted vs relocated tokens"); ax[2].legend(fontsize=7)
    fig.suptitle(f"Round 3, ctx {ctx}"); fig.tight_layout()
    fig.savefig(os.path.join(out, f"fig_r3_ctx{ctx}.png"), dpi=130); plt.close(fig)

open(os.path.join(out, "ROUND3_RESULTS.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
