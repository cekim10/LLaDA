"""
Aggregate a pilot run into the three prospective figures + summary tables.

  python plot.py results/pilot
Outputs (in the same dir):
  fig1_layer_by_delta.png     cos-sim of old vs new hidden state, layer x delta type
  fig2_reuse_frontier.png     quality vs. compute saved for the layer-reuse sweep
  fig3_locality.png           drift vs. distance from the update, per layer band
  fig4_step_by_layer.png      drift over diffusion steps, per layer band
  summary.md                  tables
"""
import glob, json, os, sys
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1]
recs = [json.loads(l) for l in open(os.path.join(out, "records.jsonl"))]
deltas = sorted({r["delta"] for r in recs if "delta" in r})

# ------------------------------------------------------------ Fig 1 / 3 / 4
sims_by_delta = defaultdict(list)   # delta -> list of [T, L+1, n]
dist_by_delta = defaultdict(list)
for f in glob.glob(os.path.join(out, "sims_*.npz")):
    d = np.load(f)
    dname = os.path.basename(f).split("_", 2)[2][:-4]
    sims_by_delta[dname].append(d["sims"].astype(np.float32))
    dist_by_delta[dname].append(d["dist"])

L1 = next(iter(sims_by_delta.values()))[0].shape[1]
fig, ax = plt.subplots(figsize=(8, 0.5 * len(deltas) + 1.5))
mat = np.array([[np.mean([s[:, l].mean() for s in sims_by_delta[d]]) for l in range(L1)] for d in deltas])
im = ax.imshow(mat, aspect="auto", vmin=0.5, vmax=1.0, cmap="viridis")
ax.set_yticks(range(len(deltas))); ax.set_yticklabels(deltas)
ax.set_xlabel("layer (residual stream entering block l; last = final output)")
ax.set_title("Fig1: mean cos-sim of shared-position hidden states, old vs new request")
plt.colorbar(im, ax=ax); plt.tight_layout(); plt.savefig(os.path.join(out, "fig1_layer_by_delta.png"), dpi=150)

# Fig 3: drift vs distance from the nearest changed token, per layer band
bands = {"shallow (0-7)": range(0, 8), "middle (8-19)": range(8, 20), "deep (20-32)": range(20, L1)}
fig, axes = plt.subplots(1, len(deltas), figsize=(4 * len(deltas), 3.5), sharey=True)
axes = np.atleast_1d(axes)
for ax, d in zip(axes, deltas):
    for bname, band in bands.items():
        xs, ys = [], []
        for s, dist in zip(sims_by_delta[d], dist_by_delta[d]):
            drift = 1 - s[:, list(band), :].mean(axis=(0, 1))  # per position
            xs.append(dist); ys.append(drift)
        xs = np.concatenate(xs); ys = np.concatenate(ys)
        bins = np.unique(np.quantile(xs, np.linspace(0, 1, 9)).astype(int))
        idx = np.digitize(xs, bins) - 1
        cx = [xs[idx == i].mean() for i in range(len(bins)) if (idx == i).any()]
        cy = [ys[idx == i].mean() for i in range(len(bins)) if (idx == i).any()]
        ax.plot(cx, cy, marker="o", label=bname)
    ax.set_title(d); ax.set_xlabel("distance (tokens) from nearest changed token")
axes[0].set_ylabel("1 - cos sim"); axes[0].legend(fontsize=8)
fig.suptitle("Fig3: spatial locality of state drift"); plt.tight_layout()
plt.savefig(os.path.join(out, "fig3_locality.png"), dpi=150)

# Fig 4: drift over diffusion step, per layer band
fig, axes = plt.subplots(1, len(deltas), figsize=(4 * len(deltas), 3.5), sharey=True)
axes = np.atleast_1d(axes)
for ax, d in zip(axes, deltas):
    for bname, band in bands.items():
        curve = np.mean([1 - s[:, list(band), :].mean(axis=(1, 2)) for s in sims_by_delta[d]], axis=0)
        ax.plot(curve, label=bname)
    ax.set_title(d); ax.set_xlabel("diffusion step")
axes[0].set_ylabel("1 - cos sim"); axes[0].legend(fontsize=8)
fig.suptitle("Fig4: state drift across denoising steps"); plt.tight_layout()
plt.savefig(os.path.join(out, "fig4_step_by_layer.png"), dpi=150)

# ------------------------------------------------------------ Fig 2 / tables
lines = ["# Pilot summary", ""]
old = [r for r in recs if r["kind"] == "old"]
lines.append(f"old requests: {len(old)}, accuracy {np.mean([r['correct'] for r in old]):.3f}")
lines.append("")
lines.append("## Full recompute of P+D")
lines.append("| delta | n | acc (gold known) | answer same as old |")
lines.append("|---|---|---|---|")
for d in deltas:
    rs = [r for r in recs if r["kind"] == "new_full" and r["delta"] == d]
    acc = [r["correct"] for r in rs if r["correct"] is not None]
    lines.append(f"| {d} | {len(rs)} | {np.mean(acc) if acc else float('nan'):.3f} | {np.mean([r['same_as_old'] for r in rs]):.2f} |")

reuse = [r for r in recs if r["kind"] == "reuse"]
if reuse:
    sources = sorted({r["source"] for r in reuse})
    ks = sorted({r["k"] for r in reuse})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for d in deltas:
        for src in sources:
            rows = []
            for k in ks:
                rs = [r for r in reuse if r["delta"] == d and r["k"] == k and (r["source"] == src or k == 0)]
                if not rs:
                    continue
                rows.append((np.mean([r["compute_saved"] for r in rs]),
                             np.mean([r["agree_full"] for r in rs]),
                             np.mean([r["tok_agree"] for r in rs]), k))
            rows = np.array(rows)
            ls = "-" if src == "aligned" else "--"
            axes[0].plot(rows[:, 0] * 100, rows[:, 1] * 100, ls, marker="o", label=f"{d} [{src}]")
            axes[1].plot(rows[:, 0] * 100, rows[:, 2] * 100, ls, marker="o", label=f"{d} [{src}]")
    axes[0].set_xlabel("% transformer compute reused"); axes[0].set_ylabel("% final answers equal to full recompute")
    axes[1].set_xlabel("% transformer compute reused"); axes[1].set_ylabel("% generated tokens equal to full recompute")
    axes[0].legend(fontsize=7); fig.suptitle("Fig2: reuse frontier (layers 0..k-1 of old request reused at shared positions)")
    plt.tight_layout(); plt.savefig(os.path.join(out, "fig2_reuse_frontier.png"), dpi=150)

    lines += ["", "## Layer-reuse sweep (Exp2)", "",
              "| delta | source | k | compute saved | answer==full | tok agree | acc (gold known) |",
              "|---|---|---|---|---|---|---|"]
    for d in deltas:
        for src in sources:
            for k in ks:
                rs = [r for r in reuse if r["delta"] == d and r["k"] == k and (r["source"] == src or k == 0)]
                if not rs:
                    continue
                acc = [r["correct"] for r in rs if r["correct"] is not None]
                lines.append(f"| {d} | {src} | {k} | {np.mean([r['compute_saved'] for r in rs]):.2f} | "
                             f"{np.mean([r['agree_full'] for r in rs]):.2f} | {np.mean([r['tok_agree'] for r in rs]):.2f} | "
                             f"{np.mean(acc) if acc else float('nan'):.2f} |")

res = [r for r in recs if r["kind"] == "resume"]
if res:
    lines += ["", "## Canvas resume (Exp5)", "", "| delta | t0 | forwards | answer==full | tok agree | acc |", "|---|---|---|---|---|---|"]
    for d in deltas:
        for t0 in sorted({r["t0"] for r in res}):
            rs = [r for r in res if r["delta"] == d and r["t0"] == t0]
            if not rs:
                continue
            acc = [r["correct"] for r in rs if r["correct"] is not None]
            ta = [r["tok_agree"] for r in rs if r["tok_agree"] is not None]
            lines.append(f"| {d} | {t0} | {np.mean([r['n_forward'] for r in rs]):.1f} | {np.mean([r['agree_full'] for r in rs]):.2f} | "
                         f"{np.mean(ta) if ta else float('nan'):.2f} | {np.mean(acc) if acc else float('nan'):.2f} |")

lines += ["", "## Mean cos-sim by layer (Fig1 data)", "", "| delta | " + " | ".join(f"L{l}" for l in range(0, L1, 4)) + " |",
          "|---|" + "---|" * len(range(0, L1, 4))]
for i, d in enumerate(deltas):
    lines.append(f"| {d} | " + " | ".join(f"{mat[i, l]:.3f}" for l in range(0, L1, 4)) + " |")
open(os.path.join(out, "summary.md"), "w").write("\n".join(lines))
print("\n".join(lines))
