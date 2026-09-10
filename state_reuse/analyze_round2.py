"""
Round 2 analysis: oracle reuse headroom and decision structure.

  python analyze_round2.py results/round2 [--tol 2.0]

Definitions (ROUND2_PREREG.md, Amendment A1):
  saved(k)            = analytic transformer-compute fraction saved (recorded per run)
  class oracle        = per (ctx, delta): largest k such that pooled accuracy loss vs full
                        recompute is >= -tol pp for every k' <= k   (gold-known deltas only)
  uniform policy      = one k for all deltas at a given ctx that satisfies the constraint in
                        every delta class (min over class oracles); the delta-blind baseline
  prompt oracle       = per (prompt, ctx, delta): largest k such that the answer equals the
                        full-recompute answer for every k' <= k  (all deltas, no gold needed)
  S_x                 = mean saved under policy x
Simple policies map a cheap feature (|D|, lexical overlap, embedding cosine, step-0 drift at
shallow layers) to k with quantile bins, fit with leave-one-prompt-out CV under the same
pooled-loss constraint; report S_simple, realized loss, and S_oracle - S_simple.
Also works on Round 1 records (ctx = 0, source 'first'); features are then absent.
"""
import sys, os, json, argparse, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--tol", type=float, default=2.0, help="max pooled accuracy loss (pp)")
ap.add_argument("--source", default="first")
args = ap.parse_args()
out, TOL = args.out, args.tol
rng = np.random.default_rng(0)

recs = [json.loads(l) for l in open(os.path.join(out, "records.jsonl"))]
for r in recs:
    r.setdefault("ctx", 0)
full = {(r["qid"], r["ctx"], r["delta"]): r for r in recs if r["kind"] == "new_full"}
old = {(r["qid"], r["ctx"]): r for r in recs if r["kind"] == "old"}
reuse = collections.defaultdict(dict)  # (qid, ctx, delta) -> {k: rec}
for r in recs:
    if r["kind"] == "reuse" and r.get("source", "first") == args.source and r["k"] > 0:
        reuse[(r["qid"], r["ctx"], r["delta"])][r["k"]] = r
ctxs = sorted({c for (_, c, _) in reuse})
deltas = [d for d in ["irrelevant_append", "hint_append", "name_edit", "multiturn_append", "number_edit"]
          if any(dd == d for (_, _, dd) in reuse)]
ks = sorted({k for v in reuse.values() for k in v})
PRES = [d for d in deltas if d != "number_edit"]
L = []


def boot_ci(x, n=2000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    bs = rng.choice(x, (n, len(x)), replace=True).mean(1)
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def keys_for(ctx, ds, qids=None):
    return [key for key in reuse if key[1] == ctx and key[2] in ds and (qids is None or key[0] in qids)
            and full[key]["correct"] is not None]


def loss_pp(keyset, k):
    """pooled accuracy loss at k (pp) over keys; k=0 -> 0."""
    if k == 0:
        return 0.0
    d = [100 * (int(reuse[key][k]["correct"]) - int(full[key]["correct"])) for key in keyset if k in reuse[key]]
    return float(np.mean(d)) if d else np.nan


def saved_at(keyset, k):
    if k == 0:
        return 0.0
    s = [reuse[key][k]["compute_saved"] for key in keyset if k in reuse[key]]
    return float(np.mean(s)) if s else np.nan


def safe_k(keyset, tol=TOL):
    """largest k with loss >= -tol for all k' <= k (monotone closure)."""
    best = 0
    for k in ks:
        l = loss_pp(keyset, k)
        if np.isnan(l) or l < -tol:
            break
        best = k
    return best


# ------------------------------------------------------------------ reference
L.append(f"# Round 2 analysis (tol = {TOL} pp, source = {args.source})\n")
L.append("## Reference accuracy (full recompute of P+D) by shared-context length\n")
L.append("| ctx | ctx tokens | old acc | " + " | ".join(deltas) + " |")
L.append("|---|---|---|" + "---|" * len(deltas))
for c in ctxs:
    toks = np.mean([r.get("ctx_tokens", 0) for (q, cc), r in old.items() if cc == c])
    oa = np.mean([r["correct"] for (q, cc), r in old.items() if cc == c])
    cells = []
    for d in deltas:
        rs = [r for (q, cc, dd), r in full.items() if cc == c and dd == d and r["correct"] is not None]
        cells.append(f"{np.mean([r['correct'] for r in rs]):.2f} (n={len(rs)})" if rs else "n/a")
    L.append(f"| {c} | {toks:.0f} | {oa:.2f} | " + " | ".join(cells) + " |")

# ------------------------------------------------------------------ loss vs k
L.append("\n## Pooled accuracy loss (pp) vs k, answer-preserving deltas, per ctx\n")
L.append("| ctx | " + " | ".join(f"k={k}" for k in ks) + " |")
L.append("|---|" + "---|" * len(ks))
loss_tab = {}
for c in ctxs:
    ks_ = keys_for(c, PRES)
    loss_tab[c] = [loss_pp(ks_, k) for k in ks]
    L.append(f"| {c} | " + " | ".join(f"{v:+.1f}" for v in loss_tab[c]) + " |")
L.append("\nCompute saved at each k (mean over preserving deltas):\n")
L.append("| ctx | " + " | ".join(f"k={k}" for k in ks) + " |")
L.append("|---|" + "---|" * len(ks))
saved_tab = {}
for c in ctxs:
    ks_ = keys_for(c, PRES)
    saved_tab[c] = [saved_at(ks_, k) for k in ks]
    L.append(f"| {c} | " + " | ".join(f"{v:.2f}" for v in saved_tab[c]) + " |")

# ------------------------------------------------------------------ oracles
L.append(f"\n## Oracle headroom (class oracle: per-delta safe k under loss >= -{TOL} pp, monotone)\n")
L.append("| ctx | " + " | ".join(f"k* {d}" for d in PRES) + " | S_uniform (k) | S_class_oracle | S_prompt_oracle (all deltas) | S_prompt_oracle (preserving) |")
L.append("|---|" + "---|" * (len(PRES) + 4))
S = collections.defaultdict(dict)
prompt_k = {}  # key -> prompt-oracle k (agreement-monotone)
for key, kv in reuse.items():
    best = 0
    for k in ks:
        if k not in kv or not kv[k]["agree_full"]:
            break
        best = k
    prompt_k[key] = best
for c in ctxs:
    cells, s_class, n_class = [], 0.0, 0
    for d in PRES:
        ks_ = keys_for(c, [d])
        if not ks_:
            cells.append("n/a"); continue
        kstar = safe_k(ks_)
        cells.append(str(kstar))
        s_class += saved_at(ks_, kstar) * len(ks_); n_class += len(ks_)
    S[c]["class"] = s_class / max(1, n_class)
    # uniform policy: one k for every delta class, safe for the WORST class (min of class k*)
    ku = min([safe_k(keys_for(c, [d])) for d in PRES if keys_for(c, [d])] or [0])
    S[c]["uniform"] = saved_at(keys_for(c, PRES), ku)
    S[c]["uniform_k"] = ku
    pk_all = [reuse[key][prompt_k[key]]["compute_saved"] if prompt_k[key] > 0 else 0.0 for key in reuse if key[1] == c]
    pk_pres = [reuse[key][prompt_k[key]]["compute_saved"] if prompt_k[key] > 0 else 0.0 for key in reuse if key[1] == c and key[2] in PRES]
    S[c]["prompt_all"] = float(np.mean(pk_all)); S[c]["prompt_pres"] = float(np.mean(pk_pres))
    L.append(f"| {c} | " + " | ".join(cells) + f" | {S[c]['uniform']:.2f} ({ku}) | {S[c]['class']:.2f} | {S[c]['prompt_all']:.2f} | {S[c]['prompt_pres']:.2f} |")
L.append("\nS_class_oracle - S_uniform = value of knowing the delta class; S_prompt_oracle - S_class_oracle = value of per-prompt choice (upper bound, agreement-based).")

# number_edit diagnostics
if "number_edit" in deltas:
    L.append("\n## D_changing diagnostics (number_edit): agreement with full and stuck-on-old-answer rate\n")
    L.append("| ctx | " + " | ".join(f"k={k}" for k in ks) + " |")
    L.append("|---|" + "---|" * len(ks))
    for c in ctxs:
        cells = []
        for k in ks:
            rs = [reuse[key][k] for key in reuse if key[1] == c and key[2] == "number_edit" and k in reuse[key]]
            if rs:
                so = [r["same_as_old"] for r in rs if r.get("same_as_old") is not None]
                cells.append(f"{np.mean([r['agree_full'] for r in rs]):.2f} / " + (f"{np.mean(so):.2f}" if so else "n/a"))
            else:
                cells.append("n/a")
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    L.append("\ncell = agree_full / same_as_old.  full recompute itself equals old answer in: " +
             ", ".join(f"ctx{c}: {np.mean([r['same_as_old'] for (q, cc, d), r in full.items() if cc == c and d == 'number_edit']):.2f}" for c in ctxs))

# ------------------------------------------------------------------ prompt-oracle distribution
L.append("\n## Per-prompt oracle k distribution (agreement-monotone), preserving deltas\n")
L.append("| ctx | " + " | ".join(f"k*={k}" for k in [0] + ks) + " | mean k* |")
L.append("|---|" + "---|" * (len(ks) + 2))
for c in ctxs:
    vals = [prompt_k[key] for key in reuse if key[1] == c and key[2] in PRES]
    cnt = collections.Counter(vals)
    L.append(f"| {c} | " + " | ".join(f"{cnt.get(k, 0)/max(1,len(vals)):.2f}" for k in [0] + ks) + f" | {np.mean(vals):.1f} |")

# ------------------------------------------------------------------ simple policies
feat_names = ["n_delta", "jaccard_q", "jaccard_prefix", "emb_cos_q", "emb_cos_prefix",
              "drift0_L2", "drift0_L4", "drift0_L8", "drift0q_L4", "drift0q_L8"]


def feat(key, name):
    r = full[key]
    f = r.get("feats")
    if f is None:
        return None
    if name.startswith("drift0q_"):
        arr = r.get("drift_step0_q"); return None if arr is None else arr[int(name.split("_L")[1])]
    if name.startswith("drift0_"):
        arr = r.get("drift_step0_all"); return None if arr is None else arr[int(name.split("_L")[1])]
    return f.get(name)


have_feats = any(full[key].get("feats") for key in reuse)
if have_feats:
    L.append(f"\n## Simple policies (leave-one-prompt-out, 4 quantile bins per feature, constraint loss >= -{TOL} pp on train)\n")
    L.append("| ctx | policy | S_policy | realized loss (pp) | S_class_oracle - S_policy | AUROC(feature -> k=16 safe) |")
    L.append("|---|---|---|---|---|---|")
    for c in ctxs:
        keyset = keys_for(c, PRES)
        qids = sorted({key[0] for key in keyset})
        for fn in feat_names:
            xs = {key: feat(key, fn) for key in keyset}
            if any(v is None for v in xs.values()):
                continue
            chosen = {}
            for q in qids:  # LOO over prompts
                train = [key for key in keyset if key[0] != q]
                test = [key for key in keyset if key[0] == q]
                if len(train) < 8:  # too few prompts to fit bins: fall back to no reuse
                    for key in test:
                        chosen[key] = 0
                    continue
                edges = np.quantile([xs[key] for key in train], [0.25, 0.5, 0.75])
                bins = collections.defaultdict(list)
                for key in train:
                    bins[int(np.searchsorted(edges, xs[key]))].append(key)
                bin_k = {b: safe_k(kk) for b, kk in bins.items()}
                for key in test:
                    chosen[key] = bin_k.get(int(np.searchsorted(edges, xs[key])), 0)
            s_pol = np.mean([reuse[key][chosen[key]]["compute_saved"] if chosen[key] > 0 else 0.0 for key in keyset])
            l_pol = np.mean([100 * (int(reuse[key][chosen[key]]["correct"]) - int(full[key]["correct"])) if chosen[key] > 0 else 0.0 for key in keyset])
            # AUROC of feature for "agreement at k=16" (or the middle k)
            kmid = 16 if 16 in ks else ks[len(ks) // 2]
            y = np.array([int(reuse[key][kmid]["agree_full"]) for key in keyset if kmid in reuse[key]])
            x = np.array([xs[key] for key in keyset if kmid in reuse[key]])
            auc = np.nan
            if 0 < y.sum() < len(y):
                order = np.argsort(x); ranks = np.empty(len(x)); ranks[order] = np.arange(1, len(x) + 1)
                auc = (ranks[y == 1].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (len(y) - y.sum()))
                auc = max(auc, 1 - auc)  # direction-free
            L.append(f"| {c} | {fn} | {s_pol:.2f} | {l_pol:+.1f} | {S[c]['class'] - s_pol:+.2f} | {auc:.2f} |")
        L.append(f"| {c} | delta-class (oracle labels) | {S[c]['class']:.2f} | (by construction >= -{TOL}) | 0.00 | - |")
else:
    L.append("\n(no cheap features in these records; policy stage skipped)")

# ------------------------------------------------------------------ figures
fig, axs = plt.subplots(1, 3, figsize=(16, 4.2))
xt = [np.mean([r.get("ctx_tokens", 0) for (q, cc), r in old.items() if cc == c]) for c in ctxs]
axs[0].plot(xt, [S[c]["uniform"] * 100 for c in ctxs], "o-", label="uniform k")
axs[0].plot(xt, [S[c]["class"] * 100 for c in ctxs], "s-", label="class oracle (delta known)")
axs[0].plot(xt, [S[c]["prompt_pres"] * 100 for c in ctxs], "^-", label="prompt oracle (agreement)")
axs[0].axhline(30, ls=":", c="gray"); axs[0].axhline(50, ls=":", c="gray")
axs[0].set_xlabel("shared context tokens"); axs[0].set_ylabel("% transformer compute saved"); axs[0].set_title("oracle headroom"); axs[0].legend()
for c in ctxs:
    axs[1].plot(ks, loss_tab[c], "o-", label=f"ctx {c}")
axs[1].axhline(-TOL, ls="--", c="r"); axs[1].set_xlabel("k (layers reused)"); axs[1].set_ylabel("accuracy loss vs full (pp)"); axs[1].set_title("quality vs k, preserving deltas"); axs[1].legend()
mat = np.array([[safe_k(keys_for(c, [d])) if keys_for(c, [d]) else np.nan for d in PRES] for c in ctxs])
im = axs[2].imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=max(ks))
axs[2].set_xticks(range(len(PRES))); axs[2].set_xticklabels(PRES, rotation=30, ha="right")
axs[2].set_yticks(range(len(ctxs))); axs[2].set_yticklabels([str(c) for c in ctxs]); axs[2].set_ylabel("ctx")
axs[2].set_title("class-oracle safe k"); plt.colorbar(im, ax=axs[2])
plt.tight_layout(); plt.savefig(os.path.join(out, "fig_r2_headroom.png"), dpi=140)

if have_feats:
    fig, axs = plt.subplots(1, 4, figsize=(17, 4))
    for ax, fn in zip(axs, ["n_delta", "jaccard_q", "emb_cos_q", "drift0_L4"]):
        for d in PRES:
            keys_d = [key for key in reuse if key[2] == d and feat(key, fn) is not None]
            ax.scatter([feat(key, fn) for key in keys_d], [prompt_k[key] + rng.normal(0, 0.4) for key in keys_d], s=8, alpha=0.5, label=d)
        ax.set_xlabel(fn); ax.set_ylabel("prompt-oracle k (jittered)")
    axs[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(os.path.join(out, "fig_r2_features.png"), dpi=140)

open(os.path.join(out, "ROUND2_RESULTS.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
