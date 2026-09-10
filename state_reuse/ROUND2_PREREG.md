# Round 1 seal and Round 2 preregistration

Written 2026-09-09, after inspecting the first 5 of 50 preregistered Round 1 prompts.
Round 2 is frozen here *before* any Round 2 data exists. Nothing in this file may be
edited after the first Round 2 run starts except the "Deviations" log at the bottom.

## 0. Frozen central claim

> dLLM execution state can be **numerically stale but semantically reusable**.

Two sub-claims, both already suggested at n=5 and to be tested at n=50:

- (C1) high hidden-state similarity does **not** imply trajectory similarity
  (k=4 layers reused, cos ~0.999, generated-token agreement drops to ~0.8).
- (C2) low trajectory similarity does **not** imply low task quality
  (k=16, token agreement ~0.5, final-answer accuracy of answer-preserving deltas
  equal to full recompute).

Interpretation to test, not assume: unlike AR KV reuse, where a perturbed
next-token distribution propagates causally, iterative denoising over the whole
canvas can self-correct stale intermediate state. Supporting signal to check at
n=50: Fig4 step-direction convergence (multiturn drift 0.12 -> 0.05 over steps).

Branch priority (fixed): canvas resume > layer-state reuse > distance-based reuse.
Distance-based invalidation is considered dead on Round 1 data unless n=50 reverses
Fig3. Layer reuse stays as characterization, not as the main mechanism.

## 1. Round 1: what is sealed

Config (unchanged from launch): LLaDA-8B-Instruct, GSM8K test[0:50], gen 128,
64 steps, block 32, deltas {irrelevant_append, hint_append, number_edit,
name_edit, multiturn_append}, k in {0,4,...,32}, sources {aligned, final, first},
resume t0 in {8,16,32,48}. Code: `run_pilot.py` at commit `1e7c322`; **no edits
to the pilot code until all 50 prompts are in** `results/pilot_gpu/records.jsonl`.

Gate before anything else: `control_selfinject.py` must show generated-token
agreement 1.0 (or equal to the run-to-run determinism floor) for k in {4,16,32}
when a request's own trajectory is injected. If it does not, Round 1 numbers are
not interpretable and instrumentation is fixed first.

Gate result (2026-09-10, elves-01 GPU, 64 steps / gen 128): run-to-run determinism 1.000;
self-inject k=4/16/32 token agreement 1.000; recorded states bit-identical across runs.
**Passed.** The k=4 divergence in Round 1 is therefore a real effect of the old request's
state, not an instrumentation artifact.

Known status: prompts 0-4 were computed on CPU (torch cu13 build vs CUDA 12.9
driver). Results are valid; remaining 45 prompts resume with `START=5 N=45` after
installing the cu128 torch build.

Round 1 primary metrics (fixed): final-answer accuracy vs gold where gold is
known, and final-answer agreement with full recompute. Generated-token agreement
is a secondary, descriptive metric only.

## 2. Round 2 hypotheses

- H1 (quality): Q_resume ~= Q_full for answer-preserving deltas.
- H2 (cost): C_resume << C_full in forward passes.
- H3 (oracle gap, the decisive one): at matched cost C,
  Q_resume(C) > Q_fresh(C), where "fresh" is a from-scratch generation of the new
  request with the same reduced step budget. If Q_resume(C) ~= Q_fresh(C) the
  canvas-resume branch has no headroom and is dropped.
- H4 (geometry): compute saved by layer reuse and by resume scales with
  L_shared / L_total; quality at fixed k does not degrade as L_shared grows.
- H5 (predictability): a cheap observable computed on the new request only
  (drift of changed-token representations, confidence of the old canvas under
  the new prompt at the first resumed step) separates preserving from changing
  deltas well enough to gate reuse. Tested as AUROC; threshold 0.8 fixed now.

## 3. Round 2 design

Baselines, all on identical prompts and step budgets:

| Baseline | Meaning |
|---|---|
| Full recompute (64 steps) | oracle-quality reference |
| Resume old canvas at t0 (64 - t0 steps) | simplest possible reuse |
| Fresh generation with 64 - t0 steps | the baseline that can kill the branch |

t0 in {16, 32, 48}. No remasking, no confidence policy in Round 2; the
phenomenon is measured before any solution mechanism is added.

Shared-context sweep: prepend a fixed, task-irrelevant shared context of
{0, 256, 512, 1024, 2048} tokens (same text for P and P+D; drawn once from a
fixed source and frozen). Output curve: reuse fraction -> compute saved ->
quality. No compute-saved threshold is chosen from these results; the Round 1
GO threshold (>=30% saved, <1 pp quality loss) is reused as-is.

Delta classes, explicit:

- D_preserving: irrelevant_append, hint_append, name_edit, multiturn_append
  (gold unchanged).
- D_changing: number_edit with gold recomputed. Gold for edited numbers is
  obtained by re-solving with the reference solution program where GSM8K
  provides arithmetic annotations; prompts whose edited gold cannot be derived
  are excluded before running (exclusion list frozen in the data file).

Prompts: GSM8K test[50:100] (disjoint from Round 1). n=50. Model, sampler,
gen 128, 64 steps, block 32 unchanged.

## 4. Decision rules (fixed now)

- Canvas resume is **kept** only if, for D_preserving at some t0 >= 32,
  Q_resume - Q_fresh >= 10 pp at matched cost and Q_full - Q_resume <= 2 pp.
- Canvas resume is **dropped** if Q_resume(C) - Q_fresh(C) < 5 pp at every t0.
- Incremental/stateful dLLM execution becomes a system candidate only if the
  above holds *and* H5 gate AUROC >= 0.8 on D_preserving vs D_changing.
- Layer reuse is reported as characterization regardless of outcome.

## 5. Analysis plan

Per (delta class, shared-context length, t0): mean accuracy with 95% bootstrap
CI over prompts, forward-pass count, and the paired difference resume - fresh.
Plots fixed in advance: (a) Q vs C for full / resume / fresh, one panel per
delta class; (b) compute saved vs L_shared/L_total with quality as color;
(c) ROC of the H5 observable.


## Amendment A1 (2026-09-10, written after the Round 1 seal and before any Round 2 run)

Round 1 (n=50) closed two branches and changed the reading of a third, so Round 2 is
narrowed before it starts. Sections 2-5 above are superseded by this amendment where they
conflict; they are kept for the record.

**Round 1 findings that drive the amendment.** (i) Canvas resume reproduces the old answer
in 92-96% of prompts at t0=48 for every delta including the answer-changing one; it is a
cache of the old answer, not reuse of state. Dropped from the main branch (H1-H3 retired).
(ii) Distance locality r ~ 0. Closed. (iii) The step-0 (all-mask canvas) prefix snapshot is
the only source with no detectable loss at every k; last-step snapshots collapse for k >= 20.
Reusable state is the *less committed* state, not the most recent one. Source is fixed to
the step-0 snapshot. (iv) hint_append (short, answer-bearing) breaks reuse while
multiturn_append (~100 tokens) does not: the question is whether D changes what the prefix
must mean, not how large or how far away D is.

**Single question for Round 2.** Under long shared context, does step-0 prefix-state reuse
have (a) large oracle compute headroom and (b) nontrivial decision structure that a cheap
runtime signal could exploit?

**Pipeline under test.** context change -> choose reusable depth k -> reuse step-0 prefix
state for layers 0..k-1 -> recompute the rest. No remasking, no adaptive mechanism.

**Workload.** Prompts: GSM8K test[50:100] (n=50, disjoint from Round 1). Shared context:
a fixed multi-turn GSM8K Q/A history built from test[100:200] (calculator annotations
stripped), identical for every prompt at a given length, target lengths
{0, 256, 512, 1024, 2048} tokens. Deltas unchanged (5 types). Sampler unchanged
(gen 128, 64 steps, block 32). k in {4,...,32}; k=0 is the full recompute.

**Oracle and thresholds (fixed now).**
- tolerance: pooled accuracy loss vs full recompute >= -2 pp.
- class oracle: per (ctx, delta) the largest k whose loss satisfies the tolerance for all
  k' <= k (monotone closure); S_class = mean saved. Requires gold, so preserving deltas only.
- uniform policy: one k per ctx that satisfies the tolerance in *every* preserving delta class
  (min over class oracles), i.e. the delta-blind baseline; S_uniform. (Pooling across classes
  would let one class's gain mask another's loss.)
- prompt oracle: per (prompt, ctx, delta) the largest k with answer == full for all k' <= k;
  S_prompt. Upper bound on per-prompt decision value; gold-free so includes number_edit.
- STOP if S_class at the longest context (2048) is < 30%.
- Weak (report as MAYBE, no policy stage) if 30% <= S_class < 50%.
- Strong if S_class >= 50%: run the policy stage.
- Decision structure exists if S_class - S_uniform >= 10 pp or S_prompt - S_class >= 15 pp.

**Policy stage (only if strong).** Cheap features recorded for every P+D during the run:
|D| in tokens, lexical overlap (Jaccard of D tokens with the question and with the whole
prefix), input-embedding cosine of D vs question / prefix, and step-0 drift of shared-prefix
states at layers 2/4/8 (cost <= 1/4 of one forward). Each feature -> k via 4 quantile bins,
leave-one-prompt-out, same tolerance. Report S_simple and S_class - S_simple. If a lexical
or embedding feature alone closes the gap to within 5 pp, the decision problem is trivial
and the systems contribution is weak; if only the drift feature closes it, or none does,
that is the result.

**number_edit (D_changing).** Gold recomputation is dropped (not derivable reliably). It is
reported through agreement with full recompute and the stuck-on-old-answer rate, and enters
the prompt oracle only.

**Instrumentation change.** `StateRecorder.start_record` gains an optional `steps` set so
only step 0 is stored at long context (memory). Default behaviour unchanged; Round 1 code
paths unaffected.

**Cost.** Per (prompt, ctx): 1 + 5 + 5x8 = 46 generations of 64 forwards. Forward time grows
with sequence length (~0.05 s at 220 tokens, est. ~0.5 s at 2300). Estimated 45-60 GPU-hours
for the full grid; split by prompt range and/or ctx across GPUs with `run_round2.sh`.

## 6. Deviations log

- 2026-09-10: `run_pilot.py` crashed on CUDA at the first delta of q5 (`sims` tensor lived on
  cuda, `.numpy()` failed; the CPU run never hit this). Fix: `.cpu()` on the drift tensor
  before saving. No change to any computation or metric. Also found that q4 of the CPU run
  was cut off mid-prompt (3 of 5 deltas), so q4 records/sims were deleted and Round 1
  resumes from `START=4 N=46`.
