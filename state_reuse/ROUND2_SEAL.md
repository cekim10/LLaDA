# Round 2 seal (2026-09-11)

Reduced grid per Amendment A2: GSM8K test[50:100] (n=50), shared context {0, 512, 2048}
target tokens (actual 0 / 623 / 2191, 0 / 3 / 12 Q/A turns), k in {8, 16, 24, 32}, source =
step-0 all-mask prefix snapshot, 5 deltas, gen 128 / 64 steps / block 32. 3570 records
(150 old, 684 full, 2736 reuse), no crashes. Runtime: 0.8 h / 1.9 h / 5.6 h GPU time for
ctx 0 / 512 / 2048. Full tables: `results/round2r/ROUND2_RESULTS.md` (analyze_round2.py).
Tolerance for "safe" was preregistered at 2 pp of paired accuracy loss vs full recompute.

## Question (a): oracle headroom under long shared context

Pooled over answer-preserving deltas (n=180 pairs per cell), paired accuracy loss in pp
[95% bootstrap CI] and transformer compute saved:

| ctx | k=8 | k=16 | k=24 | k=32 |
|---|---|---|---|---|
| 0 | -1.1 [-5.6, +3.3] (7%) | -3.3 [-8.9, +2.2] (14%) | +1.7 [-3.9, +7.2] (21%) | +1.1 [-4.4, +6.7] (28%) |
| 512 | -3.3 [-7.8, +1.1] (20%) | +2.8 [-1.7, +7.8] (40%) | +2.8 [-2.2, +7.2] (60%) | -2.2 [-7.2, +2.8] (79%) |
| 2048 | +1.1 [-3.9, +6.1] (23%) | +3.3 [-1.7, +8.3] (46%) | +6.7 [+1.1, +12.2] (69%) | +4.4 [-1.1, +10.0] (93%) |

At 2048 shared tokens, reusing the step-0 prefix state through all 32 layers removes 93% of
per-step transformer compute with no detectable accuracy loss (point estimate positive).
Preregistered oracle numbers at ctx 2048: S_class_oracle = 0.73, S_prompt_oracle = 0.63-0.69,
S_uniform(2 pp) = 0.23. S_class >= 0.50 puts the result in the preregistered **strong** band,
so the policy stage was run.

Reuse also does not carry the old answer: for number_edit (answer-changing) the rate at
which the reused output equals the OLD answer is 0.08-0.27 across cells, against 0.19 for
full recompute itself. The step-0 snapshot is uncommitted state, as the Round 1 reading
predicted.

Accuracy of full recompute itself drops with context (old request 0.78 -> 0.72;
multiturn_append 0.80 -> 0.64), so the headroom is measured against a slightly weaker
reference at 2048. Reuse is paired against that same reference.

## Question (b): decision structure

Formally, at ctx 2048 S_class - S_uniform = 0.50 >= 0.10, which the preregistration counts
as "decision structure exists". It does not survive inspection:

1. The gap comes from a single cell. irrelevant_append at k=16, ctx 2048 is
   -4.0 pp [-14.0, +6.0]; it fails the 2 pp tolerance on its point estimate and caps the
   uniform k at 8. Every other preserving cell at 2048 is >= -2 pp. With tolerance 5 pp the
   uniform policy is k=32 and S_uniform = 0.93 > S_class(2 pp) = 0.73.
2. The 2 pp tolerance is far below the noise floor at n=50 (per-cell CI half-width about
   10 pp, pooled about 5 pp). The class-oracle heatmap (k* jumping between 0 and 32 across
   ctx for the same delta) is noise, not structure.
3. Per-prompt predictability is absent: AUROC of every feature (|D|, Jaccard, embedding
   cosine, step-0 shallow drift) for "k=16 safe on this prompt" is 0.52-0.68 at every ctx,
   against the preregistered 0.80 gate.
4. Simple policies match the class oracle: at 2048, emb_cos_q reaches S = 0.75 and
   drift0q_L8 0.76 with realized loss +1.1 pp, i.e. within 5 pp of the oracle. By the
   preregistered reading this flags the decision problem as **trivial** at long context.
5. The Round 1 semantic failure mode (hint_append) weakens as context grows:
   ctx 0: -2 to -6 pp; ctx 512: -6 to +8; ctx 2048: +2 to +10. At long context the
   answer-bearing delta no longer breaks reuse.

Conclusion for (b): at long shared context the right depth is "all of it". Knowing the
delta class or the prompt adds nothing measurable. Structure exists only at short context
(Round 1: hint_append -10 to -16 pp at k >= 16), where the compute at stake is small.

## Verdict against the preregistered rules

- Headroom: **strong** (S_class 0.73 at 2048; uniform k=32 saves 93% within noise).
- Decision structure: formally met on one noisy cell; substantively **not** present.
  Per-prompt gate (AUROC >= 0.8): failed. Simple-policy-matches-oracle flag: raised.
- Therefore: the phenomenon is confirmed and large, and the mechanism it supports is the
  simplest one ("cache the step-0 prefix state of the old request, reuse it at every layer,
  recompute only D and the canvas"). An adaptive depth-selection runtime is not supported
  by this data. Status of "Incremental/Stateful dLLM Execution" as a research candidate:
  the *execution* half is supported; the *decision* half is not.

## Positioning, stated plainly

For append deltas, Round 2's setting is structurally BiCache's shared-prefix setting: P is a
prefix of P+D and the suffix differs. BiCache reported that safe reuse depth grows with the
shared-prefix ratio; our ctx 0 (ratio ~0.3, safe depth <= 20) and ctx 2048 (ratio ~0.93,
safe depth 32) results are consistent with that trend, so the headroom curve itself
replicates and extends BiCache rather than contradicting it. What this project adds beyond
BiCache, on the data so far:

1. Source matters more than step alignment: the uncommitted step-0 snapshot is safe at
   every depth while the final-step snapshot collapses (Round 1). One snapshot, not a
   trajectory, is the right cache object.
2. In-place edits (name_edit, number_edit), which are not prefix sharing, behave like
   appends: name_edit at 2048 is +10 to +17 pp at k >= 16 (n=30), number_edit does not get
   stuck on the old answer.
3. Trajectory/quality decoupling (Round 1 C1/C2) and the step-direction convergence
   (Fig4) are diffusion-specific observations with no AR analogue.
4. A short answer-bearing delta breaks reuse at short context while a 100-token
   multi-turn append does not; the effect disappears at long context.

## What would change the verdict on (b)

Only a workload in which the delta must rewrite what the shared prefix means at long context
(retrieval that contradicts the history, instructions that reinterpret earlier turns) could
restore a non-trivial decision problem. GSM8K deltas do not do this. That is a Round 3
question, to be preregistered separately if pursued; it is not a reason to add mechanism now.

## Limitations

n=50 per cell, tolerance below noise, one model, one task family, name_edit n=30,
number_edit without gold, shared context is task-homogeneous (GSM8K Q/A turns), and the
2048 cell is 12 turns of the same fixed history for every prompt.
