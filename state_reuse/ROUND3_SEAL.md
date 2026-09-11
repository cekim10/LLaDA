# Round 3 seal (2026-09-11)

Preregistered design completed with no deviations: GSM8K test[50:100] (n=50), 2048-token
history (12 Q/A turns), mutations {edit, insert, delete, reorder} x turn position
{10, 25, 50, 75}%, conditions {full, ours, prefix} at k=32, source = step-0 snapshot.
2450 records (50 old, 800 full, 800 ours, 800 prefix), 800 drift files, no skips, no crashes,
single GPU. Full tables: `results/round3/ROUND3_RESULTS.md`; figure `fig_r3_ctx2048.png`.

## Gate: PASS

Pooled over edit + insert + delete at p <= 50% (n = 450 paired cells):

| criterion | preregistered threshold | result |
|---|---|---|
| quality: paired loss ours vs full | point >= -5 pp and CI low >= -10 pp | **-0.2 pp [-4.0, +3.6]** |
| headroom: S_ours - S_prefix | >= +30 pp | **+66 pp** (0.93 vs 0.27) |

Per mutation (p <= 50%): edit +2.0 [-4.0, +8.0], insert +0.0 [-6.0, +6.0],
delete -2.7 [-10.0, +4.7]; gap +0.64 to +0.67 for all three.

Reorder (stress, not gated): losses -6, -10, +2, +12 pp at p = 10/25/50/75, pooled about
-0.5 pp; S_ours 0.94 vs S_prefix 0.12-0.60. The moved turn itself is relocated by
~1000-1900 positions and reused.

## What the data says

1. **Reuse does not stop at the prefix boundary.** "ours" saves 90-95% of per-step
   transformer FLOPs regardless of where the mutation sits; "prefix" saves 12-65% and falls
   to 12-13% for an early mutation. The gap is largest exactly where prefix caching is
   weakest (early mutation: +0.78 to +0.82).
2. **Relocation costs nothing measurable.** For insert and delete, tokens shifted by one
   turn (~130-170 positions) have step-0 drift at layer 32 of 0.019-0.021 (cos > 0.98)
   and 0.0004-0.0011 at layer 16, *lower* than the unshifted tokens before the mutation
   (0.035-0.040). Positional relocation of the residual state, with K/V and RoPE
   recomputed at the new position, does not add drift beyond the context change itself.
   The far-relocated reorder block drifts most (0.093 at L32) and still passes on quality.
3. **Edit is essentially free**: drift 0.003 at layer 32, loss +2.0 pp.
4. **Same decoupling as before**: answer agreement with full is 0.66-0.84 while accuracy
   is unchanged. Reuse changes which answer appears in about one prompt in four, with
   flips in both directions. Stuck-on-old-answer rate of ours (0.72-0.74) is not above
   full's (0.77-0.90).

## Promotion, per the preregistered rule

The thesis "dLLM step-0 residual states may remain reusable across context mutations and
token relocation" is supported, so the claim is promoted to:

> The reuse boundary of a diffusion LM need not coincide with the textual prefix
> boundary. The cache object is the uncommitted step-0 residual state, addressed by
> token-block content rather than by (prefix, position).

Status board:

- GO: step-0 state as reusable execution object (Rounds 1-2).
- GO: reuse beyond prefix-preserving updates, including relocation (Round 3).
- KILL: semantic-aware adaptive depth selection (Round 2).
- KILL: distance-based invalidation, canvas resume as reuse (Round 1).

"Incremental / stateful dLLM execution" is now a research candidate: the execution half
is supported on three rounds of preregistered data, and the decision half is simply not
needed on these workloads.

## Caveats that must go into the paper

- The history is task-homogeneous GSM8K Q/A and the mutated turn is semantically
  unrelated to the question, so every Round 3 mutation is an "irrelevant" change. Round 2
  showed the one failure mode is a delta that changes what the prefix must mean
  (hint_append at short context), and it faded at long context. A mutation that the
  question depends on (a follow-up referencing the edited turn; a contradiction) is the
  obvious adversarial workload and has not been run.
- The "prefix" baseline is our own reconstruction (step-0 residual state, all 32 layers,
  exact prefix), generous to BiCache in depth but not BiCache's KV mechanism or its
  shallow-layer policy. A direct comparison needs their code or a faithful reimplementation.
- Compute saved is a FLOP proxy. Wall-clock, including state load, k/v projection, RoPE,
  gather/scatter and attention layout, is unmeasured. The re-projection cost estimate is
  not a claim.
- n=50 per cell (CI half-width about 10-12 pp per cell, 4 pp pooled); one model, one task
  family, gen 128 / 64 steps.
- Answer agreement with full recompute is only 0.66-0.84: the mechanism preserves
  accuracy, not outputs. Any deployment claim must be about task quality.

## AR positioning (to verify citations before writing)

AR non-prefix reuse (Prompt Cache, CacheBlend, EPIC, CacheSlide / RoPE re-rotation) needs
positional correction and, for chunk composition, contextual repair by partial recompute.
Round 3 shows a dLLM residual state relocated across turns with no repair beyond the
block's own re-projection, on a bidirectional model where the prefix is not even
theoretically independent of the suffix. That contrast, not "KV cannot move", is the claim.

## Suggested next steps (not preregistered; for discussion)

- Round 4a, adversarial: mutations the question depends on; expect the first real
  failure mode beyond the prefix and a bound on the claim.
- Round 4b, systems: wall-clock implementation of the cache (store step-0 residuals per
  turn/block, content hash, relocate, recompute D + canvas) against full recompute and a
  faithful prefix-cache baseline, on multi-turn agent traces.
- Round 4c, deferred hypothesis: mixed-commitment cache for generated tokens, so that
  the assistant turn of request t costs nothing in request t+1.
