# Round 4a' preregistration: is the Round 4a failure caused by mutation, or by freezing a block the generation must read?

Frozen 2026-09-11 before any Round 4a' data. Single purpose: separate

- H1: the old/mutated state is wrong (mutation-dependent failure), from
- H2: freezing step-0 prompt state across denoising steps blocks the co-evolution that long-range
  retrieval needs, even when P' = P exactly (mutation-independent failure).

No mechanism, no liveness policy, no attention-based estimator is built or acted on this round.

## Workload

Round 4a format: 12-turn GSM8K history (2048 tokens), the problem statement placed as a
"remember this" user turn at relative position p in {10, 50, 90}%, final turn asks to solve it.
Prompts test[50:100], n = 50. Model and sampler unchanged.

Format changes relative to Round 4a, made to reduce extraction noise and declared here:
gen_length 256 with steps 128 (same 16 denoising steps per 32-token block as every previous round),
and the final turn adds "Do not restate the problem; go straight to the solution." Round 4a' numbers
are never pooled with Round 4a.

**There is no mutation in this round.** P' = P for every condition. Gold is the original gold for
every cell. The step-0 source is the request's own step-0 state from the full run.

## Conditions (k = 32, step-0 source, frozen at every denoising step)

| condition | frozen set | question it answers |
|---|---|---|
| full | nothing | reference |
| identity | every prompt token | does freezing alone, with no mutation, break retrieval? |
| all_but_dep | every prompt token except the problem block | is co-evolving only the needed block sufficient? |
| dep_only | only the problem block | does freezing the needed block alone reproduce the failure? |
| dep_fresh_num | every prompt token except the digit-bearing tokens inside the problem block | diagnostic only ("hook" cell) |

Problem block = token span of the "remember this" user message containing the problem.
Per (prompt, p): 1 full + 4 frozen = 5 generations; 15 per prompt.

## Metrics

Paired accuracy loss vs full (pp) with 95% bootstrap CI over cells; answer agreement with full;
generated-token count and restating rate reported as format diagnostics.

## Primary gate: identity, pooled over p (n = 150)

- loss(identity) <= -15 pp: **mutation-independent freezing failure exists** (H2 supported).
- 95% CI of loss(identity) entirely above -5 pp: **pure freezing does not explain Round 4a** (H1
  side; mutation interaction must be revisited).
- otherwise: **INCONCLUSIVE**.

p = 10% (the cell that failed in Round 4a) is reported with the same thresholds as the direct
replication. If the pooled gate is INCONCLUSIVE while p = 10% is <= -15 pp and p = 90% is within
-5 pp, the outcome is recorded as "distance-implicated, inconclusive on the pooled gate"; distance is
not claimed as a mechanism this round.

## Localization test (secondary, only interpreted if the primary gate is not INCONCLUSIVE)

Localized if loss(all_but_dep) >= -5 pp with CI low >= -10 and loss(dep_only) <= -10 pp.
Not localized otherwise. dep_fresh_num is reported and not used in any decision.

## Pre-declared next actions

- Outcome A (identity <= -15, localized): validity is dependency-sensitive because retrieved
  blocks must co-evolve with generation. Next question, to be preregistered separately: is block
  liveness cheaply observable (candidate: canvas -> block attention mass at step 0)? This is an
  adaptive block-liveness question and does not conflict with the Round 2 KILL of adaptive depth.
- Outcome B (identity within -5): Round 4a losses are mutation-specific; revisit how context
  mutation changes retrieval-relevant representations globally. Candidate still alive, different
  mechanism.
- Outcome C (cells not separable, CIs overlapping across identity / dep_only / all_but_dep): the
  GSM8K retrieval setup is too noisy for validity work; change workload before adding GPUs.
- 4b (runtime, wall-clock) stays deferred until a cache-validity abstraction exists.

## Deviations log

(empty)
