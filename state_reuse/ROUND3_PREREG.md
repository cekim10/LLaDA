# Round 3 preregistration: does step-0 state reuse survive context mutation and relocation?

Frozen 2026-09-11, before any Round 3 data exists. Only the Deviations log may change after
the first run starts.

## Status carried in from Rounds 1-2

- GO: the step-0 (all-mask canvas) residual state of an old request is a reusable execution
  object. At 2048 shared tokens, reusing it through all 32 layers removes 93% of per-step
  transformer FLOPs with no detectable accuracy loss (Round 2).
- KILL: semantic-aware adaptive depth selection. At long context "reuse everything" is within
  noise of the oracle; no cheap feature predicts per-prompt safety (Round 2).
- OPEN (this round): does reuse extend beyond prefix-preserving updates?

## Thesis under test (deliberately modest)

> Unlike position-bound prefix reuse, dLLM step-0 residual states **may** remain reusable
> across context mutations and token relocation.

If the gate below passes, and only then, the claim is promoted to: the reuse boundary of a
diffusion LM need not coincide with the textual prefix boundary (prefix-addressed caching ->
content-addressed execution-state caching).

What is *not* claimed: that relocatability follows from the architecture. RoPE makes the
re-projection of K/V at a new position cheap, but a layer-l residual h_l already encodes the
old neighbourhood, old relative placement and old context through l rounds of bidirectional
attention. Whether h_l^old(i) is a usable stand-in for h_l^new(i + delta) is an empirical
question and is exactly what Round 3 measures.

Prior work this must be positioned against (verify citations before writing): BiCache
(dLLM shared-prefix KV, shallow layers); Prompt Cache (modular KV with positional accuracy);
CacheBlend (non-prefix chunk KV reuse plus selective token recompute); EPIC / position-
independent context caching; CacheSlide and RoPE re-rotation of cached keys. The AR line
shows relocated cached state generally needs positional correction plus contextual repair.
The candidate novelty is that an *uncommitted* dLLM residual state may tolerate both
mutation and relocation with no repair beyond re-projection. Nothing about "KV cannot be
moved" is claimed.

## Workload

Model, sampler, gen 128 / 64 steps / block 32, prompts GSM8K test[50:100] (n=50), history
pool test[100:200]: all unchanged from Round 2. Shared context: the fixed 2048-token history
(12 Q/A turns). ctx 512 may be run as a secondary condition with the same design.

Old request P = [history][question]. New request P' = [mutated history][same question].
The question and its gold answer never change, so accuracy is defined for every cell.

Mutations, each applied at a relative turn position p in {10, 25, 50, 75}% of the history:

| mutation | what changes | what it tests | role |
|---|---|---|---|
| edit | same-token-length in-place edit of one number (fallback: name) inside turn t | context change, positions preserved | required |
| insert | a fresh Q/A turn from the unused pool inserted before turn t | relocation + context change | core |
| delete | turn t removed | relocation + context change | core |
| reorder | turn t moved to the end of the history | extreme relocation | stress only |

Position map: every token of P' whose turn survives unchanged is mapped to its token in P
(turn spans from incremental chat-template tokenization; for edit, position-wise equal ids
inside the edited turn). Tokens of the mutated/inserted turn are "changed".

## Conditions per (prompt, mutation, p)

| condition | what is reused (k = 32 for all reused positions) | compute saved (FLOP proxy) |
|---|---|---|
| full | nothing | 0 |
| ours | every mapped token, before *and after* the mutation, relocated where needed | n_mapped / (P' + 128) |
| prefix | only the longest common prefix before the first changed token (BiCache-style, generously given full depth) | n_prefix / (P' + 128) |

Depth is fixed at 32 because Round 2 killed depth selection; k = 0 is the full condition.
Source is the step-0 snapshot of P. Reuse mechanism is `StateRecorder.start_inject` with
distinct old/new index sets; K/V and RoPE are recomputed at the new positions by the block
itself, so "ours" is the relocatable-state mechanism with no extra code path.

Diagnostics recorded for every cell: step-0 cos-sim between h_old(i) and h_new(map(i)) per
layer, split into pre-mutation positions (unshifted) and post-mutation positions (shifted
for insert/delete/reorder); generated-token agreement; answer agreement; stuck-on-old-answer
rate.

## Primary metrics

Paired accuracy loss vs full (pp) with 95% bootstrap CI over prompts, and the compute gap
S_ours - S_prefix. Tolerance rules are CI-based this round; no 2 pp point tolerance.

## Gate (fixed now)

Pooled over {edit, insert, delete} at p <= 50%:

1. quality: paired loss of "ours" vs full has point estimate >= -5 pp and 95% CI lower
   bound >= -10 pp;
2. headroom: S_ours - S_prefix >= 30 pp (FLOP proxy).

Both must hold for PASS. PASS -> the OPEN item closes YES and the thesis is promoted.
FAIL on (1) for insert/delete but PASS for edit -> "context-change tolerant, not
relocation tolerant": the direction converges to a better-understood prefix reuse and stops.
FAIL on (1) for edit too -> stop. Reorder never affects the gate; it is reported.

Secondary, reported not gated: loss vs p for each mutation (the "reuse % vs edit position"
curve for ours and prefix); pre- vs post-mutation drift per layer; ctx 512 if run.

## Explicitly deferred

- Mixed-commitment cache (prompt tokens at step 0, generated tokens at commit step): a
  different hypothesis about the canonical step per token; Round 4 candidate only if Round 3
  passes.
- Wall-clock accounting. Round 3 uses the FLOP proxy. The 12% re-projection estimate
  (k_proj + v_proj of a layer) is unmeasured and must not appear as a claim; any promotion
  to a system candidate requires measured latency including state load, projection, RoPE,
  gather/scatter and attention layout.
- Any adaptive, remasking or repair mechanism.

## Analysis plan

`analyze_round3.py`: per (mutation, p) table of acc_full, acc_ours, acc_prefix, paired loss
with CI, agreement, S_ours, S_prefix, gap; gate evaluation on the pooled set; figure 1:
compute saved vs p for ours and prefix (the structural-headroom plot); figure 2: loss vs p
per mutation; figure 3: step-0 drift per layer, pre vs post mutation.

## Cost

Per prompt at 2048: 1 old + 4 mutations x 4 positions x 3 conditions = 49 generations,
about 15-18 min on one A-class GPU; 50 prompts about 7 h on two GPUs.

## Deviations log

(empty)
