# Round 4a'' seal (2026-09-12): block liveness KILLED

Preregistered confirmatory replication completed on unseen prompts (GSM8K test[200:300], n=100,
p in {10, 90}, 200 cells per condition, identical protocol to Round 4a'). 800 records, 200 attention
logs, no crashes. Full-recompute accuracy 0.77. Tables: `results/round4app/ROUND4APP_RESULTS.md`.

## Primary: D = Q(all_but_dep) - Q(identity)

| | 4a' (p 10/90, n=100) | **4a'' (unseen, n=200)** | pooled (secondary, n=300) |
|---|---|---|---|
| D | +9.0 | **+2.5 [-5.0, +10.0]** | +4.7 [-1.3, +10.7] |
| identity - full | -15.0 | **-9.0 [-16.5, -1.5]** | -11.0 [-17.0, -5.3] |
| all_but_dep - full | -6.0 | **-6.5 [-13.5, +0.5]** | -6.3 [-11.7, -0.7] |

Rule: D >= 5 pp with CI low > 0 -> GO; CI includes 0 and point < 5 -> KILL.
**Result: KILL.** The 4a' localization did not replicate. The 4a' vs 4a'' difference in D is
+6.5 [-5.5, +18.5], consistent with sampling noise on the first draw.

## What replicated and what did not

- **Replicated: the freezing cost.** Freezing 100% of a 2048-token context at step 0, with no
  context change, costs 9 pp on unseen prompts (4a': 12.7). Across 4a', 4a'' and Round 4a's
  controls, full-depth step-0 freezing in a format where the generation must reason over history
  content costs 9-14 pp consistently.
- **Did not replicate: localization.** Keeping the 3% dependency block live recovers nothing
  measurable (-6.5 vs -9.0). Cell-level: among the 38 cells where freezing broke a correct answer,
  all_but_dep was right in 71%, exactly its base rate (70%); it also broke 22 of 115 cells that
  identity got right. The cost is distributed over the context, not concentrated in the block the
  answer depends on. This matches Round 4a's dep_move observation that moving the relevant turn
  moved every token's state (drift 0.12-0.20): when the task reasons over history, the context
  representation reorganizes globally during denoising.
- **Passive attention log (descriptive).** At step 0 the canvas puts 9.3x the problem block's token
  share of attention on it at layer 31 (6.5x at 28, about 1x below layer 16). The block is clearly
  identified by deep-layer attention, so a liveness predictor would have had a signal; the target
  it would predict does not recover quality. Attention share is the same whether freezing breaks the
  cell or not (0.316 vs 0.303). Nothing more is done with this log.

## Status board after Round 4a''

| item | status |
|---|---|
| step-0 residual state as reusable execution object, incl. relocation beyond the prefix (Rounds 1-3) | GO, **bounded**: holds when the generation does not need to reason over the frozen content |
| headroom vs prefix caching (early mutation: 12-13% vs 90-95% saved) | stands, same bound |
| semantic-aware adaptive depth (Round 2) | KILL |
| distance-based invalidation, canvas resume (Round 1) | KILL |
| dependency-changing mutation as the failure mode (Round 4a) | not the failure mode |
| mutation-independent freezing cost in retrieval-heavy tasks (4a', 4a'') | real, 9-14 pp |
| block liveness / selective co-evolution (4a'') | **KILL** |

Per the preregistration: no further GSM8K GPU time on the liveness branch.

## What the boundary now looks like

Reuse safety is a property of the *request*, not of blocks: when the final turn is self-contained
(Rounds 2-3) the entire context can be frozen at step 0, relocated and mutated, at no cost; when the
final turn requires reasoning over history, freezing any large part of the context costs about 10 pp
and no block-level exemption fixes it. This is a request-level cold/hot distinction, not a
block-level one. Whether it is cheaply observable at the request level (for example, the share of
step-0 canvas attention that lands on the history rather than on the final turn, compared between
the two formats) is a one-forward-per-prompt question that has not been asked; it is the only
remaining cheap test on this workload and is not preregistered here.

## What survives for a paper

A characterization paper with a clear positive core and an honest boundary:

1. dLLM execution state is numerically stale but semantically reusable (C1/C2, Round 1).
2. The reusable object is the uncommitted step-0 residual, not the latest state (Rounds 1-2).
3. It survives mutation and relocation far beyond prefix caching, with 66 pp more compute saved
   than a prefix cache on early mutations (Round 3).
4. The limit: reuse fails, by about 10 pp, when the generation must reason over the frozen
   history, and the failure is global rather than block-local (Rounds 4a-4a''); adaptive
   depth and block liveness both fail as mechanisms.

A systems paper needs the request-level gate above, or a workload class (agent traces with
self-contained tool calls) shown to sit inside the bound. Neither is established.
