# Round 4a'' preregistration: confirmatory replication of block liveness on unseen prompts

Frozen 2026-09-11 before any Round 4a'' data. Round 4a' is sealed as INCONCLUSIVE on its own gate
and is not reopened; this round tests a new confirmatory hypothesis, defined from the 4a' contrast,
on prompts that no earlier round has used.

## Primary question

> Does keeping the dependency-bearing block live recover the quality lost by freezing the rest of
> the context at step 0, on unseen prompts?

## Workload (identical protocol to Round 4a')

12-turn GSM8K history (2048 tokens; turns from test[100:112] as in every round), the problem placed
as a "remember this" user turn at p in {10, 90}%, final turn asks to solve it with the digit-free
topic hint and the "do not restate" sentence. gen 256 / 128 steps / block 32, k = 32, step-0 source
= the request's own full run. No mutation (P' = P).

Prompts: GSM8K test[200:300] from `gsm8k_test_full.json` (n = 100; items 0-199 of that file are
byte-identical to the frozen `gsm8k_test200.json`). Cells: 100 x 2 = 200 per condition.

Conditions: full, identity (freeze 100% of context tokens), all_but_dep (freeze everything except
the problem block, about 97%). dep_only and dep_fresh_num are dropped.

## Primary statistic and rule

D = Q(all_but_dep) - Q(identity), paired per cell, 95% bootstrap CI over the 200 cells.

- **CONFIRMED (block liveness: GO)**: D >= 5 pp and CI lower bound > 0.
- **KILL (4a' localization unstable)**: CI of D includes 0 and point estimate < 5 pp, with
  Q(identity) - Q(full) still significantly negative (freezing cost real, not localized), or D
  near 0 with no freezing cost at all.
- Otherwise: report as not confirmed; no further GSM8K GPU time on this branch.

5 pp is deliberately below the 10 pp seen in 4a'. It is the smallest effect worth building a
mechanism on.

## Secondary statistics (reported, not gated)

- Q(identity) - Q(full): replication of the freezing cost (expected negative).
- Q(all_but_dep) - Q(full): how close a 3% live block gets to full recompute (expected small).
- Legacy Round 4a' metric, identity <= -15 pp, reported for continuity only.
- Per-p breakdown (10 vs 90). No distance claim regardless of outcome.
- Pooled 4a' + 4a'' D is reported as a secondary sequential estimate, clearly labeled; the
  primary decision uses 4a'' cells only.

## Passive logging (no thresholds, no policy this round)

For every cell, one extra all-mask forward records the step-0 attention mass from canvas positions
onto each context position at layers {4, 8, 12, 16, 20, 24, 28, 31}, saved per cell
(`attn0_q*_p*.npz`) together with the problem block's share of context attention. This is stored
for a later block-liveness predictor round and is not looked at for any decision here.

## Pre-declared next actions

- CONFIRMED: a short block-liveness predictor / oracle-headroom round (which blocks are live, is
  attention mass at step 0 predictive, what fraction of an agent context is cold) before any
  runtime work. 4b only after that.
- KILL: stop spending GPU on GSM8K for this branch; the execution-object result (Rounds 1-3,
  bounded) stands as characterization.

## Cost

Per cell: 3 generations + 1 probe forward; about 3.3 min per cell on one A-class GPU; 200 cells
about 11 h on one GPU, 5.5 h on two.

## Deviations log

(empty)
