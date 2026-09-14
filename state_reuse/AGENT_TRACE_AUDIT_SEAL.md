# SWE-agent workload audit seal (2026-09-13): INCONCLUSIVE

Preregistered manual audit completed (`AGENT_TRACE_AUDIT_PREREG.md`). 400 sampled turns from 96
SWE-agent trajectories (2,739 generation turns, 89 trajectories in the sample), two blind annotators
on disjoint halves plus a 100-turn overlap, all 47 overlap disagreements adjudicated against the
original packets. 400/400 resolved. No GPU time used. Artifacts: `manual_labels/` (labels A, B and
the adjudication) and `results/swe_agent_manual400/analysis/`.

## Decision: INCONCLUSIVE

| metric | ambiguity-conservative lower | ambiguity-optimistic upper |
|---|---|---|
| safe-turn share `f_safe` | 59.0% [51.1, 67.3] | 71.6% [64.4, 79.0] |
| oracle context saving `S_oracle` | **33.3% [25.0, 41.7]** | 44.5% [35.4, 53.5] |

Rule: GO if the CI lower endpoint of `S_lower` exceeds 0.30; STOP if the CI upper endpoint of
`S_upper` is below 0.20. Neither fires: 0.250 is not above 0.30, and 0.535 is not below 0.20.
Final label counts: safe 179, history_dependent 86, ambiguous 38, bookkeeping 97.

The point estimate of the conservative saving sits almost exactly on the GO bar (33.3% vs 30%),
so the verdict is driven by sampling width, not by a clearly negative result.

## Why the turn-level and token-level numbers differ

Most turns are safe (59-72%), but the oracle *token* saving is much lower (33-45%), because safe
turns are systematically the cheap ones:

| final label | n | mean turn index | mean cold-history tokens | mean context tokens |
|---|---|---|---|---|
| safe | 179 | 16.1 | 5,957 | 8,606 |
| history_dependent | 86 | 28.2 | 11,322 | 14,019 |
| ambiguous | 38 | 77.6 | 9,485 | 12,048 |
| bookkeeping | 97 | 69.1 | 10,281 | 12,889 |

Safe turns occur early, when there is little history to reuse; by the time the context is large the
generation is usually continuing an earlier plan. This is the workload-level version of the same
boundary Round 4a-4a'' found in the model: reuse is cheap exactly where it is least valuable.

## Annotator reliability

Headline agreement is 53.0% with Cohen's kappa 0.343, which looks poor, but it decomposes:

- On the axis the metric depends on, agreement is good. Restricted to the 60 overlap items where
  both annotators chose `safe` or `history_dependent`: agreement 86.7%, kappa 0.653.
- The disagreement is a label-definition split. A used `bookkeeping` 98/250 and `ambiguous` 2/250;
  B used `bookkeeping` 5/250 and `ambiguous` 60/250. 25 of the 47 disagreements are exactly
  A=bookkeeping vs B=ambiguous on malformed retry loops. Adjudication upheld A's reading in 30 of
  A's 35 overlap `bookkeeping` calls (86%).
- Adjudication moved only one way on the substantive axis. All 8 A=safe / B=history_dependent
  items resolved to history_dependent (7) or ambiguous (1); none to safe. B is the stricter
  annotator and was right each time it mattered.

That last point is a real bias risk: 150 items were labeled by A alone and never cross-checked.
A post-hoc sensitivity analysis re-weighting each solo half by that annotator's empirical
P(final label | own label), measured on the adjudicated overlap, gives `f_safe` 57.7-64.4% and
`S_oracle` 32.1% [24.9, 39.2] to 38.6% [31.3, 45.9]. The decision is unchanged (INCONCLUSIVE) and
the ambiguity band narrows, so the headline is robust to the annotator asymmetry, but the
corrected numbers are slightly lower than the primary ones. This correction is post hoc and is not
a preregistered result.

## What the preregistration says to do next

"Otherwise: inconclusive; increase the manual sample or inspect tau-bench before spending GPU."
No GPU gate experiment is unlocked.

Practical note on the first option: the confidence interval is clustered over trajectories, and the
sample already contains 89 of them with a half-width of about 8 pp on `S_lower`. Moving the CI
lower endpoint from 25% to above 30% while the point estimate stays near 33% needs roughly a 2.5x
narrower interval, i.e. on the order of 6x more trajectories and a few thousand more hand-labeled
turns. That is not a reasonable manual budget, and it would only pay off if the point estimate held
exactly where it is. Inspecting a second workload (tau-bench) is the cheaper of the two
preregistered options, and a third is to accept ~1/3 as the honest headroom estimate for this
workload class and stop treating the 30% bar as decisive.

## Bottom line for the project

This audit was the workload-side gate for a request-level reuse mechanism. It neither authorizes
nor kills that mechanism. Combined with the model-side results (Rounds 1-3 GO within a bound,
Rounds 4a-4a'' killing adaptive depth and block liveness), the position is unchanged: the
characterization result stands, and a systems claim still lacks both a validated request-level gate
and a workload demonstrated to sit inside the bound.

## Deviations from the preregistration

None in design or thresholds. The run was interrupted twice by account usage limits and resumed
from annotator checkpoints; the final artifacts contain no duplicate or out-of-packet item ids
(verified by the analyzer, which rejects both).
