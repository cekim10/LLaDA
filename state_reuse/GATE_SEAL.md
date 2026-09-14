# Request-level reuse gate seal (2026-09-13): STOP, with a clean explanation

Preregistered offline test (`GATE_PREREG.md`) on the 400 adjudicated SWE-agent turns, 303 of them
non-bookkeeping across 87 trajectories. Zero GPU. Grouped 5-fold CV by trajectory, 5 repeats,
operating point selected on training folds only, cluster bootstrap over trajectories.

## Verdict: STOP for the safe-request gate

The preregistered gate (threshold on P(safe), training budget 0.10) gives out-of-fold
`S_gate` 12.5% at `unsafe_reuse_share` 12.5%, and the best feasible point on its out-of-fold sweep
within budget is 8.5%. That is below the 15% STOP line, so the preregistered rule fires STOP.

The value/risk deviation (D1) reaches `S_gate` 24.3% at `unsafe_reuse_share` 8.9%, which would pass
GO, but the preregistered honesty requirement disqualifies it: a value-only baseline that ranks by
cold-history tokens with **no safety model at all** matches or beats it at every budget.

| training budget | D1 value/risk gate | value-only (no safety model) |
|---|---|---|
| 0.06 | 19.4% S at 7.6% unsafe | 18.9% S at 6.3% unsafe |
| 0.08 | 24.3% S at 8.9% unsafe | 26.1% S at 8.6% unsafe |
| 0.10 | 31.6% S at 10.9% unsafe | 33.4% S at 10.3% unsafe |

## Why it fails, precisely

The safety signal is real. The classifier reaches out-of-fold AUROC 0.736 on safe vs
history-dependent, from cheap text features alone; the dominant one is the word overlap between the
latest observation and the task text (standardized coefficient +1.18), followed by cold share
(-0.75) and turn index (-0.62).

The signal is also almost useless for saving compute, because predicted safety and the value of
reuse are strongly anti-correlated: `r(P(safe), cold tokens) = -0.845`, and cold tokens on their own
are an *inverse* safety predictor (AUROC 0.268). At matched reuse rates:

| reuse rate | P(safe) ranking: S / unsafe-among-reused | cold-token ranking: S / unsafe-among-reused | random |
|---|---|---|---|
| 10% | 0.1% / 3.3% | 20.2% / 63.5% | 9.0% / 56.3% |
| 20% | 1.0% / 6.7% | 35.3% / 55.1% | 14.3% / 43.2% |
| 30% | 3.1% / 11.1% | 47.6% / 57.8% | 20.4% / 42.2% |
| 50% | 12.0% / 25.9% | 64.5% / 57.7% | 35.9% / 40.4% |

Base rate: 41.0% of weighted non-bookkeeping turns are unsafe. So the P(safe) ranking discriminates
strongly (3.3% unsafe among reused at a 10% reuse rate, against a 41% base rate) while saving
essentially nothing, and the cold-token ranking saves a lot while discriminating nothing at all
(56-63% of its reuses are history-dependent, i.e. *worse* than the base rate).

In one line: **the requests you can confidently call safe are exactly the requests with almost no
history worth reusing.** This is the workload-level restatement of the model-side boundary from
Rounds 4a-4a''. Safe turns are early (mean turn index 16, 6.0k cold tokens); history-dependent turns
are late (28, 11.3k).

## What is left standing, honestly

A value-only policy is deployable and is not nothing: "reuse when the history is long, accept the
risk" gives 26-33% of context compute at an `unsafe_reuse_share` of 8.6-10.3%, which under the
assumed 10 pp cost per wrongly-reused request is roughly 0.9-1.0 pp average quality loss. Two
caveats make me not call this a systems result:

1. It is not a gate. It uses none of the safety signal and its per-request risk is worse than the
   base rate: about 56% of the requests it reuses on are history-dependent. The loss is concentrated
   on roughly a tenth of all requests, and in agent workloads a corrupted turn can derail the rest of
   the trajectory, so an average-loss argument is weak here.
2. The 10 pp cost constant comes from GSM8K retrieval with LLaDA, not from SWE-agent. It has never
   been measured on this workload, and it cannot be with this model: LLaDA's max sequence length is
   4096 while these contexts run 8.6k-14k tokens.

## Consequence for the project

The systems branch stops here as preregistered. The missing mechanism was a safe-request gate; the
test says a usable one does not exist on this workload, and it says why, which is a better outcome
than an inconclusive result. Everything that stands from the model side is unchanged:

1. dLLM execution state is numerically stale but semantically reusable (Round 1).
2. The reusable object is the uncommitted step-0 residual, not the latest state (Rounds 1-2).
3. It survives mutation and relocation far beyond prefix caching, saving 66 pp more than a prefix
   cache on early mutations (Round 3).
4. The limit: reuse costs about 10 pp when the generation must reason over the frozen history, the
   failure is global rather than block-local, and neither adaptive depth nor block liveness fixes it
   (Rounds 4a-4a'').
5. New: on a real agent workload the oracle headroom is about a third, but it is not addressable,
   because predicted safety is anti-correlated with the value of reuse (r = -0.85).

Finding 5 is the honest ending of the systems question and belongs in the paper as a limitation
section with these numbers, not as a failure to hide. It also names the condition under which the
mechanism would become useful: a workload where long-context requests are frequently
self-contained. SWE-agent is not one, and the reason is structural rather than incidental.

## Reproduction

```bash
python gate_eval.py --turns results/swe_agent_audit_random100/turns.jsonl \
  --manifest results/swe_agent_manual400/sample_manifest.jsonl \
  --resolved results/swe_agent_manual400/analysis/resolved_labels.jsonl \
  --out results/swe_agent_gate --budget 0.10
```
