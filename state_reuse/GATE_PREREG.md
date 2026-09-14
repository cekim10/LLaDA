# Request-level reuse gate preregistration

Frozen 2026-09-13, before computing any feature-label relationship. Offline only, zero GPU.
This is the last decisive test of the systems branch. If it fails, the systems branch stops and
the work is written up as characterization.

## Question

The audit gives an oracle context-compute saving of 33.3-44.5% on SWE-agent traces, but the oracle
knows each turn's true label. A serving system does not. So:

> Can a cheap, causal, request-level signal decide "reuse cached residual state" vs "recompute"
> well enough to realize a useful share of the oracle saving without materially reusing on
> history-dependent requests?

## Data

The 400 adjudicated turns from `AGENT_TRACE_AUDIT_SEAL.md` (labels in `manual_labels/`, turn text
and token counts in `results/swe_agent_audit_random100/turns.jsonl`). No new labeling, no new
sampling, no GPU.

## Causality constraint (hard)

A gate runs *before* generation. Features may be computed only from `turn_index`, `task_text`,
`old_history`, `latest_observation` and their token counts. The following are forbidden because
they are functions of the generation or of the trajectory outcome, and using them would make the
result meaningless: `ai_text`, `command`, `command_kind`, `command_family`, `heuristic_label`,
`heuristic_evidence`, `exit_status`, `target`. `command_family` is also the stratification variable,
so it stays out of the model and is used only through `sampling_weight`.

Note on why this gate is text-only: LLaDA's `max_sequence_length` is 4096 while these agent contexts
average 8.6k-14k tokens, so the step-0 attention observable used in Round 4a'' cannot be computed on
this workload with this model without truncation. Attention-based gating is out of scope here.

## Frozen feature set

All cheap string/count features, computed per turn:

1. `turn_index`
2. `log1p(old_history_tokens_est)`
3. `log1p(latest_observation_tokens_est)`
4. `cold_share_est` = old-history tokens / context tokens
5. `obs_is_error` - latest observation matches command-not-found / Traceback / error / failed / no such file
6. `obs_is_empty_success` - latest observation reports a command that produced no output
7. `obs_is_file_view` - latest observation contains a `[File: ...]` listing
8. `obs_edit_rejected` - latest observation contains "Your changes have NOT been applied"
9. `jaccard_obs_task` - word-type Jaccard between latest observation and task text
10. `jaccard_obs_hist` - word-type Jaccard between latest observation and old history
11. `obs_novel_rate` - share of latest-observation word types absent from task and old history
12. `obs_is_repeat` - latest observation duplicates an observation already present in old history
13. `n_prior_assistant_turns` - count of assistant turns in old history
14. `obs_new_path_rate` - share of file-path-like tokens in the latest observation that do not appear in the task text

## Model and protocol

- Model: L2 logistic regression on standardized features, class-balanced. Reported alongside
  single-feature threshold rules.
- Validation: `GroupKFold` by trajectory (`row_idx`), 5 folds, 5 repeats with different group
  shuffles, seed 0. Turns of one trajectory never span train and test.
- Operating point: chosen **on the training folds only** as the most permissive probability
  threshold whose training unsafe-reuse share is at or below the budget, then applied unchanged to
  the held-out fold. All reported metrics are pooled out-of-fold predictions.

## Metrics

Weighted by `sampling_weight`, over non-bookkeeping turns, matching the audit's accounting:

```text
S_gate            = sum(w * cold_history_tokens for turns the gate reuses)
                    / sum(w * context_tokens for all non-bookkeeping turns)
unsafe_reuse_share = sum(w for turns that are reused AND not safe)
                    / sum(w for all non-bookkeeping turns)
```

`S_gate` counts every reuse decision, including wrong ones, because the system does save that
compute; the risk is carried by `unsafe_reuse_share`. The perfect gate gives `S_gate = S_oracle`
(33.3% conservative, 44.5% optimistic) at `unsafe_reuse_share = 0`.

Primary bound treats `ambiguous` as unsafe; the optimistic bound treats it as safe.
Uncertainty is an approximate 95% cluster bootstrap over trajectories, seed 0, 10,000 trials.

Budget rationale: the model-side rounds put the quality cost of freezing a history-dependent
request at about 10 pp. An `unsafe_reuse_share` of 0.10 therefore corresponds to roughly 1 pp
expected quality loss across the workload. This 10 pp constant is measured on GSM8K retrieval with
LLaDA, not on SWE-agent, and is an assumption, not a measurement on this workload.

## Decision rule

- **GO**: at the training-selected operating point, out-of-fold `S_gate >= 0.20` with
  `unsafe_reuse_share <= 0.10`, and the bootstrap 95% CI lower bound of `S_gate` is at least 0.15.
  Proceed to 4b: implement the runtime and measure wall-clock.
- **STOP**: no point on the out-of-fold sweep reaches `S_gate >= 0.15` at
  `unsafe_reuse_share <= 0.10`. The systems branch ends; write up characterization.
- Otherwise **INCONCLUSIVE**, which for this branch is treated as STOP unless the user decides
  otherwise, because this was preregistered as the last rescue attempt.

## Mandatory baselines and honesty requirements

Reported next to the learned gate: always-reuse, never-reuse, random reuse at the matched reuse
rate, and every single-feature threshold rule. If any single-feature rule comes within 3 pp of the
learned gate's `S_gate` at the same or lower `unsafe_reuse_share`, the result must be described as
that heuristic rather than as a learned mechanism. The `heuristic_label` from
`agent_trace_audit.py` is reported for reference only and marked illegal, since it reads the
generation.

## Deviations log

- **D1 (2026-09-13, logged after seeing the primary result):** the preregistered gate thresholds
  `P(safe)` alone. That ignores that a reused request is worth its cold-history tokens, so the
  system-correct selection for "maximise saving under an unsafe budget" is a fractional knapsack on
  `cold_i / (1 - p_i)`. A value/risk-ranked variant was added and evaluated under the identical
  grouped CV and training-only threshold selection. It is reported as a deviation, never as the
  preregistered primary.
- **D2 (same date):** a value-only baseline (rank by cold-history tokens, no safety model at all)
  was added after D1, because D1 could not otherwise be distinguished from "reuse the requests with
  the most history". This baseline is what triggered the preregistered honesty requirement.
- **D3 (same date):** the training unsafe budget was swept over {0.04, 0.06, 0.08, 0.10, 0.15} to
  show the achievable frontier, rather than being fixed at 0.10. Because the operating point was
  then read off against out-of-fold numbers, any point selected this way is optimistic and is
  reported as such.
