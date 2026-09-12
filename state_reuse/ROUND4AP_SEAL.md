# Round 4a' seal (2026-09-11): INCONCLUSIVE on the gate, H2-direction on the evidence

Preregistered design completed: 50 prompts x p in {10, 50, 90} x {full, identity, all_but_dep,
dep_only, dep_fresh_num}, k = 32, no mutation (P' = P), gen 256 / 128 steps, "do not restate"
instruction. 750 records, no crashes. Full-recompute accuracy 0.75 (Round 4a format: 0.63-0.72).
Tables: `results/round4ap/ROUND4AP_RESULTS.md`, figure `fig_r4ap.png`.

## Preregistered gate: INCONCLUSIVE

| condition | frozen | loss vs full (pp), pooled n=150 | p=10 | p=50 | p=90 |
|---|---|---|---|---|---|
| identity | 100% | **-12.7 [-20.7, -4.7]** | -10.0 | -8.0 | -20.0 |
| all_but_dep | 97% (problem block live) | -2.7 [-10.7, +4.7] | -8.0 | +4.0 | -4.0 |
| dep_only | 3% (problem block only) | -6.0 [-13.3, +1.3] | -12.0 | 0.0 | -6.0 |
| dep_fresh_num (diagnostic) | 100% minus digit tokens | -5.3 [-13.3, +3.3] | -4.0 | +6.0 | -18.0 |

identity is -12.7, above the -15 pp threshold for "freezing failure exists" and with a CI that is
not above -5 pp, so by the preregistered rule the primary gate is INCONCLUSIVE and the localization
test is formally not interpreted. Everything below is reported as evidence, not as a gate outcome.

## What the data says

1. **Freezing with zero mutation costs a real 13 pp.** The identity CI excludes 0. The magnitude
   matches Round 4a's control losses (ctrl_other -14.0 [-25, -3]; dep_move -18) obtained with
   mutations. The Round 4a "control failures" are therefore reproduced without any mutation: H1
   (the mutated state is wrong) is disfavored as the explanation. Robust to the answer extractor
   (a strict "answer is"-first extractor gives -12.0 [-20.7, -3.3]).

2. **Keeping the 3% problem block live recovers most of it.** Paired identity - all_but_dep =
   **-10.0 [-17.3, -2.7]** (significant). Freezing only the problem block costs -6.0 (CI includes
   0); dep_only - all_but_dep = -3.3 [-11.3, +4.7]. Losses are roughly additive (-6.0 + -2.7 vs
   -12.7). Direction as H2 predicts; the strict localization rule (dep_only <= -10) is not met.

3. **Distance is not the mechanism.** p=90 (problem adjacent to the final turn) is the *worst* cell
   (-20), the reverse of Round 4a's p=10 pattern. Per-p cells have +-14 pp CIs; the p-pattern is
   noise. Dropped.

4. **The frozen block is read, but reasoning over it degrades.** Fraction of the problem's numbers
   that reappear in the output: full 0.972, identity 0.945 (-2.7 pp, CI includes 0), all_but_dep
   0.973. Among cells where every number was retrieved, accuracy is 0.78 (full) vs 0.66 (identity).
   So co-evolution is not needed to *copy* tokens out of a frozen block; it is needed for whatever
   the model does with them afterwards. Post hoc; to be tested.

5. **Round 4a's dep_number (+2.9 pp) is no longer evidence of tolerance.** Its full-recompute
   baseline was 0.51 (floor effect) on n=68; the analogous cell here, dep_fresh_num, is -5.3
   pooled and -18 at p=90. The "fresh token as hook" idea is not supported.

6. **Format.** "Do not restate the problem" was ignored: 58-67% of outputs restate it and 63-74%
   fill the 256-token canvas (mean 228 tokens). Accuracy still rose to 0.75. Extraction noise is
   symmetric across conditions and does not change any conclusion above.

## Where this leaves the claims

- Round 3 stays bounded as in the Round 4a seal, now with the bound sharpened: full-depth step-0
  freezing costs about 13 pp when the generation must reason over content inside the frozen
  region, *even with no context change*. Rounds 2-3 (decorative history, self-contained
  question) sit outside that regime, which is why they showed no loss.
- The cost is mostly removable by keeping the one block the generation depends on live
  (all_but_dep -2.7, still 97% of tokens frozen). That is the "static vs live blocks" picture
  ("Selective Stateful Execution"), supported directionally, not yet at the preregistered bar.
- Adaptive depth stays KILLed. Block liveness is a live question but not yet earned: the primary
  gate did not pass and one of the two localization halves is not significant.
- 4b stays deferred.

## Recommended next step (not preregistered here; for decision)

The cells separate (identity vs all_but_dep is significant), so the setup is not too noisy to
study; it is underpowered for the -15 bar. Two options:

- **4a'' power replication** (recommended): identity, all_but_dep, full only; 100 new prompts
  GSM8K test[200:300] (`gsm8k_test_full.json` added, first 200 items identical to the frozen file);
  p in {10, 90}; identical format. Primary: same identity thresholds on the new prompts alone
  (n=200, CI about +-7). Secondary: identity - all_but_dep paired difference. Cost about 5 h on
  two GPUs. No pooling with 4a' unless declared as a sequential analysis in its preregistration.
- **Workload change** (if the reasoning-vs-copying distinction in finding 4 is the target): a
  task with a single-token answer copied from the block would test whether pure retrieval from
  frozen blocks is lossless, separating "read" from "reason". Cheap (short generations) but a
  different question.

## Deviations log

(empty)
