# Round 4a preregistration: adversarial (dependency-changing) mutations

Frozen 2026-09-11 before any Round 4a data. Falsification round: it can kill the candidate.
Characterization only; no validity mechanism is built or assumed.

## Question

Rounds 1-3 showed: large structural mutation does not invalidate cached step-0 state.
Every Round 3 mutation was semantically unrelated to the final question. The untested
attack is: a mutation the final answer *depends on*. Round 2 already produced one
counterexample (hint_append at short context). So:

> Is step-0 state reuse position-insensitive but dependency-sensitive?

## Workload

Same model, sampler, gen 128 / 64 steps / block 32, prompts GSM8K test[50:100], history
pool test[100:200], 12-turn 2048-token history as in Round 3.

Dependency turn: the GSM8K problem statement q is placed in the history as a user turn
("Here is a problem I want to solve later. Please just remember it for now: q") with a
short assistant acknowledgement, at relative position p in {10, 50}% of the history. The
final user turn does not restate the problem: "Now solve the problem I asked you to
remember earlier. Show your work and give the final answer." The answer therefore depends
on the content of the dependency turn, thousands of tokens back.

Conditions, all same-length in-place edits or pure moves so that structural severity is
matched and only semantic dependency varies:

| condition | mutation | answer | tests |
|---|---|---|---|
| ctrl_other | number edit in the unrelated turn adjacent to the dependency turn | preserved | Round 3 replication in this format |
| ctrl_name | name edit inside the dependency turn (same token count) | preserved | related content changed, answer preserved |
| dep_number | first number of the problem edited inside the dependency turn (same digit count) | **changes** | answer-determining fact changed |
| dep_move | dependency turn moved to the end of the history, unchanged | preserved | relocation of the *relevant* content |

Gold for dep_number is recomputed by re-evaluating GSM8K's calculator annotations
(`<<a op b = c>>`) with the edited number substituted and intermediate results propagated.
A prompt is eligible only if the unedited chain reproduces the original gold, the edited
number occurs exactly once in the question, and no operand is ambiguous (equal to both the
edited number and a previous intermediate result). Ineligible prompts still run; they are
scored by agreement only. Eligibility is decided by code before any generation.

Conditions per cell: full (k=0), ours (all mapped tokens, k=32, step-0 source), prefix
(common prefix only, k=32). Per prompt: 2 positions x (1 old + 4 x 3) = 26 generations.

## Primary metrics

- safe rate: P(answer_ours == answer_full), per condition, with 95% bootstrap CI.
- paired accuracy loss ours vs full (pp) where gold is known (all controls; dep_number on
  the eligible subset with recomputed gold).
- stale rate for dep_number: P(answer_ours == answer_old and answer_full != answer_old),
  the signature of serving the pre-mutation answer.
- prefix condition reported alongside as the structural lower bound.

Recorded for later offline analysis only (no mechanism this round): step-0 drift per layer
at mapped tokens, overall and restricted to the dependency turn's tokens.

## Preregistered readings

Let safe(c) be the safe rate of condition c pooled over p, and loss(c) the paired loss.

- **A (broadly tolerant)**: safe(dep_number) >= safe(ctrl_other) - 10 pp, stale rate
  (ours) <= stale rate (full) + 10 pp, and loss(dep_number, eligible) >= -5 pp. Reading:
  step-0 state is broadly mutation-tolerant; no validity policy needed; go to 4b.
- **B (dependency-sensitive boundary)**: safe(dep_number) <= safe(ctrl_other) - 20 pp with
  the CI of the difference excluding 0, while ctrl_other, ctrl_name and dep_move each have
  loss >= -5 pp (CI low >= -10). Reading: reuse is position-insensitive but
  dependency-sensitive; a reuse-vs-invalidate decision problem exists. Next step is to test
  whether cheap observables separate the conditions (offline, on the recorded drift), not
  to build a mechanism.
- **C (unstructured)**: anything else, in particular a control condition failing its loss
  bound or dep_move failing while ctrl passes. Reading: validity is not explained by
  dependency or position; the candidate is at risk and practical headroom must be
  re-estimated under conservative recompute.

dep_move is the direct test of the content-addressed thesis on relevant content. If
dep_move fails (loss < -5 pp, CI low < -10) while ctrl_other passes, the Round 3 promotion
is retracted regardless of A/B.

## Coverage and format notes (from the tokenizer-only dry run, before any generation)

- dep_number: the replacement number is chosen (same digit count) so that the re-evaluated
  gold is a changed positive integer; 34/50 prompts are gold-eligible, the other 16 run with
  gold = None (agreement and stale rate only).
- ctrl_name: a same-token-count name edit exists for 31/50 prompts (start-of-question name,
  or any capitalised non-sentence-initial non-stopword token); the rest skip ctrl_name.
- The final turn carries a digit-free topic hint (first words of the problem) so the model can
  locate the remembered problem; the hint never contains the edited number.
- The GPU sanity run (2 prompts, 8 steps) checks only that full-recompute accuracy in this
  "remember, then solve" format is not degenerate. If it is (< 0.3 on the sanity prompts),
  the FINAL wording may be adjusted once before the main run and logged below; nothing else
  may change.

## Not in this round

No invalidation mechanism, no adaptive depth, no remasking, no wall-clock. Round 4b
(system, wall-clock, faithful prefix-cache baseline) follows if A or B. 4c (mixed
commitment) stays deferred.

## Deviations log

(empty)
