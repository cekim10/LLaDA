# Round 4a seal (2026-09-11): reading C, with a specific structure

Preregistered design completed: 50 prompts x p in {10, 50} x {ctrl_other, ctrl_name, dep_number,
dep_move} x {full, ours k=32, prefix k=32}; 1174 records, no crashes. ctrl_name ran on 31/50 prompts,
dep_number gold-eligible 34/50 (68 cells). Old-request accuracy in this "remember, then solve"
format: 0.63; full recompute after control mutations: 0.66-0.72. Tables: `results/round4a/ROUND4A_RESULTS.md`.

Note: the first analyzer version printed reading A because it did not require the control conditions
to pass; ROUND4A_PREREG.md defines a failing control as reading C. Fixed before this seal was written.

## Preregistered reading: C (unstructured)

| condition | what is frozen (ours) | loss ours (pp) | loss prefix (pp) | safe ours | stale ours |
|---|---|---|---|---|---|
| ctrl_other | everything but one number in an unrelated turn | **-14.0 [-25.0, -3.0]** | -11.0 [-21.0, -1.0] | 0.56 | 0.04 |
| ctrl_name | everything but a name in the problem | -9.7 [-24.2, +4.8] | -3.2 [-11.3, +4.8] | 0.55 | 0.08 |
| dep_number | everything but the answer-determining number | **+2.9 [-5.9, +11.8]** | +2.9 [-7.4, +13.2] | 0.47 | 0.02 |
| dep_move | everything, problem turn relocated ~2000 positions | **-18.0 [-29.0, -7.0]** | -3.0 [-12.0, +6.0] | 0.50 | 0.04 |

safe(dep_number) - safe(ctrl_other) = -9.1 pp [-23.5, +4.2]. Reading A's dep_number criteria are all
met, but two control conditions fail their loss bound, which the preregistration defines as C.

## What the data actually says

1. **The adversarial condition did not fail.** Editing the number the answer depends on costs
   nothing (+2.9 pp), and the stale-answer signature is essentially absent (0.02 vs full's own
   0.15 rate of keeping the old answer). Frozen step-0 states of the unchanged problem tokens do not
   carry the old number. "Dependency-changing mutation" is not the failure mode.

2. **The controls failed instead, and the loss localizes to freezing the problem turn.**
   ctrl_other at p=10: `prefix` freezes only turn 0 plus the problem turn (17% of tokens, states
   bit-for-bit within cos 0.9997 of full's own step-0 states) and still loses -18 pp, almost all of
   ours' -24. At p=50 both lose -4. dep_move: `prefix` excludes the problem turn and loses -3; ours
   includes it (relocated) and loses -18. Round 3's identical edit type cost +2 pp when the history
   was decorative. The boundary is therefore not the mutation's semantics and not position per se,
   but **whether the generation must retrieve from content whose state is frozen**.

3. **The apparent paradox is the most informative cell.** dep_number at p=10 freezes the same problem
   turn except for the 1-2 edited number tokens, which are recomputed at every denoising step, and
   loses nothing (+8.8 pp) where ctrl_other loses -24. Post-hoc hypotheses, to be tested and not
   assumed: (i) retrieved content must co-evolve with the canvas across denoising steps
   (bidirectional attention: the problem's deep states change as the solution is decoded), and a
   step-0 snapshot cannot supply that; (ii) a single fresh token inside a frozen block is enough of a
   "hook". Distance is implicated but unresolved: adjacent question frozen (Round 2, ctx 2048) about
   0 pp; 1100 tokens (p=50) -4; 2000 tokens (p=10) -18 to -24; CIs are +-14.

4. **Relocating relevant content reorganizes the whole state.** In dep_move every other token's
   step-0 state moves by cos 0.80-0.88 (drift 0.12-0.20 at layer 32), against 0.02 for Round 3's
   reorder of an irrelevant turn. When the final turn points at the moved content, global attention
   reorganizes around its new position, and frozen states are wrong everywhere. Relocation
   tolerance (Round 3) does not extend to content the generation depends on.

5. **Format confounds, symmetric across conditions.** 96-100% of outputs in every condition,
   including full and old, fill the 128-token canvas because the model restates the problem first
   (40-70% of outputs); the answer extractor (last number) is therefore noisy for everyone. This
   inflates variance but does not explain paired losses of 14-24 pp, and dep_number is unaffected.
   The design also lacked an identity control (P' = P, freeze everything, no mutation), which is
   the cleanest test of "freezing vs co-evolution".

## Consequences for the claims

- Round 3 promotion: the retraction clause (dep_move fails while ctrl_other passes) did not fire
  because ctrl_other failed too. The claim is **bounded**, not retracted: the reuse boundary need not
  coincide with the prefix boundary *when the generation does not need to retrieve from the frozen
  content at long range*. Rounds 2-3 satisfied that condition by construction (decorative history,
  self-contained question). Round 4a violates it and loses 14-24 pp under full-depth freezing.
- KILL for "semantic-aware adaptive depth" stands. A different decision problem has appeared and
  is not killed: **which cached blocks does this generation depend on?** Characterized only; no
  mechanism built.
- 4b (real system) is premature until the boundary is pinned down; the wall-clock story would be
  for a workload class whose validity condition is not yet defined.

## Proposed Round 4a' (to be preregistered separately; cheap, about 2 GPU-hours)

Same format, gen 256 and an explicit "do not restate the problem" instruction (logged as a format
change), p in {10, 50, 90}, n = 50, k = 32:

| condition | frozen set | isolates |
|---|---|---|
| identity | everything, no mutation (P' = P) | pure cost of freezing vs co-evolution |
| all_but_dep | everything except the problem turn (recomputed every step) | whether one fresh block restores retrieval |
| dep_only | only the problem turn | whether freezing the retrieved block alone is the loss |
| dep_fresh_num | everything except the problem's number tokens | the "hook" hypothesis from finding 3 |

Predictions written down now: identity at p=10 loses about as much as ctrl_other did (-15 to -25 pp);
all_but_dep loses about 0; dep_only loses about as much as identity; dep_fresh_num loses about 0 if
the hook hypothesis is right and about as much as identity if the co-evolution hypothesis is right.
Decision rule: if identity loses <= 5 pp at every p the 4a losses were mutation-specific and 4a is
re-read; if identity loses >= 15 pp at p=10 the validity condition is "no long-range retrieval from
frozen blocks" and the next step is to characterize whether retrieval dependence is cheaply
observable (attention mass from the canvas to each cached block at step 0 is the obvious candidate,
recorded but not acted on).
