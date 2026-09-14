# Wavefront headroom audit preregistration

Frozen 2026-09-13, before building the simulator or measuring anything. Audit only: no runtime is
implemented this round. Preregistered as a branch that can be cleanly stopped.

## The claim being audited

New diffusion workloads make a request a dynamic fork/join generation DAG rather than a linear
sequence of block work. The hypothesis is that flattening that DAG into independent requests
destroys a large share of achievable GPU efficiency, and that the recoverable part requires a
lineage-aware execution substrate (COW lineage state, wavefront fusion, criticality-fusion
scheduling) rather than a scheduler tweak.

## The objection this audit must answer first

Stated before any measurement, because it is the reason the branch could be dead on arrival:

> Two of the three proposed mechanisms may already exist under other names. Copy-on-write lineage
> state is what paged-attention prefix sharing already does for a shared ancestor. Fusing ready
> nodes with the same signature is what continuous batching already does, since sibling branches of
> a diffusion request have identical shapes and step semantics by construction. If so, the only
> genuinely new part is join-aware scheduling, which is ordinary DAG scheduling and was already
> ruled insufficiently novel.

A single oracle-vs-flatten number cannot answer this, because it would credit the new substrate with
savings that existing systems already capture. The audit is therefore designed to **decompose** the
gap, not to measure it.

## Scheduler ladder (the core design)

Each rung adds exactly one capability. All rungs share the same cost model, memory model and DAG
set, so consecutive differences are attributable.

| rung | scheduler | adds | attributable to |
|---|---|---|---|
| S0 | flatten + continuous batching, FCFS, ancestor state duplicated per branch | - | naive baseline |
| S1 | S0 + shared ancestor state | lineage store | existing prefix sharing / paged attention |
| S2 | S1 + SRPT | size-aware ordering | ordinary serving |
| S3 | S2 + critical-path / join-aware priority | DAG awareness | SAGA-style workflow scheduling |
| S4 | S3 + wavefront fusion of same-signature ready nodes | fusion | the novel claim |
| S5 | clairvoyant oracle: knows cancellations in advance, never runs doomed work, packs maximal compatible batches | - | optimistic upper bound, not optimal |

Gap decomposition, all as fraction of the S0 makespan:

```text
G_share  = S0 - S1     already solved elsewhere
G_order  = S1 - S2     already solved elsewhere
G_dag    = S2 - S3     already solved elsewhere (SAGA)
G_fuse   = S3 - S4     the novel mechanism
G_resid  = S4 - S5     what no proposed mechanism captures
```

## Decision rule

Primary quantity: `G_novel = (S3 - S5) / S3`, the share of S3's makespan that a strong,
already-publishable DAG-aware baseline still leaves on the table.

- **GO**: `G_novel >= 0.25` at the measured hardware cost model, in a region of DAG parameter space
  that real workloads plausibly occupy, and `G_fuse` is at least half of `G_novel` (otherwise the
  headroom is not attributable to the proposed mechanism). Next step is a real-trace parameter
  measurement, then implementation.
- **STOP**: `G_novel < 0.15` across the swept parameter space. The branch ends.
- Otherwise inconclusive, which for this branch means narrowing to the sub-region where
  `G_novel >= 0.25` and asking whether any real workload lives there before spending more.

The preregistered bar is deliberately on `S3 - S5`, not on `S0 - S5`. A large `S0 - S5` with a
small `S3 - S5` means the win belongs to prefix sharing and DAG scheduling, which are taken.

## Cost model (measured, not assumed)

`L(B, S)` = wall-clock latency of one LLaDA forward at batch size `B` and sequence length `S`,
measured on the target GPU with `wavefront_microbench.py`; a grid over `B` in {1,2,4,8,16,32} and
`S` in {512, 1024, 2048, 4096}, median of repeated trials after warmup, plus the peak-memory curve
that sets the token budget.

This microbenchmark is run **first** and can stop the branch on its own: if `L(B, S)` is close to
linear in `B` over the relevant range, then batching is not profitable at the margin, fusion cannot
pay, and `G_fuse` is near zero by construction. The preregistered early stop is:

> If the measured batching efficiency `E(B) = B * L(1,S) / L(B,S)` stays below 1.5 for every `B` at
> every `S` in the grid, report that and STOP without running the simulator.

## DAG parameter space

Rather than guessing the parameters of three specific systems, sweep the space and report where the
novel gap lives. Parameters, each swept over the listed grid:

- `fork_width` k: 2, 4, 8, 16
- `branch_steps` heterogeneity: coefficient of variation 0.0, 0.25, 0.5, 1.0 around a mean of 32
- `shared_prefix_len`: 512, 2048 tokens
- `branch_seq_len`: 256, 1024 tokens
- `cancel_frac` (speculative branches killed before completion): 0.0, 0.5, 0.75
- `fork_rounds` (sequential fork/join phases per request): 1, 4
- offered load: a fixed batch of N=64 requests released at t=0 (offline makespan), plus a Poisson
  arrival sweep for latency as a secondary metric

The three motivating systems map onto sub-regions of this space rather than being simulated
individually: planned-diffusion-like (k small, cancel 0, heterogeneous branch lengths),
trajectory-speculation-like (k large, cancel high, homogeneous), rollout-like (k moderate,
cancel high at a late join). Which sub-region each really occupies is a measurement for the next
round and is explicitly *not* claimed here.

## Metrics

Primary: makespan for the fixed request set, per scheduler. Secondary: useful GPU-seconds per
completed request, wasted GPU-seconds on cancelled branches, p50/p99 latency under Poisson arrivals,
peak resident tokens.

## Validity requirements

- The simulator's cost model is the measured `L(B, S)` table with linear interpolation; any
  extrapolation beyond the measured grid is flagged in the output and excluded from decisions.
- Memory: a token budget from the measured peak-memory curve; a schedule that exceeds it is
  infeasible and the scheduler must shed batch size, for every rung equally.
- Every rung is run on identical DAG instances with the same seed.
- A rung may never use information a real implementation could not have, except S5, which is
  labelled clairvoyant everywhere it appears.

## Not in this round

No runtime, no kernel work, no real-system traces, no claim about which workload occupies which
region of the parameter space.

## Deviations log

(empty)
