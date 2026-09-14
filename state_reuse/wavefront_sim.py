"""
Wavefront headroom audit simulator (WAVEFRONT_PREREG.md).

Batch-synchronous discrete-event simulation of diffusion fork/join DAGs under the preregistered
scheduler ladder S0..S5, reporting the gap decomposition
G_share / G_order / G_dag / G_fuse / G_resid and the primary quantity G_novel = (S3-S5)/S3.

Compute model, stated explicitly because it decides the answer:
  * A diffusion step recomputes the whole sequence (LLaDA has no KV cache for MDM), so sibling
    branches CANNOT share prefix compute. The lineage store therefore saves memory, not compute.
  * A batch is padded to its longest member: cost = L(m, max_S). Mixing shapes wastes compute,
    which is the only mechanism by which signature-homogeneous "fusion" can beat plain continuous
    batching. That makes S3 -> S4 a real, modelled effect rather than a relabelling.

  python wavefront_sim.py --cost-model results/wavefront/cost_model.json --out results/wavefront
"""
import argparse, itertools, json, os, random, statistics

SCHEDULERS = ["S0", "S1", "S2", "S3", "S4", "S5"]


# ---------------------------------------------------------------- cost model
class Cost:
    def __init__(self, path=None, b_max=32, mem_gib=80.0):
        self.measured = False
        self.b_max = b_max
        self.table = {}
        self.token_budget = None
        if path and os.path.exists(path):
            raw = json.load(open(path))
            for cell in raw["grid"].values():
                if cell.get("error") or cell["latency_s"] != cell["latency_s"]:
                    continue
                self.table[(cell["B"], cell["S"])] = cell["latency_s"]
                if cell["peak_gib"] == cell["peak_gib"] and cell["peak_gib"] < mem_gib:
                    self.token_budget = max(self.token_budget or 0, cell["B"] * cell["S"])
            self.measured = bool(self.table)
            if self.measured:
                self.b_max = max(b for b, _ in self.table)
        if not self.measured:
            self.token_budget = 128 * 1024
        self.Bs = sorted({b for b, _ in self.table}) or [1, 2, 4, 8, 16, 32]
        self.Ss = sorted({s for _, s in self.table}) or [512, 1024, 2048, 4096]

    def __call__(self, m, S):
        """Latency of one padded forward of m sequences of length S."""
        if not self.measured:
            # placeholder only; results are marked UNVALIDATED when this path is used
            return (0.004 + 0.0009 * m) * (S / 1024.0)
        return self._interp(max(1, m), S)

    def _interp(self, m, S):
        def at(b, s):
            if (b, s) in self.table:
                return self.table[(b, s)]
            bs = [x for x in self.Bs if (x, s) in self.table]
            if not bs:
                return None
            lo = max([x for x in bs if x <= b], default=bs[0])
            hi = min([x for x in bs if x >= b], default=bs[-1])
            if lo == hi:
                return self.table[(lo, s)] * (b / lo)
            t = (b - lo) / (hi - lo)
            return self.table[(lo, s)] * (1 - t) + self.table[(hi, s)] * t
        lo = max([x for x in self.Ss if x <= S], default=self.Ss[0])
        hi = min([x for x in self.Ss if x >= S], default=self.Ss[-1])
        a, b_ = at(m, lo), at(m, hi)
        if a is None or b_ is None:
            return (a or b_ or 0.01) * (S / (lo if a else hi))
        if lo == hi:
            return a
        t = (S - lo) / (hi - lo)
        return a * (1 - t) + b_ * t


# ---------------------------------------------------------------- DAG
class Branch:
    __slots__ = ("rid", "phase", "bid", "steps", "seq", "done", "cancel_at", "alive")

    def __init__(self, rid, phase, bid, steps, seq, cancel_at):
        self.rid, self.phase, self.bid = rid, phase, bid
        self.steps, self.seq = steps, seq
        self.done = 0
        self.cancel_at = cancel_at      # None, or the step at which a verifier kills it
        self.alive = True

    @property
    def remaining(self):
        end = self.cancel_at if self.cancel_at is not None else self.steps
        return max(0, end - self.done)

    @property
    def finished(self):
        end = self.cancel_at if self.cancel_at is not None else self.steps
        return self.done >= end


SEQ_MIX = [256, 1024, 4096]


def make_requests(p, n, rng):
    reqs = []
    for rid in range(n):
        # shape heterogeneity across concurrent requests is the ONLY channel by which
        # signature-homogeneous fusion can beat plain continuous batching under this cost model
        seq = rng.choice(SEQ_MIX) if p["seq_mix"] else p["branch_seq_len"]
        phases = []
        for ph in range(p["fork_rounds"]):
            branches = []
            for b in range(p["fork_width"]):
                mu, cv = p["branch_steps_mean"], p["branch_steps_cv"]
                steps = max(1, int(round(rng.gauss(mu, mu * cv)))) if cv > 0 else mu
                cancel = None
                if b > 0 and rng.random() < p["cancel_frac"]:
                    cancel = max(1, int(steps * rng.uniform(0.2, 0.8)))
                branches.append(Branch(rid, ph, b, steps, seq, cancel))
            phases.append(branches)
        reqs.append({"rid": rid, "phases": phases, "phase": 0, "prefix": p["shared_prefix_len"]})
    return reqs


# ---------------------------------------------------------------- simulation
def simulate(reqs, cost, sched, p):
    """Batch-synchronous loop: form one batch of ready branch-steps, advance time by its cost."""
    share = sched != "S0"
    clairvoyant = sched == "S5"
    homogeneous = sched in ("S4", "S5")
    t = 0.0
    useful = wasted = 0.0
    finish = {}
    n_batches = 0
    peak_tokens = 0

    if clairvoyant:  # never execute work a verifier will discard
        for r in reqs:
            for ph in r["phases"]:
                for b in ph:
                    if b.cancel_at is not None:
                        b.cancel_at = 0

    while True:
        ready = []
        for r in reqs:
            if r["phase"] >= len(r["phases"]):
                continue
            for b in r["phases"][r["phase"]]:
                if not b.finished:
                    ready.append((r, b))
        if not ready:
            break

        # memory: resident tokens of every live branch; with sharing the prefix is counted once
        resident = 0
        seen = set()
        for r, b in ready:
            resident += b.seq
            if share:
                if r["rid"] not in seen:
                    resident += r["prefix"]
                    seen.add(r["rid"])
            else:
                resident += r["prefix"]
        peak_tokens = max(peak_tokens, resident)

        # priority
        if sched == "S2":
            ready.sort(key=lambda rb: rb[1].remaining)
        elif sched in ("S3", "S4", "S5"):
            # critical path: the phase join waits for the longest sibling, so prioritise the
            # branch that determines the join time, longest-remaining first within a request,
            # requests with the longest remaining critical path first
            crit = {}
            for r in reqs:
                tot = 0
                for ph in range(r["phase"], len(r["phases"])):
                    tot += max((b.remaining for b in r["phases"][ph]), default=0)
                crit[r["rid"]] = tot
            ready.sort(key=lambda rb: (-crit[rb[0]["rid"]], -rb[1].remaining))

        # batch formation
        if homogeneous:
            groups = {}
            for r, b in ready:
                groups.setdefault(b.seq, []).append((r, b))
            key = max(groups, key=lambda k: len(groups[k]))
            pool = groups[key]
        else:
            pool = ready

        batch, tokens = [], 0
        for r, b in pool:
            if len(batch) >= cost.b_max:
                break
            need = b.seq + (0 if (share and r["rid"] in {x[0]["rid"] for x in batch}) else r["prefix"])
            if tokens + need > cost.token_budget and batch:
                break
            batch.append((r, b))
            tokens += need
        if not batch:
            batch = [pool[0]]

        S = max(b.seq + r["prefix"] for r, b in batch)
        dt = cost(len(batch), S)
        t += dt
        n_batches += 1
        for r, b in batch:
            b.done += 1
            if b.cancel_at is not None:
                wasted += dt / len(batch)
            else:
                useful += dt / len(batch)

        # advance phases and record completion
        for r in reqs:
            if r["phase"] < len(r["phases"]) and all(b.finished for b in r["phases"][r["phase"]]):
                r["phase"] += 1
                if r["phase"] >= len(r["phases"]):
                    finish[r["rid"]] = t

    lat = sorted(finish.values())
    return {
        "makespan": t, "n_batches": n_batches, "useful_gpu_s": useful, "wasted_gpu_s": wasted,
        "peak_tokens": peak_tokens,
        "p50_latency": statistics.median(lat) if lat else float("nan"),
        "p99_latency": lat[min(len(lat) - 1, int(0.99 * len(lat)))] if lat else float("nan"),
    }


def run_point(p, cost, n_requests, seed):
    out = {}
    for sched in SCHEDULERS:
        rng = random.Random(seed)
        out[sched] = simulate(make_requests(p, n_requests, rng), cost, sched, p)
    m = {s: out[s]["makespan"] for s in SCHEDULERS}
    gaps = {
        "G_share": (m["S0"] - m["S1"]) / m["S0"],
        "G_order": (m["S1"] - m["S2"]) / m["S0"],
        "G_dag": (m["S2"] - m["S3"]) / m["S0"],
        "G_fuse": (m["S3"] - m["S4"]) / m["S0"],
        "G_resid": (m["S4"] - m["S5"]) / m["S0"],
        "G_total": (m["S0"] - m["S5"]) / m["S0"],
        "G_novel": (m["S3"] - m["S5"]) / m["S3"],
        "G_fuse_share_of_novel": ((m["S3"] - m["S4"]) / (m["S3"] - m["S5"])) if m["S3"] > m["S5"] else float("nan"),
    }
    return {"makespan": m, "detail": out, "gaps": gaps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost-model")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results", "wavefront"))
    ap.add_argument("--requests", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="small grid for a smoke test")
    args = ap.parse_args()

    cost = Cost(args.cost_model)
    tag = "MEASURED" if cost.measured else "UNVALIDATED (placeholder cost model; decisions require --cost-model)"
    print(f"cost model: {tag}; b_max={cost.b_max} token_budget={cost.token_budget}", flush=True)

    grid = dict(
        fork_width=[2, 4, 8, 16], branch_steps_cv=[0.0, 0.25, 0.5, 1.0],
        shared_prefix_len=[512, 2048], branch_seq_len=[256, 1024], seq_mix=[False, True],
        cancel_frac=[0.0, 0.5, 0.75], fork_rounds=[1, 4],
    )
    if args.quick:
        grid = dict(fork_width=[4, 16], branch_steps_cv=[0.0, 1.0], shared_prefix_len=[2048],
                    branch_seq_len=[1024], seq_mix=[False, True], cancel_frac=[0.0, 0.75], fork_rounds=[1])
    keys = list(grid)
    rows = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        p = dict(zip(keys, combo), branch_steps_mean=32)
        r = run_point(p, cost, args.requests, args.seed)
        rows.append({"params": p, **r["gaps"], "makespan": r["makespan"]})
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "sim_sweep.json"), "w") as f:
        json.dump({"cost_model_measured": cost.measured, "rows": rows}, f, indent=2)

    def q(vals, p_):
        v = sorted(vals)
        return v[min(len(v) - 1, int(p_ * len(v)))]

    novel = [r["G_novel"] for r in rows]
    fuse = [r["G_fuse"] for r in rows]
    print(f"\n{len(rows)} parameter points, {args.requests} requests each")
    print("\ngap decomposition, as a fraction of the S0 makespan (median over the sweep):")
    for g in ("G_share", "G_order", "G_dag", "G_fuse", "G_resid", "G_total"):
        print(f"  {g:10s} {100*statistics.median(r[g] for r in rows):6.1f}%")
    print(f"\nG_novel = (S3-S5)/S3   median {100*statistics.median(novel):.1f}%"
          f"   p90 {100*q(novel,0.9):.1f}%   max {100*max(novel):.1f}%")
    best = max(rows, key=lambda r: r["G_novel"])
    print(f"  best point: {best['params']}")
    print(f"    G_novel {100*best['G_novel']:.1f}%  G_fuse {100*best['G_fuse']:.1f}%"
          f"  fuse share of novel {100*best['G_fuse_share_of_novel']:.0f}%")
    frac_go = sum(1 for r in rows if r["G_novel"] >= 0.25 and (r["G_fuse_share_of_novel"] or 0) >= 0.5) / len(rows)
    decision = ("GO" if frac_go > 0 and max(novel) >= 0.25
                else "STOP" if max(novel) < 0.15 else "INCONCLUSIVE")
    print(f"\nparameter points meeting the GO condition: {100*frac_go:.0f}%")
    print(f"DECISION: {decision}" + ("" if cost.measured else "   [UNVALIDATED cost model - not a decision]"))


if __name__ == "__main__":
    main()
