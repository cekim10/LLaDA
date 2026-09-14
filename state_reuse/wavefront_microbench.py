"""
Wavefront audit step 0 (WAVEFRONT_PREREG.md): measure the LLaDA forward cost model L(B, S)
and the peak-memory curve that sets the token budget.

This runs BEFORE the simulator and can stop the branch on its own: if batching efficiency
E(B) = B * L(1,S) / L(B,S) never reaches 1.5, fusion cannot pay and the audit stops.

  python wavefront_microbench.py --out results/wavefront/cost_model.json
"""
import argparse, json, os, statistics, time

import torch
from transformers import AutoModel

MASK_ID = 126336


def time_forward(model, B, S, device, iters, warmup):
    x = torch.full((B, S), MASK_ID, dtype=torch.long, device=device)
    try:
        with torch.no_grad():
            for _ in range(warmup):
                model(x)
            if device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            ts = []
            for _ in range(iters):
                t0 = time.perf_counter()
                model(x)
                if device == "cuda":
                    torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
        peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else float("nan")
        return statistics.median(ts), peak, None
    except RuntimeError as e:
        msg = str(e)
        if device == "cuda":
            torch.cuda.empty_cache()
        return float("nan"), float("nan"), ("oom" if "out of memory" in msg.lower() else msg[:120])
    finally:
        del x
        if device == "cuda":
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--batches", default="1,2,4,8,16,32")
    ap.add_argument("--seqlens", default="512,1024,2048,4096")
    ap.add_argument("--iters", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results", "wavefront", "cost_model.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={device}", flush=True)
    if device == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)} mem={torch.cuda.get_device_properties(0).total_memory/2**30:.0f}GiB", flush=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()

    Bs = [int(b) for b in args.batches.split(",")]
    Ss = [int(s) for s in args.seqlens.split(",")]
    grid = {}
    for S in Ss:
        for B in Bs:
            lat, peak, err = time_forward(model, B, S, device, args.iters, args.warmup)
            grid[f"{B}x{S}"] = {"B": B, "S": S, "latency_s": lat, "peak_gib": peak, "error": err}
            tok = B * S
            if err:
                print(f"  B={B:3d} S={S:5d}  {err}", flush=True)
            else:
                print(f"  B={B:3d} S={S:5d}  {1000*lat:8.1f} ms   {tok/lat/1000:8.1f} Ktok/s   peak {peak:5.1f} GiB", flush=True)

    # batching efficiency E(B) = B * L(1,S) / L(B,S)
    eff = {}
    for S in Ss:
        base = grid.get(f"1x{S}", {}).get("latency_s")
        if not base or base != base:
            continue
        eff[str(S)] = {}
        for B in Bs:
            l = grid.get(f"{B}x{S}", {}).get("latency_s")
            if l and l == l:
                eff[str(S)][str(B)] = B * base / l
    best_eff = max((v for d in eff.values() for v in d.values()), default=float("nan"))

    out = {
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else device,
        "grid": grid,
        "batching_efficiency": eff,
        "max_batching_efficiency": best_eff,
        "early_stop_triggered": bool(best_eff == best_eff and best_eff < 1.5),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print("\nbatching efficiency  E(B) = B*L(1,S)/L(B,S)   (1.0 = batching buys nothing)")
    hdr = "  S\\B  " + "".join(f"{b:>8d}" for b in Bs)
    print(hdr)
    for S in Ss:
        row = eff.get(str(S), {})
        print(f"  {S:5d}" + "".join(f"{row[str(b)]:8.2f}" if str(b) in row else f"{'-':>8s}" for b in Bs))
    print(f"\nmax batching efficiency: {best_eff:.2f}")
    print("PREREGISTERED EARLY STOP TRIGGERED" if out["early_stop_triggered"]
          else "early stop not triggered; proceed to the simulator")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
