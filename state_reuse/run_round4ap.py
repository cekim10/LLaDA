"""
Round 4a' (ROUND4AP_PREREG.md): no mutation, P' = P. Freeze subsets of the request's own step-0
state at every denoising step and measure retrieval-task accuracy.

  conditions: full | identity | all_but_dep | dep_only | dep_fresh_num      (k = 32)

Usage:
  python run_round4ap.py --start 50 --n 50 --pos 10,50,90 --out results/round4ap
  python run_round4ap.py --dry_run
"""
import argparse, json, os, re, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deltas import chat_prompt, gold_answer, extract_pred
from run_round3 import encode_ids, build_history, decode_gen
from run_round4a import REMEMBER, ACK, find_span

EOT = 126348
EOS = 126081

FINAL = ("Now solve the problem I asked you to remember earlier{hint}. "
         "Do not restate the problem; go straight to the solution and give the final answer.")


def final_turn(q):
    words = q.split()
    for n in (6, 5, 4, 3):
        head = " ".join(words[:n])
        if not re.search(r"\d", head):
            return FINAL.format(hint=f' (the one that begins with "{head}...")')
    return FINAL.format(hint="")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "gsm8k_test200.json"))
    ap.add_argument("--start", type=int, default=50)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--hist_start", type=int, default=100)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--pos", default="10,50,90")
    ap.add_argument("--conds", default="identity,all_but_dep,dep_only,dep_fresh_num")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--gen_length", type=int, default=256)
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results", "round4ap"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    data = json.load(open(args.data))
    prompts = data[args.start:args.start + args.n]
    pool = data[args.hist_start:]
    positions = [int(p) for p in args.pos.split(",") if p]
    conds = [c for c in args.conds.split(",") if c]
    turns, _ = build_history(tok, pool, args.ctx)
    T = len(turns)
    print(f"history: {T} turns", flush=True)

    if not args.dry_run:
        import torch
        from transformers import AutoModel
        from instrument import StateRecorder
        from sampler import generate
        device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        store_device = "cuda" if device == "cuda" else "cpu"
        print(f"device={device} store={store_device}", flush=True)
        model = AutoModel.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
        rec = StateRecorder(model, store_device=store_device)
        gen_kw = dict(steps=args.steps, gen_length=args.gen_length, block_length=args.block_length, record_canvas=False)
        os.makedirs(args.out, exist_ok=True)
        fout = open(os.path.join(args.out, "records.jsonl"), "a")

        def emit(r):
            fout.write(json.dumps(r) + "\n"); fout.flush()

    for qi, item in enumerate(prompts):
        qid = args.start + qi
        q, gold = item["q"], gold_answer(item["a"])
        dep_turn = ({"role": "user", "content": REMEMBER.format(q=q)}, {"role": "assistant", "content": ACK})
        for p in positions:
            t = min(T, max(0, round(p / 100 * T)))
            hist = turns[:t] + [dep_turn] + turns[t:]
            msgs = [m for tt in hist for m in tt] + [{"role": "user", "content": final_turn(q)}]
            ids_l = encode_ids(tok, chat_prompt(tok, msgs))
            P = len(ids_l)
            span = find_span(ids_l, encode_ids(tok, REMEMBER.format(q=q)))
            if span is None:
                print(f"  skip q{qid} p{p}: problem span not found", flush=True); continue
            dep_idx = list(range(span[0], span[1]))
            num_idx = [i for i in dep_idx if re.search(r"\d", tok.decode([ids_l[i]]))]
            all_idx = list(range(P))
            sets = {
                "identity": all_idx,
                "all_but_dep": [i for i in all_idx if not (span[0] <= i < span[1])],
                "dep_only": dep_idx,
                "dep_fresh_num": [i for i in all_idx if i not in set(num_idx)],
            }
            info = dict(p=p, dep_turn=t, P=P, dep_span=list(span), n_dep=len(dep_idx), n_num=len(num_idx))
            if args.dry_run:
                print(f"  q{qid} p{p:2d} P={P} dep_span={span} n_dep={len(dep_idx)} n_num={len(num_idx)} "
                      f"nums={[tok.decode([ids_l[i]]) for i in num_idx]}", flush=True)
                continue

            t_start = time.time()
            ids = torch.tensor(ids_l, device=device).unsqueeze(0)
            rec.start_record(torch.arange(P, device=device), steps={0})
            out_full = generate(model, ids, recorder=rec, **gen_kw)
            rec.stop()
            h0 = rec.traj[0]  # [L+1, P, d]
            full_ids = out_full["x"][0, P:].cpu()
            text_full = decode_gen(tok, full_ids); pred_full = extract_pred(text_full)
            n_gen_full = int(((full_ids != EOS) & (full_ids != EOT)).sum())
            emit(dict(qid=qid, kind="full", **info, gold=gold, pred=pred_full, correct=(pred_full == gold),
                      n_gen=n_gen_full, text=text_full))
            print(f"[q{qid} p{p}] full: pred={pred_full} gold={gold} n_gen={n_gen_full} ({time.time()-t_start:.0f}s)", flush=True)

            for cond in conds:
                idx = sets[cond]
                if not idx:
                    continue
                idx_t = torch.tensor(idx, device=device)
                rec.start_inject(idx_t, {0: h0[:, idx, :]}, args.k, source="first")
                o = generate(model, ids, recorder=rec, **gen_kw)
                rec.stop()
                ids_c = o["x"][0, P:].cpu()
                text_c = decode_gen(tok, ids_c); pred_c = extract_pred(text_c)
                emit(dict(qid=qid, kind=cond, **info, k=args.k, gold=gold, pred=pred_c, correct=(pred_c == gold),
                          agree_full=(pred_c == pred_full), tok_agree=float((ids_c == full_ids).float().mean()),
                          n_frozen=len(idx), frozen_frac=len(idx) / P,
                          n_gen=int(((ids_c != EOS) & (ids_c != EOT)).sum()), text=text_c))
                print(f"    {cond:13s} frozen={len(idx)}/{P} pred={pred_c} correct={pred_c == gold} agree={pred_c == pred_full}", flush=True)

            del h0
            if device == "cuda":
                torch.cuda.empty_cache()
            print(f"[q{qid} p{p}] done in {time.time()-t_start:.0f}s", flush=True)

    if not args.dry_run:
        fout.close()
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
