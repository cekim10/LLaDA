"""
Round 2: oracle headroom and decision structure of step-0 prefix-state reuse
under long shared context.  See ROUND2_PREREG.md (Amendment A1).

For each GSM8K question q (old request P) and shared-context length C:
  history  = fixed multi-turn GSM8K Q/A history of ~C tokens (same for all prompts)
  P        = history + [user q]              -> full generation, record step-0 states
  P+D      = history + delta(q)              -> full recompute (reference), record step-0 states
             -> cheap features: |D|, lexical overlap, embedding cosine, step-0 drift per layer
  reuse(k) = P+D with layers 0..k-1 of the shared prefix taken from P's step-0 states
             ('first' source), k sweep -> answer, accuracy, agreement, compute saved

Usage:
  python run_round2.py --start 50 --n 50 --ctx 0,256,512,1024,2048 --ks 4,8,12,16,20,24,28,32 --out results/round2
"""
import argparse, json, os, re, sys, time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from instrument import StateRecorder, shared_positions
from sampler import generate, MASK_ID
from deltas import build_variants, chat_prompt, gold_answer, extract_pred

EOT = 126348
EOS = 126081


def decode_gen(tok, ids):
    ids = [int(i) for i in ids if int(i) not in (EOS, EOT, MASK_ID)]
    return tok.decode(ids, skip_special_tokens=True)


def encode_ids(tok, text):
    return tok(text, add_special_tokens=False)["input_ids"]


def clean_solution(a):
    body, _, ans = a.rpartition("####")
    body = re.sub(r"<<[^>]*>>", "", body).strip()
    return body + f"\nThe answer is {ans.strip()}."


def build_history(tok, pool, target_tokens):
    """Fixed multi-turn Q/A history with at least target_tokens tokens (0 -> empty)."""
    msgs = []
    if target_tokens <= 0:
        return msgs
    for item in pool:
        msgs += [{"role": "user", "content": item["q"]},
                 {"role": "assistant", "content": clean_solution(item["a"])}]
        n = len(encode_ids(tok, tok.apply_chat_template(msgs, tokenize=False)))
        if n >= target_tokens:
            break
    return msgs


def common_prefix_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))


def mean_emb(wte, ids, device):
    if len(ids) == 0:
        return None
    with torch.no_grad():
        e = wte(torch.tensor(ids, device=device)).float().mean(0)
    return e


def cos(a, b):
    if a is None or b is None:
        return None
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "gsm8k_test200.json"))
    ap.add_argument("--start", type=int, default=50, help="Round 2 prompts = data[start:start+n]")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--hist_start", type=int, default=100, help="history pool = data[hist_start:]")
    ap.add_argument("--ctx", default="0,256,512,1024,2048")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--gen_length", type=int, default=128)
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--ks", default="4,8,12,16,20,24,28,32")
    ap.add_argument("--deltas", default="irrelevant_append,hint_append,number_edit,name_edit,multiturn_append")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results", "round2"))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    store_device = "cuda" if device == "cuda" else "cpu"
    os.makedirs(args.out, exist_ok=True)
    ks = [int(k) for k in args.ks.split(",") if k != "" and int(k) > 0]
    ctxs = [int(c) for c in args.ctx.split(",") if c != ""]
    delta_names = args.deltas.split(",")

    print(f"device={device} store={store_device}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    inner = model.model if hasattr(model, "model") else model
    wte = inner.transformer.wte
    rec = StateRecorder(model, store_device=store_device)
    L = rec.n_layers

    data = json.load(open(args.data))
    prompts = data[args.start:args.start + args.n]
    pool = data[args.hist_start:]
    histories = {c: build_history(tok, pool, c) for c in ctxs}
    for c, h in histories.items():
        n = len(encode_ids(tok, tok.apply_chat_template(h, tokenize=False))) if h else 0
        print(f"ctx target {c}: {len(h)//2} Q/A turns, {n} tokens", flush=True)

    gen_kw = dict(steps=args.steps, gen_length=args.gen_length, block_length=args.block_length, record_canvas=False)
    fout = open(os.path.join(args.out, "records.jsonl"), "a")

    def emit(r):
        fout.write(json.dumps(r) + "\n"); fout.flush()

    for qi, item in enumerate(prompts):
        qid = args.start + qi
        q, gold = item["q"], gold_answer(item["a"])
        for C in ctxs:
            t_start = time.time()
            hist = histories[C]
            hist_ids = encode_ids(tok, tok.apply_chat_template(hist, tokenize=False)) if hist else []

            # ---------------- old request P -----------------------------------
            msgs_old = hist + [{"role": "user", "content": q}]
            ids_old_l = encode_ids(tok, chat_prompt(tok, msgs_old))
            ids_old = torch.tensor(ids_old_l, device=device).unsqueeze(0)
            P_old = ids_old.shape[1]
            hist_len = common_prefix_len(hist_ids, ids_old_l) if hist_ids else 0
            q_ids = ids_old_l[hist_len:]

            rec.start_record(torch.arange(P_old, device=device), steps={0})
            out_old = generate(model, ids_old, recorder=rec, **gen_kw)
            rec.stop()
            h_old0 = rec.traj[0]  # [L+1, P_old, d] step-0 states of P
            text_old = decode_gen(tok, out_old["x"][0, P_old:])
            pred_old = extract_pred(text_old)
            emit(dict(qid=qid, kind="old", ctx=C, ctx_tokens=hist_len, P=P_old, gold=gold, pred=pred_old,
                      correct=(pred_old == gold), text=text_old))
            print(f"[q{qid} ctx{C}] old: P={P_old} pred={pred_old} gold={gold} ({time.time()-t_start:.0f}s)", flush=True)

            variants = build_variants(tok, q, gold, text_old)
            for dname in delta_names:
                if dname not in variants:
                    continue
                msgs_d, gold_new, meta = variants[dname]
                msgs_new = hist + msgs_d
                ids_new_l = encode_ids(tok, chat_prompt(tok, msgs_new))
                ids_new = torch.tensor(ids_new_l, device=device).unsqueeze(0)
                P_new = ids_new.shape[1]
                sh = shared_positions(ids_old_l, ids_new_l)
                if len(sh) < 4:
                    print(f"  skip {dname}: too few shared positions", flush=True); continue
                sh_t = torch.tensor(sh, device=device)
                changed_new = sorted(set(range(P_new)) - set(sh))
                d_ids = [ids_new_l[i] for i in changed_new]
                traj_old = {0: h_old0[:, sh, :]}

                # ---------------- full recompute of P+D (reference) --------
                rec.start_record(sh_t, steps={0})
                out_new = generate(model, ids_new, recorder=rec, **gen_kw)
                rec.stop()
                h_new0 = rec.traj[0]  # [L+1, n_shared, d]
                sims0 = torch.nn.functional.cosine_similarity(traj_old[0].float(), h_new0.float(), dim=-1).cpu()  # [L+1, n_shared]
                sh_np = np.array(sh)
                q_mask = sh_np >= hist_len
                drift_all = (1 - sims0.mean(1)).tolist()
                drift_q = (1 - sims0[:, torch.tensor(q_mask)].mean(1)).tolist() if q_mask.any() else None
                full_ids = out_new["x"][0, P_new:].cpu()
                text_new = decode_gen(tok, full_ids)
                pred_new = extract_pred(text_new)
                feats = dict(
                    n_delta=len(d_ids), n_shared=len(sh), P_new=P_new, hist_len=hist_len,
                    jaccard_q=jaccard(d_ids, q_ids), jaccard_prefix=jaccard(d_ids, [ids_old_l[i] for i in sh]),
                    emb_cos_q=cos(mean_emb(wte, d_ids, device), mean_emb(wte, q_ids, device)),
                    emb_cos_prefix=cos(mean_emb(wte, d_ids, device), mean_emb(wte, [ids_old_l[i] for i in sh], device)),
                )
                np.savez_compressed(os.path.join(args.out, f"sims0_q{qid}_c{C}_{dname}.npz"),
                                    sims0=sims0.numpy().astype(np.float16), shared=sh_np, hist_len=hist_len)
                emit(dict(qid=qid, kind="new_full", ctx=C, ctx_tokens=hist_len, delta=dname, gold=gold_new,
                          pred=pred_new, correct=(pred_new == gold_new) if gold_new else None,
                          same_as_old=(pred_new == pred_old), text=text_new, meta=meta, feats=feats,
                          drift_step0_all=drift_all, drift_step0_q=drift_q))
                print(f"  [{dname}] full: pred={pred_new} gold={gold_new} shared={len(sh)}/{P_new} "
                      f"drift0 L4/L8/L32={drift_all[4]:.4f}/{drift_all[8]:.4f}/{drift_all[32]:.4f}", flush=True)
                del h_new0

                # ---------------- reuse sweep, source = step-0 snapshot ------
                for k in ks:
                    rec.start_inject(sh_t, traj_old, k, source="first")
                    o = generate(model, ids_new, recorder=rec, **gen_kw)
                    rec.stop()
                    ids_k = o["x"][0, P_new:].cpu()
                    text_k = decode_gen(tok, ids_k)
                    pred_k = extract_pred(text_k)
                    saved = (k / L) * len(sh) / (P_new + args.gen_length)
                    emit(dict(qid=qid, kind="reuse", ctx=C, ctx_tokens=hist_len, delta=dname, source="first", k=k,
                              gold=gold_new, pred=pred_k, correct=(pred_k == gold_new) if gold_new else None,
                              agree_full=(pred_k == pred_new), same_as_old=(pred_k == pred_old),
                              tok_agree=float((ids_k == full_ids).float().mean()), compute_saved=saved, text=text_k))
                    print(f"    reuse k={k:2d} saved={saved:.2f} pred={pred_k} agree={pred_k == pred_new}", flush=True)

            del h_old0
            if device == "cuda":
                torch.cuda.empty_cache()
            print(f"[q{qid} ctx{C}] done in {time.time()-t_start:.0f}s", flush=True)

    fout.close()
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
