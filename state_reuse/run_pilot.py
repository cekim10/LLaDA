"""
Pilot for: "Small context updates cause structured/localized dLLM state drift,
rather than invalidating all prior state."

For each GSM8K question (old request P) and each context update D:
  Exp1  full recompute of P and P+D, record residual streams at shared positions
        -> cos-sim drift tensor sims[step, layer, pos]   (+ Exp3 distance, Exp4 step)
  Exp2  re-run P+D injecting P's states for layers 0..k-1 at shared positions
        (k sweep, source in {aligned, final}) -> agreement with full recompute,
        GSM8K accuracy, analytic compute saved
  Exp5  resume P+D's denoising from P's canvas at step t0 (t0 sweep)

Usage:
  python run_pilot.py --n 8 --steps 64 --gen_length 128 --ks 0,4,8,12,16,20,24,28,32 \
                      --sources aligned,final --resume_steps 8,16,32 --out results/pilot
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from instrument import StateRecorder, shared_positions, cosine_drift
from sampler import generate, MASK_ID
from deltas import build_variants, chat_prompt, gold_answer, extract_pred

EOT = 126348
EOS = 126081


def decode_gen(tok, ids):
    ids = [int(i) for i in ids if int(i) not in (EOS, EOT, MASK_ID)]
    return tok.decode(ids, skip_special_tokens=True)


def encode(tok, messages, device):
    s = chat_prompt(tok, messages)
    ids = tok(s, add_special_tokens=False)["input_ids"]
    return torch.tensor(ids, device=device).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "gsm8k_test200.json"))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--gen_length", type=int, default=128)
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--ks", default="0,4,8,12,16,20,24,28,32")
    ap.add_argument("--sources", default="aligned,final")
    ap.add_argument("--resume_steps", default="")
    ap.add_argument("--deltas", default="irrelevant_append,hint_append,number_edit,name_edit,multiturn_append")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results", "pilot"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--store_device", default=None, help="where to keep trajectories (default: model device on cuda, cpu otherwise)")
    ap.add_argument("--skip_exp2", action="store_true")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    store_device = args.store_device or ("cuda" if device == "cuda" else "cpu")
    os.makedirs(args.out, exist_ok=True)
    ks = [int(k) for k in args.ks.split(",") if k != ""]
    sources = [s for s in args.sources.split(",") if s]
    resume_steps = [int(s) for s in args.resume_steps.split(",") if s]
    delta_names = args.deltas.split(",")

    print(f"device={device} store={store_device}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    rec = StateRecorder(model, store_device=store_device)
    L = rec.n_layers

    data = json.load(open(args.data))[args.start:args.start + args.n]
    gen_kw = dict(steps=args.steps, gen_length=args.gen_length, block_length=args.block_length)
    all_steps = list(range(args.steps))

    results_path = os.path.join(args.out, "records.jsonl")
    fout = open(results_path, "a")

    for qi, item in enumerate(data):
        q, gold = item["q"], gold_answer(item["a"])
        t_start = time.time()
        # ---------------- old request P -----------------------------------
        msgs_old = [{"role": "user", "content": q}]
        ids_old = encode(tok, msgs_old, device)
        rec.start_record(torch.arange(ids_old.shape[1], device=device))
        out_old = generate(model, ids_old, recorder=rec, **gen_kw)
        rec.stop()
        traj_old_full = rec.traj  # step -> [L+1, P_old, d]
        text_old = decode_gen(tok, out_old["x"][0, ids_old.shape[1]:])
        pred_old = extract_pred(text_old)
        rec_old = dict(qid=args.start + qi, kind="old", gold=gold, pred=pred_old,
                       correct=(pred_old == gold), text=text_old, P=ids_old.shape[1],
                       n_forward=out_old["n_forward"])
        fout.write(json.dumps(rec_old) + "\n"); fout.flush()
        print(f"[q{args.start+qi}] old: pred={pred_old} gold={gold} ({time.time()-t_start:.0f}s)", flush=True)

        variants = build_variants(tok, q, gold, text_old)
        for dname in delta_names:
            if dname not in variants:
                continue
            msgs_new, gold_new, meta = variants[dname]
            ids_new = encode(tok, msgs_new, device)
            sh = shared_positions(ids_old[0], ids_new[0])
            if len(sh) < 4:
                print(f"  skip {dname}: too few shared positions"); continue
            sh_t = torch.tensor(sh, device=device)
            # trajectory of P restricted to shared positions
            traj_old = {t: h[:, sh, :] for t, h in traj_old_full.items()}

            # ---------------- Exp1: full recompute of P+D, record ----------
            rec.start_record(sh_t)
            out_new = generate(model, ids_new, recorder=rec, **gen_kw)
            rec.stop()
            traj_new = rec.traj
            sims = cosine_drift(traj_old, traj_new, all_steps)  # [T, L+1, n_shared]
            text_new = decode_gen(tok, out_new["x"][0, ids_new.shape[1]:])
            pred_new = extract_pred(text_new)
            # distance of each shared position to the nearest changed position
            changed = sorted(set(range(ids_new.shape[1])) - set(sh))
            first_changed = changed[0] if changed else ids_new.shape[1]
            dist = np.array([min(abs(p - c) for c in changed) if changed else 0 for p in sh])
            np.savez_compressed(os.path.join(args.out, f"sims_q{args.start+qi}_{dname}.npz"),
                                sims=sims.numpy().astype(np.float16), shared=np.array(sh),
                                dist=dist, P_old=ids_old.shape[1], P_new=ids_new.shape[1],
                                first_changed=first_changed,
                                canvas_old=out_old["canvases"].numpy(), canvas_new=out_new["canvases"].numpy())
            rec_new = dict(qid=args.start + qi, kind="new_full", delta=dname, gold=gold_new, pred=pred_new,
                           correct=(pred_new == gold_new) if gold_new else None, text=text_new,
                           P_new=ids_new.shape[1], n_shared=len(sh), first_changed=first_changed,
                           same_as_old=(pred_new == pred_old), meta=meta,
                           sim_by_layer=sims.mean(dim=(0, 2)).tolist())
            fout.write(json.dumps(rec_new) + "\n"); fout.flush()
            print(f"  [{dname}] full: pred={pred_new} gold={gold_new} shared={len(sh)}/{ids_new.shape[1]} "
                  f"sim L0/8/16/24/32={[round(float(sims[:, l].mean()),3) for l in (0,8,16,24,32)]}", flush=True)
            del traj_new

            # ---------------- Exp2: layer-reuse sweep -----------------------
            if not args.skip_exp2:
                full_ids = out_new["x"][0, ids_new.shape[1]:].cpu()
                for source in sources:
                    for k in ks:
                        if k == 0 and source != sources[0]:
                            continue  # k=0 is identical for every source
                        rec.start_inject(sh_t, traj_old, k, source=source)
                        o = generate(model, ids_new, recorder=rec, record_canvas=False, **gen_kw)
                        rec.stop()
                        ids_k = o["x"][0, ids_new.shape[1]:].cpu()
                        text_k = decode_gen(tok, ids_k)
                        pred_k = extract_pred(text_k)
                        tok_agree = float((ids_k == full_ids).float().mean())
                        saved = (k / L) * len(sh) / (ids_new.shape[1] + args.gen_length)
                        r = dict(qid=args.start + qi, kind="reuse", delta=dname, source=source, k=k,
                                 gold=gold_new, pred=pred_k, correct=(pred_k == gold_new) if gold_new else None,
                                 agree_full=(pred_k == pred_new), tok_agree=tok_agree,
                                 compute_saved=saved, text=text_k)
                        fout.write(json.dumps(r) + "\n"); fout.flush()
                        print(f"    reuse src={source:7s} k={k:2d} saved={saved:.2f} pred={pred_k} "
                              f"agree={pred_k == pred_new} tok_agree={tok_agree:.2f}", flush=True)

            # ---------------- Exp5: canvas resume ---------------------------
            if resume_steps and dname != "multiturn_append":
                # P and P+D share the generation region only if the canvas is
                # placed at the new prompt's end; token positions differ, but we
                # test whether P's partially decoded text is a usable start point.
                for t0 in resume_steps:
                    init = out_old["canvases"][t0]  # canvas after t0 steps of P
                    o = generate(model, ids_new, init_gen=init, start_step=t0, record_canvas=False, **gen_kw)
                    ids_r = o["x"][0, ids_new.shape[1]:].cpu()
                    text_r = decode_gen(tok, ids_r); pred_r = extract_pred(text_r)
                    r = dict(qid=args.start + qi, kind="resume", delta=dname, t0=t0, gold=gold_new,
                             pred=pred_r, correct=(pred_r == gold_new) if gold_new else None,
                             agree_full=(pred_r == pred_new),
                             tok_agree=float((ids_r == full_ids).float().mean()) if not args.skip_exp2 else None,
                             n_forward=o["n_forward"], text=text_r)
                    fout.write(json.dumps(r) + "\n"); fout.flush()
                    print(f"    resume t0={t0:2d} fwd={o['n_forward']} pred={pred_r} agree={pred_r == pred_new}", flush=True)

        del traj_old_full
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"[q{args.start+qi}] done in {time.time()-t_start:.0f}s", flush=True)

    fout.close()
    print("wrote", results_path)


if __name__ == "__main__":
    main()
