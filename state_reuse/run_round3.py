"""
Round 3: does step-0 state reuse survive context mutation and token relocation?
See ROUND3_PREREG.md.

For each GSM8K question q and shared history H (12 Q/A turns, ~2048 tokens):
  P  = H + [user q]                       -> old request, record step-0 states
  P' = mutate(H, type, p) + [user q]      -> new request
      types: edit (same-length in-place), insert, delete, reorder; p = relative turn position
  conditions (k=32): full | ours (all mapped tokens, relocated) | prefix (common prefix only)

Usage:
  python run_round3.py --start 50 --n 50 --ctx 2048 --types edit,insert,delete,reorder --pos 10,25,50,75
  python run_round3.py --dry_run   (tokenizer only: prints mutation/alignment statistics)
"""
import argparse, difflib, json, os, re, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deltas import chat_prompt, gold_answer, extract_pred, _same_len, NAMES

EOT = 126348
EOS = 126081
MASK_ID = 126336


def encode_ids(tok, text):
    return tok(text, add_special_tokens=False)["input_ids"]


def clean_solution(a):
    body, _, ans = a.rpartition("####")
    body = re.sub(r"<<[^>]*>>", "", body).strip()
    return body + f"\nThe answer is {ans.strip()}."


def build_history(tok, pool, target_tokens):
    """Fixed multi-turn Q/A history with at least target_tokens tokens. Returns (turns, n_used)
    where turns is a list of (user_msg, assistant_msg) pairs."""
    turns = []
    if target_tokens <= 0:
        return turns, 0
    for i, item in enumerate(pool):
        turns.append(({"role": "user", "content": item["q"]},
                      {"role": "assistant", "content": clean_solution(item["a"])}))
        msgs = [m for t in turns for m in t]
        n = len(encode_ids(tok, tok.apply_chat_template(msgs, tokenize=False)))
        if n >= target_tokens:
            return turns, i + 1
    return turns, len(pool)


# ------------------------------------------------------------------ mutations
def edit_turn_text(tok, text):
    """Same-token-length in-place edit: first number -> different number with the same digit
    count; fallback: first capitalised name -> same-token-count name. Returns new text or None."""
    m = re.search(r"\d+", text)
    if m:
        old = m.group(0)
        for cand in ("7" * len(old), "3" * len(old), "5" * len(old), "9" * len(old)):
            if cand != old and not cand.startswith("0"):
                new = text[:m.start()] + cand + text[m.end():]
                if _same_len(tok, text, new):
                    return new
    m = re.search(r"\b([A-Z][a-z]+)\b", text)
    if m:
        old = m.group(1)
        for nm in NAMES:
            if nm != old:
                new = text[:m.start()] + nm + text[m.end():]
                if _same_len(tok, text, new):
                    return new
    return None


def mutate(tok, turns, mtype, t, spare):
    """Return (new_turns, keys) where keys[i] identifies the origin of new turn i:
    ('old', j) for an unchanged old turn j, ('edit', j) for the edited old turn, ('new', 0)
    for an inserted turn. None if the mutation is not applicable."""
    keys = [("old", j) for j in range(len(turns))]
    if mtype == "edit":
        u, a = turns[t]
        new_q = edit_turn_text(tok, u["content"])
        if new_q is None:
            return None, None
        new_turns = list(turns)
        new_turns[t] = ({"role": "user", "content": new_q}, a)
        keys[t] = ("edit", t)
        return new_turns, keys
    if mtype == "insert":
        new_turns = turns[:t] + [spare] + turns[t:]
        keys = keys[:t] + [("new", 0)] + keys[t:]
        return new_turns, keys
    if mtype == "delete":
        new_turns = turns[:t] + turns[t + 1:]
        keys = keys[:t] + keys[t + 1:]
        return new_turns, keys
    if mtype == "reorder":
        new_turns = turns[:t] + turns[t + 1:] + [turns[t]]
        keys = keys[:t] + keys[t + 1:] + [keys[t]]
        return new_turns, keys
    raise ValueError(mtype)


# ------------------------------------------------------------------ alignment
def align(ids_old, ids_new, min_residual_block=8):
    """Map old -> new positions of identical tokens. Pass 1: monotone longest-block matching
    (difflib) over the whole sequences; this covers unchanged turns, shifted or not. Pass 2:
    match the residual (unmatched) tokens of both sides again, so a block that moved out of
    order (reorder) is also relocated; only residual blocks of >= min_residual_block tokens
    count, to avoid pairing stray template tokens. Returns (pairs, n_pass2)."""
    pairs = []
    sm = difflib.SequenceMatcher(None, ids_old, ids_new, autojunk=False)
    for a, b, n in sm.get_matching_blocks():
        pairs += [(a + j, b + j) for j in range(n)]
    mo = {o for o, _ in pairs}; mn = {n for _, n in pairs}
    ro = [i for i in range(len(ids_old)) if i not in mo]
    rn = [i for i in range(len(ids_new)) if i not in mn]
    n2 = 0
    if ro and rn:
        sm2 = difflib.SequenceMatcher(None, [ids_old[i] for i in ro], [ids_new[i] for i in rn], autojunk=False)
        for a, b, n in sm2.get_matching_blocks():
            if n >= min_residual_block:
                pairs += [(ro[a + j], rn[b + j]) for j in range(n)]
                n2 += n
    pairs.sort(key=lambda t: t[1])
    return pairs, n2


def common_prefix_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def decode_gen(tok, ids):
    ids = [int(i) for i in ids if int(i) not in (EOS, EOT, MASK_ID)]
    return tok.decode(ids, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "gsm8k_test200.json"))
    ap.add_argument("--start", type=int, default=50)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--hist_start", type=int, default=100)
    ap.add_argument("--ctx", default="2048")
    ap.add_argument("--types", default="edit,insert,delete,reorder")
    ap.add_argument("--pos", default="10,25,50,75", help="relative turn positions in percent")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--gen_length", type=int, default=128)
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results", "round3"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--dry_run", action="store_true", help="tokenizer only; print alignment stats")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    data = json.load(open(args.data))
    prompts = data[args.start:args.start + args.n]
    pool = data[args.hist_start:]
    ctxs = [int(c) for c in args.ctx.split(",") if c]
    types = [t for t in args.types.split(",") if t]
    positions = [int(p) for p in args.pos.split(",") if p]

    histories = {}
    for C in ctxs:
        turns, used = build_history(tok, pool, C)
        spare = ({"role": "user", "content": pool[used]["q"]},
                 {"role": "assistant", "content": clean_solution(pool[used]["a"])})
        histories[C] = (turns, spare)
        print(f"ctx target {C}: {len(turns)} turns, spare turn = pool[{used}]", flush=True)

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
        L = rec.n_layers
        gen_kw = dict(steps=args.steps, gen_length=args.gen_length, block_length=args.block_length, record_canvas=False)
        os.makedirs(args.out, exist_ok=True)
        fout = open(os.path.join(args.out, "records.jsonl"), "a")

        def emit(r):
            fout.write(json.dumps(r) + "\n"); fout.flush()

    for qi, item in enumerate(prompts):
        qid = args.start + qi
        q, gold = item["q"], gold_answer(item["a"])
        for C in ctxs:
            turns, spare = histories[C]
            T = len(turns)
            msgs_old = [m for t in turns for m in t] + [{"role": "user", "content": q}]
            keys_old = [("old", j) for j in range(T)] + [("old", T)]  # question turn = key T
            ids_old_l = encode_ids(tok, chat_prompt(tok, msgs_old))
            P_old = len(ids_old_l)
            t_start = time.time()

            if not args.dry_run:
                ids_old = torch.tensor(ids_old_l, device=device).unsqueeze(0)
                rec.start_record(torch.arange(P_old, device=device), steps={0})
                out_old = generate(model, ids_old, recorder=rec, **gen_kw)
                rec.stop()
                h_old0 = rec.traj[0]
                text_old = decode_gen(tok, out_old["x"][0, P_old:])
                pred_old = extract_pred(text_old)
                emit(dict(qid=qid, kind="old", ctx=C, P=P_old, gold=gold, pred=pred_old,
                          correct=(pred_old == gold), text=text_old))
                print(f"[q{qid} ctx{C}] old: P={P_old} pred={pred_old} gold={gold} ({time.time()-t_start:.0f}s)", flush=True)
            else:
                pred_old = None

            for mtype in types:
                for p in positions:
                    t = min(T - 1, max(0, round(p / 100 * (T - 1))))
                    new_turns, keys_hist = mutate(tok, turns, mtype, t, spare)
                    if new_turns is None:
                        print(f"  skip {mtype}@{p}%: not applicable", flush=True); continue
                    msgs_new = [m for tt in new_turns for m in tt] + [{"role": "user", "content": q}]
                    ids_new_l = encode_ids(tok, chat_prompt(tok, msgs_new))
                    P_new = len(ids_new_l)
                    pairs, n_pass2 = align(ids_old_l, ids_new_l)
                    how = f"difflib+{n_pass2}"
                    n_prefix = common_prefix_len(ids_old_l, ids_new_l)
                    old_idx = [o for o, n in pairs]
                    new_idx = [n for o, n in pairs]
                    shifted = [int(n != o) for o, n in pairs]
                    first_changed_new = min([n for n in range(P_new) if n not in set(new_idx)] + [P_new])
                    S_ours = len(pairs) / (P_new + args.gen_length)
                    S_prefix = n_prefix / (P_new + args.gen_length)
                    info = dict(mtype=mtype, p=p, turn=t, P_old=P_old, P_new=P_new, n_mapped=len(pairs),
                                n_prefix=n_prefix, n_shifted=int(sum(shifted)), align=how,
                                S_ours=S_ours, S_prefix=S_prefix)
                    if args.dry_run:
                        print(f"  q{qid} {mtype:7s}@{p:2d}% turn={t:2d} P {P_old}->{P_new} mapped={len(pairs)} "
                              f"prefix={n_prefix} shifted={sum(shifted)} S_ours={S_ours:.2f} S_prefix={S_prefix:.2f} [{how}]", flush=True)
                        continue

                    ids_new = torch.tensor(ids_new_l, device=device).unsqueeze(0)
                    new_t = torch.tensor(new_idx, device=device)
                    traj_ours = {0: h_old0[:, old_idx, :]}
                    pre_t = torch.tensor(list(range(n_prefix)), device=device)
                    traj_pre = {0: h_old0[:, :n_prefix, :]}

                    # ---- full recompute (reference), record step-0 at mapped positions
                    rec.start_record(new_t, steps={0})
                    out_full = generate(model, ids_new, recorder=rec, **gen_kw)
                    rec.stop()
                    h_new0 = rec.traj[0]
                    sims0 = torch.nn.functional.cosine_similarity(traj_ours[0].float(), h_new0.float(), dim=-1).cpu()  # [L+1, n_mapped]
                    sh_mask = torch.tensor(shifted, dtype=torch.bool)
                    pre_mask = torch.tensor([n < first_changed_new for n in new_idx])
                    post_mask = ~pre_mask
                    drift = dict(
                        pre=(1 - sims0[:, pre_mask].mean(1)).tolist() if pre_mask.any() else None,
                        post=(1 - sims0[:, post_mask].mean(1)).tolist() if post_mask.any() else None,
                        shifted=(1 - sims0[:, sh_mask].mean(1)).tolist() if sh_mask.any() else None,
                    )
                    np.savez_compressed(os.path.join(args.out, f"sims0_q{qid}_c{C}_{mtype}_p{p}.npz"),
                                        sims0=sims0.numpy().astype(np.float16), old_idx=np.array(old_idx),
                                        new_idx=np.array(new_idx), n_prefix=n_prefix, first_changed_new=first_changed_new)
                    full_ids = out_full["x"][0, P_new:].cpu()
                    text_full = decode_gen(tok, full_ids); pred_full = extract_pred(text_full)
                    emit(dict(qid=qid, kind="full", ctx=C, **info, gold=gold, pred=pred_full,
                              correct=(pred_full == gold), same_as_old=(pred_full == pred_old), text=text_full, drift0=drift))
                    print(f"  [{mtype}@{p}%] full: pred={pred_full} mapped={len(pairs)}/{P_new} prefix={n_prefix} "
                          f"S_ours={S_ours:.2f} S_prefix={S_prefix:.2f} drift0 post L8/L32="
                          f"{(drift['post'][8] if drift['post'] else float('nan')):.4f}/{(drift['post'][32] if drift['post'] else float('nan')):.4f}", flush=True)
                    del h_new0

                    # ---- ours: all mapped tokens (relocated), k=32
                    for cond, idx_t, traj, S in (("ours", new_t, traj_ours, S_ours), ("prefix", pre_t, traj_pre, S_prefix)):
                        if idx_t.numel() == 0:
                            continue
                        rec.start_inject(idx_t, traj, args.k, source="first")
                        o = generate(model, ids_new, recorder=rec, **gen_kw)
                        rec.stop()
                        ids_c = o["x"][0, P_new:].cpu()
                        text_c = decode_gen(tok, ids_c); pred_c = extract_pred(text_c)
                        emit(dict(qid=qid, kind=cond, ctx=C, **info, k=args.k, gold=gold, pred=pred_c,
                                  correct=(pred_c == gold), agree_full=(pred_c == pred_full), same_as_old=(pred_c == pred_old),
                                  tok_agree=float((ids_c == full_ids).float().mean()), compute_saved=S, text=text_c))
                        print(f"    {cond:6s} k={args.k} saved={S:.2f} pred={pred_c} agree={pred_c == pred_full}", flush=True)

            if not args.dry_run:
                del h_old0
                if device == "cuda":
                    torch.cuda.empty_cache()
                print(f"[q{qid} ctx{C}] done in {time.time()-t_start:.0f}s", flush=True)

    if not args.dry_run:
        fout.close()
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
