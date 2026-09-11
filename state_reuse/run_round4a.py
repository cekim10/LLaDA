"""
Round 4a: dependency-changing mutations (ROUND4A_PREREG.md).

  history (12 turns) with the problem statement placed as a "remember this" turn at position p
  final user turn asks to solve the remembered problem
  conditions: ctrl_other | ctrl_name | dep_number | dep_move   x   {full, ours(k=32), prefix(k=32)}

Usage:
  python run_round4a.py --start 50 --n 50 --pos 10,50 --out results/round4a
  python run_round4a.py --dry_run      (tokenizer only: eligibility, alignment stats)
"""
import argparse, json, os, re, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deltas import chat_prompt, gold_answer, extract_pred, number_edit, name_edit
from run_round3 import (encode_ids, clean_solution, build_history, edit_turn_text, align,
                        common_prefix_len, decode_gen)

REMEMBER = "Here is a problem I want to solve later. Please just remember it for now:\n\n{q}"
ACK = "Understood. I will remember the problem. Ask me when you are ready to solve it."
FINAL = ("Now solve the problem I asked you to remember earlier{hint}. "
         "Show your work and give the final answer.")


def final_turn(q):
    """Topic hint = first words of q that contain no digit (so an edited number never leaks)."""
    words = q.split()
    for n in (6, 5, 4, 3):
        head = " ".join(words[:n])
        if not re.search(r"\d", head):
            return FINAL.format(hint=f' (the one that begins with "{head}...")')
    return FINAL.format(hint="")


# ------------------------------------------------------------------ gold re-evaluation
def _num(s):
    s = s.replace(",", "").replace("$", "")
    try:
        f = float(s)
    except ValueError:
        return None
    return f


def _fmt(f):
    return str(int(round(f))) if abs(f - round(f)) < 1e-9 else str(round(f, 4))


def edited_gold(q, a, old_num, new_num):
    """Re-evaluate the GSM8K calculator chain with old_num -> new_num substituted and
    intermediate results propagated. Returns (new_gold_str or None, reason)."""
    exprs = re.findall(r"<<([^=<>]+)=([^<>]+)>>", a)
    if not exprs:
        return None, "no_annotations"
    if len(re.findall(r"(?<![\d.])%s(?![\d.])" % re.escape(old_num), q)) != 1:
        return None, "num_not_unique_in_q"
    gold = gold_answer(a)
    old_f, new_f = float(old_num), float(new_num)
    orig_results, new_results = [], []
    for lhs, rhs in exprs:
        lhs = lhs.replace(",", "").replace("$", "").strip()
        if not re.fullmatch(r"[\d.\s+\-*/()]+", lhs):
            return None, "bad_lhs"
        try:
            v = eval(lhs, {"__builtins__": {}})
        except Exception:
            return None, "eval_fail"
        rv = _num(rhs)
        if rv is None or abs(v - rv) > 1e-6 * max(1, abs(rv)):
            return None, "chain_mismatch"
        # substitute operands
        def sub(m):
            t = float(m.group(0))
            hits_old = abs(t - old_f) < 1e-9
            hits_prev = [i for i, r in enumerate(orig_results) if abs(t - r) < 1e-9]
            if hits_old and hits_prev:
                raise ValueError("ambiguous")
            if hits_old:
                return repr(new_f)
            if hits_prev:
                return repr(new_results[hits_prev[-1]])
            return m.group(0)
        try:
            new_lhs = re.sub(r"\d+(?:\.\d+)?", sub, lhs)
            nv = eval(new_lhs, {"__builtins__": {}})
        except ValueError:
            return None, "ambiguous_operand"
        except Exception:
            return None, "eval_fail_new"
        orig_results.append(v); new_results.append(nv)
    if _fmt(orig_results[-1]) != gold.replace(",", ""):
        return None, "final_mismatch"
    if abs(new_results[-1] - orig_results[-1]) < 1e-9:
        return None, "answer_unchanged"
    if new_results[-1] < 0:
        return None, "negative_answer"
    return _fmt(new_results[-1]), "ok"


def choose_dep_edit(q, a):
    """Pick a same-digit-count replacement for the first number in q such that the re-evaluated
    gold is a changed, non-negative integer. Returns (new_q, old, new, gold_new, reason) or None."""
    m = re.search(r"\d+", q)
    if m is None:
        return None
    old = m.group(0); n = int(old); L_ = len(old)
    cands = [n + d for d in (1, -1, 2, -2, 3, -3, 5, -5, 10, -10, 4, -4, 6, 7, 8, 9, 20, 50, 100)]
    cands += [n * 2, n * 3, n // 2, n // 3]
    seen, last_reason = set(), "no_candidate"
    for c in cands:
        if c <= 0 or c == n or len(str(c)) != L_ or c in seen:
            continue
        seen.add(c)
        g, why = edited_gold(q, a, old, str(c))
        last_reason = why
        if g is not None and re.fullmatch(r"\d+", g):
            return q[:m.start()] + str(c) + q[m.end():], old, str(c), g, "ok"
    # no integer-gold candidate: fall back to N+1 without gold
    c = n + 1 if len(str(n + 1)) == L_ else n - 1
    return q[:m.start()] + str(c) + q[m.end():], old, str(c), None, last_reason


STOP = {"How", "What", "If", "The", "She", "He", "They", "It", "In", "On", "At", "For", "After", "Each",
        "There", "This", "That", "When", "A", "An", "To", "Of", "And", "But", "So", "Then", "Every", "One",
        "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Half", "Monday", "Tuesday",
        "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "January", "February", "March", "April",
        "May", "June", "July", "August", "September", "October", "November", "December"}


def name_edit_anywhere(tok, q):
    """Replace one capitalised, non-sentence-initial, non-stopword token (a likely name) by a
    NAMES entry with the same token count. Returns (new_q, old, new) or None."""
    from deltas import NAMES, _same_len
    for m in re.finditer(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]{2,})\b", q):
        old = m.group(1)
        if old in STOP:
            continue
        for cand in NAMES:
            if cand == old:
                continue
            new_q = re.sub(r"\b%s\b" % re.escape(old), cand, q)
            if _same_len(tok, q, new_q):
                return new_q, old, cand
    return None


def find_span(ids, sub_ids):
    """Approximate span of sub_ids inside ids via the longest matching block."""
    import difflib
    sm = difflib.SequenceMatcher(None, ids, sub_ids, autojunk=False)
    a, b, n = sm.find_longest_match(0, len(ids), 0, len(sub_ids))
    return (a, a + n) if n > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "gsm8k_test200.json"))
    ap.add_argument("--start", type=int, default=50)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--hist_start", type=int, default=100)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--pos", default="10,50")
    ap.add_argument("--conds", default="ctrl_other,ctrl_name,dep_number,dep_move")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--gen_length", type=int, default=128)
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results", "round4a"))
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
    turns, used = build_history(tok, pool, args.ctx)
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

    elig = {"ok": 0}
    for qi, item in enumerate(prompts):
        qid = args.start + qi
        q, gold = item["q"], gold_answer(item["a"])
        dep_turn = ({"role": "user", "content": REMEMBER.format(q=q)}, {"role": "assistant", "content": ACK})
        de = choose_dep_edit(q, item["a"])
        if de is None:
            ne, gold_dep, why = None, None, "no_number"
        else:
            ne, gold_dep, why = (de[0], de[1], de[2]), de[3], de[4]
        elig[why] = elig.get(why, 0) + 1
        nm = name_edit(tok, q) or name_edit_anywhere(tok, q)

        for p in positions:
            t = min(T, max(0, round(p / 100 * T)))  # dependency turn inserted before old turn t
            hist_old = turns[:t] + [dep_turn] + turns[t:]
            msgs_old = [m for tt in hist_old for m in tt] + [{"role": "user", "content": final_turn(q)}]
            ids_old_l = encode_ids(tok, chat_prompt(tok, msgs_old))
            P_old = len(ids_old_l)
            dep_span_old = find_span(ids_old_l, encode_ids(tok, q))
            t_start = time.time()

            if not args.dry_run:
                ids_old = torch.tensor(ids_old_l, device=device).unsqueeze(0)
                rec.start_record(torch.arange(P_old, device=device), steps={0})
                out_old = generate(model, ids_old, recorder=rec, **gen_kw)
                rec.stop()
                h_old0 = rec.traj[0]
                text_old = decode_gen(tok, out_old["x"][0, P_old:]); pred_old = extract_pred(text_old)
                emit(dict(qid=qid, kind="old", p=p, P=P_old, gold=gold, pred=pred_old, correct=(pred_old == gold), text=text_old))
                print(f"[q{qid} p{p}] old: P={P_old} pred={pred_old} gold={gold} ({time.time()-t_start:.0f}s)", flush=True)
            else:
                pred_old = None

            for cond in conds:
                gold_new = gold
                if cond == "ctrl_other":
                    j = t + 1 if t + 1 < len(hist_old) else t - 1  # adjacent unrelated turn (index in hist_old)
                    u, a_ = hist_old[j]
                    new_q = edit_turn_text(tok, u["content"])
                    if new_q is None:
                        print(f"  skip {cond}: no editable token", flush=True); continue
                    hist_new = list(hist_old); hist_new[j] = ({"role": "user", "content": new_q}, a_)
                    meta = dict(turn=j)
                elif cond == "ctrl_name":
                    if nm is None:
                        print(f"  skip {cond}: no same-length name edit", flush=True); continue
                    hist_new = list(hist_old)
                    hist_new[t] = ({"role": "user", "content": REMEMBER.format(q=nm[0])}, dep_turn[1])
                    meta = dict(old=nm[1], new=nm[2])
                elif cond == "dep_number":
                    if ne is None:
                        print(f"  skip {cond}: no number", flush=True); continue
                    hist_new = list(hist_old)
                    hist_new[t] = ({"role": "user", "content": REMEMBER.format(q=ne[0])}, dep_turn[1])
                    gold_new = gold_dep  # None if ineligible
                    meta = dict(old=ne[1], new=ne[2], gold_reason=why)
                elif cond == "dep_move":
                    hist_new = hist_old[:t] + hist_old[t + 1:] + [dep_turn]
                    meta = dict(moved_to=len(hist_new) - 1)
                else:
                    raise ValueError(cond)

                msgs_new = [m for tt in hist_new for m in tt] + [{"role": "user", "content": final_turn(q)}]
                ids_new_l = encode_ids(tok, chat_prompt(tok, msgs_new))
                P_new = len(ids_new_l)
                pairs, n2 = align(ids_old_l, ids_new_l)
                n_prefix = common_prefix_len(ids_old_l, ids_new_l)
                old_idx = [o for o, n in pairs]; new_idx = [n for o, n in pairs]
                S_ours = len(pairs) / (P_new + args.gen_length); S_prefix = n_prefix / (P_new + args.gen_length)
                dep_mask = [dep_span_old is not None and dep_span_old[0] <= o < dep_span_old[1] for o in old_idx]
                info = dict(cond=cond, p=p, dep_turn=t, P_old=P_old, P_new=P_new, n_mapped=len(pairs), n_prefix=n_prefix,
                            n_shifted=int(sum(o != n for o, n in pairs)), n_dep_mapped=int(sum(dep_mask)),
                            S_ours=S_ours, S_prefix=S_prefix, meta=meta)
                if args.dry_run:
                    print(f"  q{qid} p{p:2d} {cond:10s} P {P_old}->{P_new} mapped={len(pairs)} prefix={n_prefix} "
                          f"dep_mapped={sum(dep_mask)} S_ours={S_ours:.2f} S_prefix={S_prefix:.2f} gold_new={gold_new} [{why if cond=='dep_number' else ''}]", flush=True)
                    continue

                ids_new = torch.tensor(ids_new_l, device=device).unsqueeze(0)
                new_t = torch.tensor(new_idx, device=device)
                traj_ours = {0: h_old0[:, old_idx, :]}
                pre_t = torch.tensor(list(range(n_prefix)), device=device)
                traj_pre = {0: h_old0[:, :n_prefix, :]}

                rec.start_record(new_t, steps={0})
                out_full = generate(model, ids_new, recorder=rec, **gen_kw)
                rec.stop()
                h_new0 = rec.traj[0]
                sims0 = torch.nn.functional.cosine_similarity(traj_ours[0].float(), h_new0.float(), dim=-1).cpu()
                dm = torch.tensor(dep_mask, dtype=torch.bool)
                drift = dict(all=(1 - sims0.mean(1)).tolist(),
                             dep=(1 - sims0[:, dm].mean(1)).tolist() if dm.any() else None,
                             nondep=(1 - sims0[:, ~dm].mean(1)).tolist() if (~dm).any() else None)
                np.savez_compressed(os.path.join(args.out, f"sims0_q{qid}_p{p}_{cond}.npz"),
                                    sims0=sims0.numpy().astype(np.float16), old_idx=np.array(old_idx),
                                    new_idx=np.array(new_idx), dep_mask=np.array(dep_mask), n_prefix=n_prefix)
                full_ids = out_full["x"][0, P_new:].cpu()
                text_full = decode_gen(tok, full_ids); pred_full = extract_pred(text_full)
                emit(dict(qid=qid, kind="full", **info, gold=gold_new, pred=pred_full,
                          correct=(pred_full == gold_new) if gold_new else None, same_as_old=(pred_full == pred_old),
                          text=text_full, drift0=drift))
                print(f"  [{cond}] full: pred={pred_full} gold={gold_new} old={pred_old} mapped={len(pairs)}/{P_new} "
                      f"S_ours={S_ours:.2f} S_prefix={S_prefix:.2f} drift0 dep L32={(drift['dep'][32] if drift['dep'] else float('nan')):.4f}", flush=True)
                del h_new0

                for cname, idx_t, traj, S in (("ours", new_t, traj_ours, S_ours), ("prefix", pre_t, traj_pre, S_prefix)):
                    if idx_t.numel() == 0:
                        continue
                    rec.start_inject(idx_t, traj, args.k, source="first")
                    o = generate(model, ids_new, recorder=rec, **gen_kw)
                    rec.stop()
                    ids_c = o["x"][0, P_new:].cpu()
                    text_c = decode_gen(tok, ids_c); pred_c = extract_pred(text_c)
                    emit(dict(qid=qid, kind=cname, **info, k=args.k, gold=gold_new, pred=pred_c,
                              correct=(pred_c == gold_new) if gold_new else None, agree_full=(pred_c == pred_full),
                              same_as_old=(pred_c == pred_old), stale=(pred_c == pred_old and pred_full != pred_old),
                              tok_agree=float((ids_c == full_ids).float().mean()), compute_saved=S, text=text_c))
                    print(f"    {cname:6s} saved={S:.2f} pred={pred_c} agree={pred_c == pred_full} stale={(pred_c == pred_old and pred_full != pred_old)}", flush=True)

            if not args.dry_run:
                del h_old0
                if device == "cuda":
                    torch.cuda.empty_cache()
                print(f"[q{qid} p{p}] done in {time.time()-t_start:.0f}s", flush=True)

    print("gold eligibility (dep_number):", elig, flush=True)
    if not args.dry_run:
        fout.close()
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
