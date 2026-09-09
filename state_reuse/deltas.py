"""
Context-update (Delta) generators for GSM8K questions.

All deltas keep the *old* tokens at their original positions (appends at the end
of the user message, or same-length in-place edits), so old and new hidden states
can be compared position-wise without RoPE re-alignment.

  irrelevant_append : content-free note appended               (gold unchanged)
  hint_append       : answer hint appended                     (gold unchanged, high relevance)
  number_edit       : first number replaced, same digit count  (gold CHANGES)
  name_edit         : first name replaced by a same-token-count name (gold unchanged)
  multiturn_append  : old assistant answer + follow-up user turn (gold unchanged)
"""
import re

NAMES = ["Maria", "David", "Sarah", "James", "Emily", "Robert", "Linda", "Michael",
         "Anna", "Peter", "Laura", "Kevin", "Tom", "Alice", "Mark", "Julia"]

FOLLOW_UP = "Are you sure? Please double-check your work and state the final answer again."


def gold_answer(a):
    return a.split("####")[-1].strip().replace(",", "")


def extract_pred(text):
    text = text.replace(",", "")
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not nums:
        return None
    v = nums[-1]
    try:
        f = float(v)
        return str(int(f)) if f == int(f) else v
    except ValueError:
        return v


def chat_prompt(tokenizer, messages):
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def _same_len(tokenizer, a, b):
    return len(tokenizer(a, add_special_tokens=False)["input_ids"]) == \
        len(tokenizer(b, add_special_tokens=False)["input_ids"])


def number_edit(q):
    m = re.search(r"\d+", q)
    if m is None:
        return None
    n = m.group(0)
    if len(n) == 1:
        new = str((int(n) % 9) + 1)  # 1..9, different from n unless n==0 handled below
        if new == n:
            new = str((int(n) + 2) % 10)
    else:
        new = str(int(n) + 1) if len(str(int(n) + 1)) == len(n) else str(int(n) - 1)
    return q[:m.start()] + new + q[m.end():], n, new


def name_edit(tokenizer, q):
    m = re.match(r"([A-Z][a-z]+)(?=['’]?s?\b)", q)
    if m is None:
        return None
    old = m.group(1)
    for cand in NAMES:
        if cand == old:
            continue
        new_q = re.sub(r"\b%s\b" % re.escape(old), cand, q)
        if _same_len(tokenizer, q, new_q):
            return new_q, old, cand
    return None


def build_variants(tokenizer, q, gold, old_answer_text):
    """Return dict name -> (messages, gold_or_None, meta).  messages are chat
    messages; the base prompt is [{'role':'user','content':q}]."""
    out = {}
    out["irrelevant_append"] = (
        [{"role": "user", "content": q + "\n\nNote: It is currently 3 PM on a Tuesday."}],
        gold, {})
    out["hint_append"] = (
        [{"role": "user", "content": q + f"\n\nHint: The correct final answer is {gold}."}],
        gold, {})
    ne = number_edit(q)
    if ne is not None:
        out["number_edit"] = ([{"role": "user", "content": ne[0]}], None,
                              {"old": ne[1], "new": ne[2]})
    nm = name_edit(tokenizer, q)
    if nm is not None:
        out["name_edit"] = ([{"role": "user", "content": nm[0]}], gold,
                            {"old": nm[1], "new": nm[2]})
    out["multiturn_append"] = (
        [{"role": "user", "content": q},
         {"role": "assistant", "content": old_answer_text},
         {"role": "user", "content": FOLLOW_UP}],
        gold, {})
    return out
