"""
Control for Exp2: inject a request's OWN recorded trajectory into itself.
If the inject mechanism is exact, tok_agree must be 1.0 for every k (up to
bf16 non-determinism, which we also measure by running the same generation twice).
"""
import sys, os, json, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoTokenizer, AutoModel
from instrument import StateRecorder
from sampler import generate
from deltas import build_variants, chat_prompt, gold_answer

steps, gen_length = int(sys.argv[1]) if len(sys.argv) > 1 else 16, int(sys.argv[2]) if len(sys.argv) > 2 else 64
device = "cuda" if torch.cuda.is_available() else "mps"
tok = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
model = AutoModel.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
rec = StateRecorder(model, store_device="cpu")
item = json.load(open(os.path.join(os.path.dirname(__file__), "gsm8k_test200.json")))[0]
q = item["q"]; gold = gold_answer(item["a"])
msgs, _, _ = build_variants(tok, q, gold, "")["irrelevant_append"]
ids = torch.tensor(tok(chat_prompt(tok, msgs), add_special_tokens=False)["input_ids"], device=device).unsqueeze(0)
P = ids.shape[1]
kw = dict(steps=steps, gen_length=gen_length, block_length=32, record_canvas=False)

t = time.time()
rec.start_record(torch.arange(P, device=device)); a = generate(model, ids, recorder=rec, **kw); rec.stop()
traj = rec.traj
print(f"full run A: {time.time()-t:.0f}s, {a['n_forward']} forwards", flush=True)
b = generate(model, ids, **kw)
ga, gb = a["x"][0, P:].cpu(), b["x"][0, P:].cpu()
print(f"determinism (A vs B, no injection): tok_agree={(ga==gb).float().mean():.3f}", flush=True)
for k in (4, 16, 32):
    rec.start_inject(torch.arange(P, device=device), traj, k, source="aligned")
    c = generate(model, ids, recorder=rec, **kw); rec.stop()
    gc = c["x"][0, P:].cpu()
    print(f"self-inject aligned k={k:2d}: tok_agree vs A={(ga==gc).float().mean():.3f}", flush=True)
# max abs difference of the recorded state vs recomputed state at one layer/step (mechanism precision)
rec.start_record(torch.arange(P, device=device)); generate(model, ids, recorder=rec, **kw); rec.stop()
d = max(float((rec.traj[s] .float()- traj[s].float()).abs().max()) for s in traj)
print(f"max |h_recordA - h_recordB| over all steps/layers = {d:.4g}", flush=True)
print("CONTROL_DONE")
