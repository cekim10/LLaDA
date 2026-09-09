"""
LLaDA sampler (low-confidence remasking, semi-autoregressive blocks) that is a
faithful copy of generate.py's logic but (1) reports the global diffusion step to
a recorder before every forward pass, (2) records the canvas and per-position
confidence at every step, and (3) can start from a partially-decoded canvas
(for trajectory-resume experiments) while keeping the same total step budget.
"""
import numpy as np
import torch
import torch.nn.functional as F

MASK_ID = 126336


@torch.no_grad()
def generate(model, prompt, steps=64, gen_length=128, block_length=32,
             mask_id=MASK_ID, recorder=None, init_gen=None, start_step=0,
             record_canvas=True):
    """
    prompt   : LongTensor [1, P]
    init_gen : optional LongTensor [gen_length] initial canvas (partially decoded)
    start_step: global step index at which the schedule resumes (used when
               init_gen comes from an old trajectory at that step); blocks whose
               masks are already gone are skipped (no forward passes spent).
    Returns dict(x=final ids [1, P+gen], canvases=[T+1, gen] ids per step,
                 conf=[T, gen] confidence of the argmax at each step,
                 n_forward=int number of forward passes actually executed)
    """
    device = model.device
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt.shape[1]] = prompt
    if init_gen is not None:
        x[0, prompt.shape[1]:] = init_gen.to(device)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    spb = steps // num_blocks  # steps per block

    P = prompt.shape[1]
    canvases = [x[0, P:].clone().cpu()]
    confs = []
    n_forward = 0
    global_step = 0

    for nb in range(num_blocks):
        bs, be = P + nb * block_length, P + (nb + 1) * block_length
        block_steps = list(range(nb * spb, (nb + 1) * spb))
        # steps of this block that still have to be executed
        todo = [s for s in block_steps if s >= start_step]
        if not todo:
            # block was already handled in the old trajectory
            continue
        n_masked = int((x[0, bs:be] == mask_id).sum())
        base, rem = divmod(n_masked, len(todo))
        transfers = [base + (1 if i < rem else 0) for i in range(len(todo))]

        for s, ntr in zip(todo, transfers):
            if ntr == 0 and (x[0, bs:be] == mask_id).sum() == 0:
                continue
            if recorder is not None:
                recorder.step = s
            mask_index = (x == mask_id)
            logits = model(x).logits
            n_forward += 1
            x0 = torch.argmax(logits, dim=-1)
            p = F.softmax(logits.float(), dim=-1)
            x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
            x0_p[:, be:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -np.inf))
            if record_canvas:
                confs.append(x0_p[0, P:].clone().cpu())
            if ntr > 0:
                _, sel = torch.topk(confidence[0], k=ntr)
                x[0, sel] = x0[0, sel]
            if record_canvas:
                canvases.append(x[0, P:].clone().cpu())
            global_step = s

    return dict(x=x, canvases=torch.stack(canvases) if record_canvas else None,
                conf=torch.stack(confs) if (record_canvas and confs) else None,
                n_forward=n_forward)
