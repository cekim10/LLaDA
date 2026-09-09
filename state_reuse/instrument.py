"""
Instrumentation for cross-invocation state reuse experiments on LLaDA.

Two capabilities, both implemented by wrapping each transformer block's forward:

1. RECORD: for every diffusion step, store the residual-stream hidden state that
   enters block l (l = 0..L-1) and the output of the last block, restricted to a
   set of "shared" token positions (positions whose token ids are identical in the
   old request P and the new request P+D).  Shape per step: [L+1, n_shared, d].

2. INJECT: emulate "reuse layers 0..k-1 of the old request for the shared
   positions".  For block l < k we (a) overwrite the block input at the shared
   positions with the stored h_l of the old request  -> the K/V those positions
   produce are exactly the old K/V (the block's K/V are a deterministic function of
   the block input), and (b) overwrite the block output at the shared positions
   with the stored h_{l+1}  -> the shared positions are effectively *not
   recomputed* at layers < k.  New positions (D tokens, generation canvas) are
   computed normally at every layer and attend to the reused K/V.  From layer k on
   everything is recomputed.  k = 0 is full recomputation, k = L reuses the whole
   stack for the shared positions.

   The stored trajectory to inject from is selected by `source`:
     'aligned' : old step t   -> new step t      (needs whole trajectory stored)
     'final'   : old last step for every new step (needs one snapshot)
     'first'   : old step 0   (all-masked canvas) for every new step
"""
import torch


class StateRecorder:
    def __init__(self, model, store_device="cpu", store_dtype=torch.bfloat16):
        self.model = model
        inner = model.model if hasattr(model, "model") else model
        self.blocks = list(inner.transformer.blocks)
        self.n_layers = len(self.blocks)
        self.store_device = store_device
        self.store_dtype = store_dtype

        self.mode = None            # None | 'record' | 'inject'
        self.step = 0               # set by the generation loop before each forward
        self.shared_idx = None      # LongTensor of positions to record / inject (on model device)
        self.traj = {}              # step -> tensor [L+1, n_shared, d]
        self._cur = None

        # inject config
        self.src = None             # dict step -> tensor [L+1, n_shared, d]
        self.k = 0
        self.source = "aligned"
        self.src_last_step = None

        self._orig = []
        for li, blk in enumerate(self.blocks):
            self._orig.append(blk.forward)
            blk.forward = self._make_wrapper(li, blk.forward)

    # ------------------------------------------------------------------ utils
    def remove(self):
        for blk, f in zip(self.blocks, self._orig):
            blk.forward = f

    def _src_step(self):
        if self.source == "aligned":
            return self.step
        if self.source == "final":
            return self.src_last_step
        if self.source == "first":
            return 0
        raise ValueError(self.source)

    def _src_h(self, layer):
        h = self.src[self._src_step()][layer]
        return h

    # ------------------------------------------------------------------ modes
    def start_record(self, shared_idx):
        self.mode = "record"
        self.shared_idx = shared_idx
        self.traj = {}

    def start_inject(self, shared_idx, src_traj, k, source="aligned"):
        self.mode = "inject"
        self.shared_idx = shared_idx
        self.src = src_traj
        self.k = k
        self.source = source
        self.src_last_step = max(src_traj.keys())

    def stop(self):
        self.mode = None

    # ---------------------------------------------------------------- wrapper
    def _make_wrapper(self, li, orig_forward):
        def fwd(x, *args, **kwargs):
            if self.mode == "record":
                if li == 0:
                    self._cur = torch.empty(
                        (self.n_layers + 1, self.shared_idx.numel(), x.shape[-1]),
                        dtype=self.store_dtype, device=self.store_device)
                self._cur[li] = x[0, self.shared_idx].to(self.store_device, self.store_dtype)
                out, cache = orig_forward(x, *args, **kwargs)
                if li == self.n_layers - 1:
                    self._cur[li + 1] = out[0, self.shared_idx].to(self.store_device, self.store_dtype)
                    self.traj[self.step] = self._cur
                    self._cur = None
                return out, cache

            if self.mode == "inject" and li < self.k:
                x = x.clone()
                x[0, self.shared_idx] = self._src_h(li).to(x.device, x.dtype)
                out, cache = orig_forward(x, *args, **kwargs)
                out = out.clone()
                out[0, self.shared_idx] = self._src_h(li + 1).to(out.device, out.dtype)
                return out, cache

            return orig_forward(x, *args, **kwargs)
        return fwd


# --------------------------------------------------------------------------
# Shared-position bookkeeping
# --------------------------------------------------------------------------
def shared_positions(tok_old, tok_new):
    """Positions (in the *old* sequence coordinate, identical to new coordinate)
    whose token ids coincide.  If the two prompts have equal length, every
    position with the same token counts (covers same-length edits: tokens both
    before and after the edit).  Otherwise only the longest common prefix
    counts (an append, or a length-changing edit)."""
    tok_old = tok_old.tolist() if torch.is_tensor(tok_old) else list(tok_old)
    tok_new = tok_new.tolist() if torch.is_tensor(tok_new) else list(tok_new)
    if len(tok_old) == len(tok_new):
        return [i for i, (a, b) in enumerate(zip(tok_old, tok_new)) if a == b]
    n = 0
    for a, b in zip(tok_old, tok_new):
        if a != b:
            break
        n += 1
    return list(range(n))


def cosine_drift(traj_old, traj_new, steps):
    """Return sims[step, layer, pos] = cos(h_old, h_new) over shared positions."""
    out = []
    for t in steps:
        a = traj_old[t].float()
        b = traj_new[t].float()
        sim = torch.nn.functional.cosine_similarity(a, b, dim=-1)  # [L+1, n_shared]
        out.append(sim)
    return torch.stack(out)  # [T, L+1, n_shared]
