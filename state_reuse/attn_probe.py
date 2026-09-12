"""
Passive attention probe for LLaDA blocks: for selected layers, recompute the attention
probabilities of canvas (query) positions over all key positions and store the mean mass
per key position (averaged over heads and over canvas rows).  Used for logging only.
"""
import math
import torch


class AttnProbe:
    def __init__(self, model, layers, query_start):
        inner = model.model if hasattr(model, "model") else model
        self.blocks = list(inner.transformer.blocks)
        self.layers = sorted(set(layers))
        self.query_start = query_start   # first canvas position
        self.mass = {}                    # layer -> tensor [T] (mean attention mass per key position)
        self._orig = {}
        for li in self.layers:
            blk = self.blocks[li]
            self._orig[li] = blk.attention
            blk.attention = self._wrap(li, blk)

    def remove(self):
        for li, f in self._orig.items():
            self.blocks[li].attention = f

    def _wrap(self, li, blk):
        orig = self._orig[li]

        def attention(q, k, v, attention_bias=None, layer_past=None, use_cache=False):
            with torch.no_grad():
                B, T, C = q.size()
                nh = blk.config.n_heads
                nkv = blk.config.effective_n_kv_heads
                hs = C // nh
                qq = q.view(B, T, nh, hs).transpose(1, 2)
                kk = k.view(B, T, nkv, hs).transpose(1, 2)
                if blk.config.rope:
                    qq, kk = blk.rotary_emb(qq, kk)
                if nkv != nh:
                    kk = kk.repeat_interleave(nh // nkv, dim=1)
                qs = qq[:, :, self.query_start:, :].float()          # canvas rows
                scores = torch.matmul(qs, kk.float().transpose(-1, -2)) / math.sqrt(hs)  # [B, nh, Tq, T]
                probs = torch.softmax(scores, dim=-1)
                self.mass[li] = probs.mean(dim=(0, 1, 2)).cpu()     # [T]
                del scores, probs, qs
            return orig(q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache)
        return attention
