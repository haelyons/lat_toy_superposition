"""Tiny causal transformer with perturbation hooks for the three robustness regimes.

forward(tokens, emb_delta=None, resid_delta=None, resid_layer=1):
  - emb_delta   : additive perturbation on the input embeddings (token+pos), the INPUT
                  space for a transformer -> used by input-AT.
  - resid_delta : additive perturbation on the residual stream AFTER block `resid_layer-1`
                  (a latent) -> used by LAT.
Both default to None (baseline). Logits are returned for every position; callers read
the '=' position (last token) to predict the answer.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_mlp):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_mlp), nn.GELU(),
                                 nn.Linear(d_mlp, d_model))

    def forward(self, x, attn_mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, vocab, seq_len, d_model=128, n_layers=2, n_heads=4, d_mlp=512, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_mlp) for _ in range(n_layers)])
        self.lnf = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, vocab, bias=False)
        self.seq_len = seq_len
        self.register_buffer("_mask", torch.triu(torch.full((seq_len, seq_len), float("-inf")), 1))

    def embed(self, tokens):
        pos = torch.arange(tokens.shape[1], device=tokens.device)
        return self.tok(tokens) + self.pos(pos)[None]

    def forward(self, tokens, emb_delta=None, resid_delta=None, resid_layer=1):
        x = self.embed(tokens)
        if emb_delta is not None:
            x = x + emb_delta
        mask = self._mask[:tokens.shape[1], :tokens.shape[1]]
        for i, blk in enumerate(self.blocks):
            x = blk(x, mask)
            if resid_delta is not None and i == resid_layer - 1:
                x = x + resid_delta
        return self.unembed(self.lnf(x))

    @torch.no_grad()
    def resid_at(self, tokens, layer):
        """Residual stream after block `layer-1` (the LAT perturbation site)."""
        x = self.embed(tokens)
        mask = self._mask[:tokens.shape[1], :tokens.shape[1]]
        for i, blk in enumerate(self.blocks):
            x = blk(x, mask)
            if i == layer - 1:
                return x
        return x
