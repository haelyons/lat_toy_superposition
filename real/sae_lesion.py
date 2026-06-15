"""Paradigm-native lesion probe: SAE features as concepts on a FIXED pretrained LM.

This is the in-paradigm version of `lesion.py` — no retraining of the base model, no
three-regime comparison. The capability-modification operation is the one the field
actually uses: ablate a concept *direction* in the residual stream (here, an SAE feature's
decoder direction) and measure the collateral damage to everything else. The question is
whether edit-cleanliness is predicted by the concept's RELIANCE (how much/often the model
uses it) and its INTERFERENCE (decoder-direction overlap with other features) — the two
axes our toy + transformer lesions identified.

For an SAE feature f, ablating its unit decoder direction d_f from the residual at `LAYER`
and measuring per-token cross-entropy change, we split:
  self_effect = mean ΔCE over tokens where f WAS active (the legitimate effect of removing
                a capability that was in use),
  collateral  = mean ΔCE over tokens where f was NOT active (damage to other contexts =
                entanglement; should be ~0 if the feature is cleanly separable).
reliance = firing frequency of f; interference_f = mean_{g≠f} cos(d_f, d_g)^2.

Dry run: distilgpt2 + a small SAE trained on the fly (CPU). GPU rung: set MODEL_NAME to a
real model and (optionally) load a pretrained SAE in `build_sae` — the probe below is
unchanged. Writes results/sae_lesion.csv + results/sae_lesion_summary.md.
"""
import os
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------- config (swap here for the GPU rung) -----------------------
MODEL_NAME = "distilgpt2"
LAYER = 4                 # ablate the residual stream after this block
DICT_MULT = 4             # SAE dictionary = DICT_MULT * d_model
TOPK = 32                 # TopK SAE sparsity
SAE_STEPS = 2500
SAE_BATCH = 4096
N_TRAIN_SEQ = 1200        # sequences for collecting SAE-training activations
N_EVAL_SEQ = 40           # sequences for the lesion eval
SEQ_LEN = 128
N_FEATURES_LESION = 160   # how many alive features to ablate (cost = one fwd each)
CORPUS = "/tmp/corpus.txt"
CORPUS_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/"
              "data/tinyshakespeare/input.txt")


# ----------------------------- corpus -----------------------------
def _get_corpus():
    """Cached tiny-shakespeare for the CPU dry run; self-fetching so a fresh container
    works. GPU rung: replace with a real corpus (wikitext / the SAE's training data)."""
    if not os.path.exists(CORPUS):
        try:
            import urllib.request
            txt = urllib.request.urlopen(CORPUS_URL, timeout=30).read().decode("utf8", "ignore")
            open(CORPUS, "w").write(txt)
        except Exception as e:                          # offline fallback (degenerate; warn)
            print(f"  [warn] corpus fetch failed ({type(e).__name__}); using tiny fallback. "
                  "Provide a real corpus at " + CORPUS + " for meaningful SAE features.")
            return "To be, or not to be, that is the question. " * 2000
    return open(CORPUS).read()


def load_tokens(tok, n_seq, seqlen, seed=0):
    text = _get_corpus()
    ids = tok(text, return_tensors="pt").input_ids[0]
    n = min(n_seq, (len(ids) - 1) // seqlen)
    g = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, len(ids) - seqlen - 1, (n,), generator=g)
    return torch.stack([ids[s:s + seqlen] for s in starts])


# ----------------------------- model activations -----------------------------
@torch.no_grad()
def collect_resid(model, tokens, layer, batch=16):
    """Residual stream after block `layer`, flattened over positions: (N, d)."""
    chunks = []
    for i in range(0, len(tokens), batch):
        out = model(tokens[i:i + batch], output_hidden_states=True)
        chunks.append(out.hidden_states[layer + 1].reshape(-1, out.hidden_states[0].shape[-1]))
    return torch.cat(chunks)


# ----------------------------- SAE -----------------------------
class SAE(nn.Module):
    def __init__(self, d, n_dict, k):
        super().__init__()
        self.k = k
        self.b_dec = nn.Parameter(torch.zeros(d))
        self.W_enc = nn.Parameter(torch.randn(d, n_dict) / d ** 0.5)
        self.b_enc = nn.Parameter(torch.zeros(n_dict))
        self.W_dec = nn.Parameter(torch.randn(n_dict, d) / n_dict ** 0.5)

    def encode(self, x):
        z = (x - self.b_dec) @ self.W_enc + self.b_enc
        topv, topi = z.topk(self.k, dim=-1)
        out = torch.zeros_like(z)
        out.scatter_(-1, topi, F.relu(topv))
        return out

    def decode(self, a):
        return a @ self.W_dec + self.b_dec

    def forward(self, x):
        return self.decode(self.encode(x))

    @torch.no_grad()
    def dec_dirs(self):                      # unit decoder directions (n_dict, d)
        return self.W_dec / self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)


def build_sae(acts, d):
    """Train a small SAE on the activations (dry run). GPU rung: replace with a pretrained
    SAE load, returning an object exposing .encode and .dec_dirs()."""
    sae = SAE(d, DICT_MULT * d, TOPK)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    sae.b_dec.data = acts.mean(0)
    gen = torch.Generator().manual_seed(0)
    for step in range(1, SAE_STEPS + 1):
        idx = torch.randint(0, len(acts), (SAE_BATCH,), generator=gen)
        x = acts[idx]
        opt.zero_grad()
        loss = ((sae(x) - x) ** 2).mean()
        loss.backward()
        with torch.no_grad():                # keep decoder rows ~unit (standard SAE practice)
            sae.W_dec.data /= sae.W_dec.data.norm(dim=1, keepdim=True).clamp_min(1e-8)
        opt.step()
        if step % 500 == 0:
            fvu = (((sae(x) - x) ** 2).mean() / x.var()).item()
            print(f"  [sae] step {step} FVU {fvu:.3f}", flush=True)
    return sae


# ----------------------------- lesion -----------------------------
@torch.no_grad()
def per_token_ce(model, tokens, batch=16, ablate=None):
    """Per-token CE (predict next token). `ablate` = unit direction to project out of the
    residual at LAYER via a forward hook. Returns (B, T-1)."""
    block = model.transformer.h[LAYER]
    handle = None
    if ablate is not None:
        d = ablate
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h - (h @ d).unsqueeze(-1) * d
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        handle = block.register_forward_hook(hook)
    try:
        outs = []
        for i in range(0, len(tokens), batch):
            tb = tokens[i:i + batch]
            logits = model(tb).logits
            ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                 tb[:, 1:].reshape(-1), reduction="none").reshape(tb.shape[0], -1)
            outs.append(ce)
        return torch.cat(outs)
    finally:
        if handle is not None:
            handle.remove()


def run_lesion(model, sae, tokens):
    d_model = model.config.n_embd
    base_ce = per_token_ce(model, tokens)                       # (B, T-1)
    acts = collect_resid(model, tokens, LAYER)                  # (B*T, d)
    feat = sae.encode(acts).reshape(len(tokens), SEQ_LEN, -1)   # (B, T, n_dict)
    feat = feat[:, :-1, :]                                      # align with next-token CE
    dirs = sae.dec_dirs()                                       # (n_dict, d) unit

    freq = (feat > 0).float().mean(dim=(0, 1))                  # firing frequency per feature
    mean_act = feat.sum(dim=(0, 1)) / (feat > 0).float().sum(dim=(0, 1)).clamp_min(1)
    alive = torch.nonzero(freq > 0.005, as_tuple=True)[0]       # features that actually fire
    # interference among ALIVE features (cos^2 of decoder dirs)
    Da = dirs[alive]
    cos2 = (Da @ Da.t()) ** 2
    cos2.fill_diagonal_(0.0)
    interf = cos2.sum(1) / max(1, len(alive) - 1)

    # choose a sample of alive features to ablate (cost = one forward each)
    sel = alive[torch.linspace(0, len(alive) - 1, min(N_FEATURES_LESION, len(alive))).long()]
    sel_set = {int(f): j for j, f in enumerate(alive)}
    rows = []
    for f in sel.tolist():
        dce = per_token_ce(model, tokens, ablate=dirs[f]) - base_ce   # (B, T-1)
        active = feat[:, :, f] > 0
        self_eff = float(dce[active].mean().item()) if active.any() else float("nan")
        collat = float(dce[~active].mean().item()) if (~active).any() else float("nan")
        rows.append({"feature": f, "reliance_freq": float(freq[f].item()),
                     "mean_act": float(mean_act[f].item()),
                     "interference": float(interf[sel_set[f]].item()),
                     "self_effect": self_eff, "collateral": collat,
                     "n_active_tok": int(active.sum().item())})
    return rows


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {MODEL_NAME} (layer {LAYER}) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME); model.eval()
    d = model.config.n_embd

    train_tokens = load_tokens(tok, N_TRAIN_SEQ, SEQ_LEN, seed=0)
    print(f"Collecting activations from {len(train_tokens)} seqs ...", flush=True)
    acts = collect_resid(model, train_tokens, LAYER)
    print(f"Training SAE (dict={DICT_MULT*d}, k={TOPK}) on {len(acts)} activations ...", flush=True)
    sae = build_sae(acts, d)

    eval_tokens = load_tokens(tok, N_EVAL_SEQ, SEQ_LEN, seed=1)
    print(f"Lesioning features on {len(eval_tokens)} eval seqs ...", flush=True)
    rows = run_lesion(model, sae, eval_tokens)

    os.makedirs("results", exist_ok=True)
    with open("results/sae_lesion.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # quick analysis
    import pandas as pd
    df = pd.DataFrame(rows)
    gen = df[df.self_effect > 0.02]                       # genuine knockouts
    def r(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        m = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 4 else float("nan")
    L = ["# SAE-feature lesion on a fixed LM — summary", "",
         f"Model {MODEL_NAME}, layer {LAYER}, SAE dict {DICT_MULT*d}/k{TOPK}. "
         f"{len(df)} features ablated. Paradigm-native: latent ablation on a fixed model, "
         "no retraining. Does edit collateral track reliance and interference?", "",
         f"- r(collateral, reliance_freq) = {r(df.collateral, df.reliance_freq):+.2f}",
         f"- r(collateral, interference) = {r(df.collateral, df.interference):+.2f}",
         f"- r(collateral, reliance) genuine-knockouts = {r(gen.collateral, gen.reliance_freq):+.2f}",
         f"- mean collateral = {df.collateral.mean():.4f}, mean self_effect = {df.self_effect.mean():.4f}",
         f"- collateral / self_effect ratio = {(df.collateral.mean()/max(1e-9,df.self_effect.mean())):.3f} "
         "(low = edits are surgical)", "",
         "Read: a positive reliance/interference correlation = the same axes that governed "
         "edit-cleanliness in the toy + transformer also govern it for real SAE features — "
         "i.e. steering/ablation cleanliness is a measurable property of the representation."]
    open("results/sae_lesion_summary.md", "w").write("\n".join(L))
    print("\n".join(L))
    print(f"\nWrote results/sae_lesion.csv ({len(rows)} features) + results/sae_lesion_summary.md")


if __name__ == "__main__":
    main()
