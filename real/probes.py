"""The two capability-modification probes, ported to the tiny transformer.

LESION  (remove an existing capability): each operation is a concept. We take its
  direction in the FINAL residual stream (diff-of-means at the '=' position), ablate it
  (project it out for all inputs), and measure collateral = accuracy drop on the OTHER
  operations vs self_drop = accuracy drop on the ablated one. Per-op interference
  I_i = mean_{j!=i} cos(g_i, g_j)^2 is the transformer analogue of the toy's interference.

INNOVABILITY (acquire a new capability): pretrain with one operation held out, fine-tune
  it in under the condition's objective, measure acquisition (new-op accuracy) and
  forgetting (old-op accuracy drop). This is the multi-skill analogue of the toy probe.
"""
import copy
import torch
import torch.nn.functional as F

from real.task import sample, ModArithTask
from real.train import accuracy, finetune, _ce


# --------------------------------------------------------------------------- lesion
# A capability = an operation, represented at its op-TOKEN EMBEDDING (the input
# representation of the skill — the transformer analogue of the toy's input-feature
# directions). Knockout = project the op's embedding direction out at the op position
# for ALL inputs; collateral on op j is mediated by the overlap of their embedding
# directions, exactly the per-op interference I_i. (Deeper residual sites are causally
# inert here — by then the answer is computed — so the embedding is the right site.)

OP_POS = 1                                       # position of the op token in [a,op,b,=]


@torch.no_grad()
def op_directions(model, split, task):
    """Unit diff-of-means op-embedding directions (at OP_POS) + per-op interference
    I_i = mean_{j!=i} cos(g_i, g_j)^2."""
    X, _, OI = split
    E = model.embed(X)[:, OP_POS, :]             # (N, d) embedding at the op position
    grand = E.mean(0)
    dirs = [((E[OI == oi].mean(0) - grand)) for oi in range(task.K)]
    G = torch.stack([g / g.norm().clamp_min(1e-12) for g in dirs])
    cos2 = (G @ G.t()) ** 2
    cos2.fill_diagonal_(0.0)
    return G, cos2.sum(1) / (task.K - 1)


@torch.no_grad()
def lesion(model, split, task):
    """Knock out each op's embedding direction; per-op self_drop / collateral / I_i."""
    X, Y, OI = split
    G, I = op_directions(model, split, task)
    E = model.embed(X)                           # (N, T, d)

    def acc_per_op(delta):
        pred = model(X, emb_delta=delta)[:, -1, :].argmax(-1)
        ok = (pred == Y)
        return torch.tensor([ok[OI == oi].float().mean() if (OI == oi).any() else float("nan")
                             for oi in range(task.K)])

    base_acc = acc_per_op(None)
    rows = []
    for i in range(task.K):
        gi = G[i]
        coeff = E[:, OP_POS, :] @ gi                       # (N,)
        delta = torch.zeros_like(E)
        delta[:, OP_POS, :] = -coeff[:, None] * gi[None, :]  # project gi out at op position
        drop = base_acc - acc_per_op(delta)
        mask = torch.ones(task.K, dtype=torch.bool); mask[i] = False
        rows.append({"op": task.ops[i], "op_idx": i, "I_i": float(I[i].item()),
                     "self_drop": float(drop[i].item()),
                     "collateral": float(drop[mask].mean().item()),
                     "base_acc": float(base_acc[i].item())})
    return rows


# ----------------------------------------------------------------------- innovability

def adapt_new_op(base, task, cfg, holdout_oi, seed, steps=600, log_every=60):
    """Adapt op `holdout_oi` into a condition-specific base with IDENTICAL clean SGD on
    the NEW op ONLY (no rehearsal of old ops — the canonical continual-learning stress
    test; the over-parameterised transformer would not forget under rehearsal). Only the
    pretrained basin differs across conditions. Returns trajectory (step, new, old)."""
    full = task.make_split(frac=1.0, seed=seed)["train"]
    old_split = _subset(full, [o for o in range(task.K) if o != holdout_oi])
    new_split = _subset(full, [holdout_oi])
    gen = torch.Generator().manual_seed(9000 + seed * 10 + holdout_oi)

    model = copy.deepcopy(base)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.finetune_lr)
    traj = []
    for step in range(0, steps + 1, log_every):
        if step > 0:
            for _ in range(log_every):
                x, y, _ = sample(new_split, cfg.batch, gen)     # NEW op only
                opt.zero_grad()
                loss = _ce(model(x), y)                          # clean SGD, identical
                loss.backward(); opt.step()
        traj.append((step, accuracy(model, new_split, task),
                     accuracy(model, old_split, task)))
    return traj


def _subset(split, ops_keep):
    X, Y, OI = split
    m = torch.zeros(len(X), dtype=torch.bool)
    for o in ops_keep:
        m |= (OI == o)
    return X[m], Y[m], OI[m]
