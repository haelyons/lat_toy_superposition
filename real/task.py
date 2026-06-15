"""Multi-operation modular arithmetic — a 'realer' substrate with separable capabilities.

Each example is the 4-token sequence  [a, op, b, =]  and the model predicts the answer
(a single token, the value (a OP b) mod p) at the '=' position. Each OPERATION is a
distinct, cleanly separable capability:
  - innovability: pretrain with one op held out, then introduce it (acquire a new skill).
  - lesion: ablate one op's residual-stream direction and measure collateral on the others.

Vocabulary layout (token ids):
  0 .. p-1            : numbers (also the answer space)
  p .. p+K-1          : operation tokens (one per op in OPS)
  p+K                 : '=' token
Small p and a high train fraction keep this in the fast (non-grokking) regime on CPU.
"""
import torch

# operation name -> function (mod p applied by caller). Six distinct, learnable skills,
# each a separable "capability" to hold out (innovability) or ablate (lesion).
OPS = {
    "add": lambda a, b, p: (a + b) % p,
    "sub": lambda a, b, p: (a - b) % p,
    "mul": lambda a, b, p: (a * b) % p,
    "lin": lambda a, b, p: (2 * a + b) % p,
    "max": lambda a, b, p: torch.maximum(a, b),
    "min": lambda a, b, p: torch.minimum(a, b),
}


class ModArithTask:
    def __init__(self, p=13, ops=("add", "sub", "mul", "lin", "max", "min")):
        self.p = p
        self.ops = list(ops)
        self.K = len(self.ops)
        self.eq_id = p + self.K
        self.vocab = p + self.K + 1
        self.seq_len = 4                      # [a, op, b, =]

    def op_token(self, oi):                   # op index -> token id
        return self.p + oi

    def all_pairs(self):
        a, b = torch.meshgrid(torch.arange(self.p), torch.arange(self.p), indexing="ij")
        return a.reshape(-1), b.reshape(-1)

    def make_split(self, frac=0.9, seed=0, ops_subset=None):
        """Build (x, y, op_idx) tensors for a train/test split, over the chosen ops.
        Returns dict with 'train' and 'test', each (x:[N,4] long, y:[N] long, oi:[N] long)."""
        g = torch.Generator().manual_seed(seed)
        ops_subset = range(self.K) if ops_subset is None else ops_subset
        a, b = self.all_pairs()
        rows_x, rows_y, rows_oi, is_train = [], [], [], []
        for oi in ops_subset:
            fn = OPS[self.ops[oi]]
            y = fn(a, b, self.p).long()
            x = torch.stack([a, self.op_token(oi) * torch.ones_like(a),
                             b, self.eq_id * torch.ones_like(a)], dim=1)
            perm = torch.randperm(len(a), generator=g)
            n_tr = int(frac * len(a))
            tr_mask = torch.zeros(len(a), dtype=torch.bool); tr_mask[perm[:n_tr]] = True
            rows_x.append(x); rows_y.append(y); rows_oi.append(oi * torch.ones_like(a))
            is_train.append(tr_mask)
        X = torch.cat(rows_x); Y = torch.cat(rows_y)
        OI = torch.cat(rows_oi); TR = torch.cat(is_train)
        return {"train": (X[TR], Y[TR], OI[TR]), "test": (X[~TR], Y[~TR], OI[~TR])}


def sample(split, batch, gen):
    X, Y, OI = split
    idx = torch.randint(0, len(X), (batch,), generator=gen)
    return X[idx], Y[idx], OI[idx]
