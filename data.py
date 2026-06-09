"""Synthetic sparse-feature data and the Elhage-style toy model (SPEC 4)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def geometric_importance(n: int, decay: float, device=None) -> torch.Tensor:
    """Importance_i = decay**i (SPEC 4 default). Normalised to mean 1."""
    imp = decay ** torch.arange(n, dtype=torch.float32, device=device)
    return imp * (n / imp.sum())


def sample_batch(batch: int, n: int, sparsity: float, generator: torch.Generator,
                 device=None) -> torch.Tensor:
    """Each feature active with prob p = 1 - S; active value ~ U(0,1) (SPEC 4)."""
    p = 1.0 - sparsity
    active = (torch.rand(batch, n, generator=generator, device=device) < p).float()
    values = torch.rand(batch, n, generator=generator, device=device)
    return active * values


class ToyModel(nn.Module):
    """Linear encoder h = W x ; decoder x' = ReLU(Wt h + b). Tied weights, W is m x n."""

    def __init__(self, n: int, m: int, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        w = torch.randn(m, n, generator=g) / (m ** 0.5)
        self.W = nn.Parameter(w)
        self.b = nn.Parameter(torch.zeros(n))

    def encode(self, x):                 # x: (B, n) -> (B, m)
        return x @ self.W.t()

    def decode(self, h):                 # h: (B, m) -> (B, n)
        return F.relu(h @ self.W + self.b)

    def forward(self, x):
        return self.decode(self.encode(x))


def weighted_mse(x, x_hat, importance):
    """Importance-weighted MSE (SPEC 4): mean over batch of sum_i imp_i (x-x')^2."""
    return (((x - x_hat) ** 2) * importance).sum(dim=1).mean()


class TwoLayerToyModel(nn.Module):
    """SPEC 9 escalation: perturb early, measure late.

    encode : h1 = W1 x                          (early latent; the LAT perturbation site)
    mid    : h2 = ReLU(W2 h1 + c)               (late latent; nonlinear processing)
    decode : x' = ReLU(W1^T h2 + b)             (tied to W1, Elhage-style)

    The extra ReLU + W2 mixing between the perturbation site (h1) and a concept's
    readout make the per-concept critical radius non-degenerate (no longer the
    analytic tau/||W_i|| of the single linear bottleneck). W2 is initialised at
    identity and the decoder is tied to W1, so the model starts close to the proven
    single-layer Elhage model and trains to low reconstruction error.
    `W` exposes W1 so ground-truth superposition is measured in the perturbed space.
    """

    def __init__(self, n: int, m: int, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W1 = nn.Parameter(torch.randn(m, n, generator=g) / (m ** 0.5))
        self.W2 = nn.Parameter(torch.eye(m) + 0.01 * torch.randn(m, m, generator=g))
        self.c = nn.Parameter(torch.zeros(m))
        self.b = nn.Parameter(torch.zeros(n))

    @property
    def W(self):                              # feature geometry in the early latent
        return self.W1

    def encode(self, x):                      # x: (B, n) -> h1: (B, m)
        return x @ self.W1.t()

    def decode(self, h1):                      # h1: (B, m) -> x': (B, n)
        h2 = torch.relu(h1 @ self.W2.t() + self.c)
        return torch.relu(h2 @ self.W1 + self.b)

    def forward(self, x):
        return self.decode(self.encode(x))

