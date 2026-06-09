"""Configuration for the LAT toy-model study.

All knobs live here. SPEC defaults only; forks (L-inf ball, targeted objective,
epsilon sweep, uniform importance, two-layer substrate) are NOT enabled.
See SPEC.md sections 4-9. Any deviation from a SPEC default is annotated.
"""
from dataclasses import dataclass, field, asdict


@dataclass
class Config:
    # ---- substrate (SPEC 9 fork) ----
    two_layer: bool = True            # True: perturb-early/measure-late two-layer toy
                                      # (escalated from single bottleneck, whose per-concept
                                      #  robust radius was analytically degenerate ~ tau/||W_i||)

    # ---- model / data (SPEC 4) ----
    m: int = 10                       # latent width (fixed across grid; both layers use m)
    n_over_m: tuple = (2, 4, 8)       # capacity grid n/m (SPEC 7: 2-3 values)
    sparsity: tuple = (0.8, 0.9, 0.99)  # S grid, high-sparsity centred (SPEC 7)
    importance_decay: float = 0.9     # geometric importance ratio (SPEC 4 default)

    # ---- conditions / training (SPEC 5) ----
    seeds: tuple = (0, 1, 2, 3, 4)    # >=5 seeds per cell (SPEC 7)
    steps: int = 5000                 # matched training steps across conditions
    batch: int = 1024
    lr: float = 1e-3

    # ---- adversarial perturbation (SPEC 5) ----
    eps_rel: float = 0.10             # relative epsilon (fraction of ||h|| or ||x||)
    pgd_steps: int = 7                # inner loop 5-10 PGD steps (SPEC 5)
    # pgd step size as fraction of eps; classic 2.5*eps/steps rule -> ~0.36
    pgd_step_frac: float = 0.35

    # ---- SAE proxy (SPEC 6) ----
    sae_dict_mult: int = 4            # dict size = sae_dict_mult * n (overcomplete)
    sae_steps: int = 3000             # fixed protocol across conditions
    sae_lr: float = 1e-3
    sae_batch: int = 1024

    # ---- robust-region probe (SPEC 6) ----
    tau: float = 0.10                 # reconstruction tolerance on a concept's output
    n_random_dirs: int = 8            # held-out directions per concept (set 2)
    radius_bisect_iters: int = 25
    radius_cap_rel: float = 20.0      # right-censor radius at cap * mean||h1||
    represented_w_norm: float = 0.30  # concepts below this ||W1_i|| are unrepresented (excluded)
    probe_samples: int = 512          # active samples used per concept for the probe

    # ---- eval ----
    eval_batch: int = 8192            # batch for metric estimation

    conditions: tuple = ("baseline", "input_at", "lat")

    def to_dict(self):
        return asdict(self)


CFG = Config()
