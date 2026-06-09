"""Metric definitions (SPEC 6).

Ground-truth superposition (from W), per-concept proxies (computed on the
bottleneck as the parent project does), and the robust-region probe.
All per-concept quantities are returned as length-n arrays so they can be
correlated against ground truth (Claim C) and compared across conditions (Claim B).
"""
import torch


# ----------------------------- ground truth (from W) -----------------------------

@torch.no_grad()
def ground_truth_dimensionality(W):
    """D_i = ||W_i||^2 / sum_j (What_i . W_j)^2  (SPEC 6). W is (m, n); W_i = column i."""
    norms = W.norm(dim=0)                          # (n,)
    What = W / norms.clamp_min(1e-12)              # unit columns
    proj = What.t() @ W                            # (n, n): (What_i . W_j)
    denom = (proj ** 2).sum(dim=1).clamp_min(1e-12)
    return (norms ** 2 / denom)                    # (n,)  in (0, 1]


@torch.no_grad()
def offdiag_interference(W):
    """Mean squared off-diagonal of the normalised Gram matrix (SPEC 6)."""
    Wn = W / W.norm(dim=0, keepdim=True).clamp_min(1e-12)
    G = Wn.t() @ Wn
    n = G.shape[0]
    off = G ** 2
    off.fill_diagonal_(0.0)
    return (off.sum() / (n * (n - 1))).item()


# ----------------------------- concept proxies -----------------------------

@torch.no_grad()
def concept_proxies(model, x):
    """Per-concept diff-of-means, top-1 SVD explained variance, and participation
    ratio of the activation-difference cloud (SPEC 6).

    Cloud_i = { h_s - mean(h | feature i inactive) } over samples where i is active.
    Spread comes from co-active features (interference), so the cloud is non-trivial.
    Returns dict of length-n arrays."""
    h = model.encode(x)                            # (B, m)
    n = x.shape[1]
    active = x > 0                                 # (B, n)

    diff_of_means = torch.zeros(n, h.shape[1])
    top1_evr = torch.zeros(n)
    pr = torch.zeros(n)

    for i in range(n):
        mask = active[:, i]
        if mask.sum() < 4 or (~mask).sum() < 4:
            top1_evr[i] = float("nan")
            pr[i] = float("nan")
            continue
        mu_inactive = h[~mask].mean(dim=0)
        mu_active = h[mask].mean(dim=0)
        diff_of_means[i] = mu_active - mu_inactive
        cloud = h[mask] - mu_inactive              # (n_active, m)
        # singular values of the cloud
        s = torch.linalg.svdvals(cloud)
        lam = s ** 2
        total = lam.sum().clamp_min(1e-12)
        top1_evr[i] = (lam[0] / total).item()
        pr[i] = ((lam.sum() ** 2) / (lam ** 2).sum().clamp_min(1e-12)).item()

    return {"diff_of_means": diff_of_means, "top1_evr": top1_evr, "pr": pr}


# ----------------------------- robust region (Claim B core) -----------------------------

def _recon_feature_i(model, h, i):
    """x'_i for a batch of latents h (keeps grad if h requires it)."""
    return model.decode(h)[:, i]


def critical_radius_along(model, h0, base_val, i, unit_dirs, tau, iters, cap):
    """Largest scalar rho (<= cap) such that mean_s |x'_i(h0 + rho*u_s) - base_val_s| <= tau,
    where u_s (rows of unit_dirs, per-sample unit vectors) define the direction.
    Binary search on rho. Returns (radius, saturated) where saturated means the
    concept tolerated the whole cap (radius is right-censored, not informative)."""
    @torch.no_grad()
    def deviation(rho):
        pert = model.decode(h0 + rho * unit_dirs)[:, i]
        return (pert - base_val).abs().mean().item()

    if deviation(cap) <= tau:
        return cap, True                       # right-censored at the cap
    lo, hi = 0.0, cap
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if deviation(mid) <= tau:
            lo = mid
        else:
            hi = mid
    return lo, False


@torch.no_grad()
def _random_unit_dirs(B, m, n_dirs, generator):
    """n_dirs random per-sample unit vectors (held-out / unseen, set 2).

    Plain random (not orthogonalised against set 1): orthogonalising against the
    worst-case direction makes random dirs barely move the concept readout, which
    right-censored the radius in the single-layer probe. Random directions are
    'unseen' in the sense that the adversary practised the gradient direction, not
    these."""
    dirs = []
    for _ in range(n_dirs):
        r = torch.randn(B, m, generator=generator)
        r = r / r.norm(dim=1, keepdim=True).clamp_min(1e-12)
        dirs.append(r)
    return dirs


def robust_region(model, x, cfg, seed=0):
    """Per concept, on the frozen model, critical radius r_i along:
      set 1 = worst-case (trained-against) direction = grad of x'_i wrt h1,
      set 2 = held-out random directions (unseen).
    Returns length-n arrays r_set1, r_set2 and per-set saturation flags (SPEC 6)."""
    n = x.shape[1]
    active = x > 0
    generator = torch.Generator().manual_seed(30_000 + seed)
    r_set1 = torch.full((n,), float("nan"))
    r_set2 = torch.full((n,), float("nan"))
    sat1 = torch.full((n,), float("nan"))
    sat2 = torch.full((n,), float("nan"))

    for i in range(n):
        idx = torch.nonzero(active[:, i], as_tuple=True)[0]
        if idx.numel() < 8:
            continue
        idx = idx[: cfg.probe_samples]
        xi = x[idx]
        h0 = model.encode(xi).detach()
        # cap radius relative to the typical latent scale (right-censoring bound)
        cap = cfg.radius_cap_rel * h0.norm(dim=1).mean().item()

        # set 1: worst-case direction = gradient of x'_i wrt h1 (what LAT practised)
        h_req = h0.clone().requires_grad_(True)
        out = _recon_feature_i(model, h_req, i).sum()
        (grad,) = torch.autograd.grad(out, h_req)
        g_unit = grad / grad.norm(dim=1, keepdim=True).clamp_min(1e-12)

        with torch.no_grad():
            base_val = model.decode(h0)[:, i]

        r1, s1 = critical_radius_along(model, h0, base_val, i, g_unit,
                                       cfg.tau, cfg.radius_bisect_iters, cap)
        r_set1[i], sat1[i] = r1, float(s1)

        # set 2: held-out random directions (unseen)
        dirs = _random_unit_dirs(h0.shape[0], h0.shape[1], cfg.n_random_dirs, generator)
        rs = [critical_radius_along(model, h0, base_val, i, u,
                                    cfg.tau, cfg.radius_bisect_iters, cap) for u in dirs]
        r_set2[i] = float(sum(r for r, _ in rs) / len(rs))
        sat2[i] = float(sum(s for _, s in rs) / len(rs))

    return {"r_set1": r_set1, "r_set2": r_set2, "sat_set1": sat1, "sat_set2": sat2}
