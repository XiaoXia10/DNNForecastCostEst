#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sampling strategies for choosing which (resolution, horizon) cells to train.
The framework replaces a full grid sweep with a sparse sample + extrapolation.
"""

import numpy as np

RESOLUTION_HOURS = {"1h": 1, "4h": 4, "8h": 8, "12h": 12, "24h": 24}


def full_grid(resolutions, horizons):
    """Return every (resolution, horizon) pair in the study grid."""
    return [(r, h) for r in resolutions for h in horizons]


def sample_random(resolutions, horizons, budget, seed=42):
    rng = np.random.default_rng(seed)
    grid = full_grid(resolutions, horizons)
    idx = rng.choice(len(grid), size=min(budget, len(grid)), replace=False)
    return [grid[i] for i in sorted(idx)]


def sample_lhs(resolutions, horizons, budget, seed=42):
    """
    Latin Hypercube Sampling: guarantees coverage across both axes
    before filling remaining budget randomly.
    """
    rng = np.random.default_rng(seed)
    n_res, n_hor = len(resolutions), len(horizons)
    n = min(budget, n_res * n_hor)

    res_perm = rng.permutation(n_res)
    hor_perm = rng.permutation(n_hor)

    cells = []
    for i in range(min(n, n_res)):
        cells.append((resolutions[res_perm[i]], horizons[hor_perm[i % n_hor]]))

    if len(cells) < n:
        grid = full_grid(resolutions, horizons)
        remaining = [c for c in grid if c not in cells]
        rng.shuffle(remaining)
        cells += remaining[: n - len(cells)]

    return sorted(cells, key=lambda c: (resolutions.index(c[0]), horizons.index(c[1])))


def sample_corners(resolutions, horizons, budget, seed=42):
    """
    Prioritise the four corners + centre of the grid — these anchor the
    power-law fit — then fill remaining budget with random cells.
    """
    rng = np.random.default_rng(seed)
    corners = [
        (resolutions[0],  horizons[0]),
        (resolutions[0],  horizons[-1]),
        (resolutions[-1], horizons[0]),
        (resolutions[-1], horizons[-1]),
    ]
    mid_r = resolutions[len(resolutions) // 2]
    mid_h = horizons[len(horizons) // 2]
    priority = list(dict.fromkeys(corners + [(mid_r, mid_h)]))  # deduplicated

    grid = full_grid(resolutions, horizons)
    remaining = [c for c in grid if c not in priority]
    rng.shuffle(remaining)

    selected = priority + remaining
    return selected[:budget]


def sample_cells(resolutions, horizons, budget, strategy="lhs", seed=42):
    """
    Return a list of (resolution, horizon) pairs to train.

    Parameters
    ----------
    resolutions : list[str]   e.g. ["1h", "4h", "8h", "12h", "24h"]
    horizons    : list[int]   e.g. [3, 6, 9, 12]
    budget      : int         number of cells to sample
    strategy    : str         "random" | "lhs" | "corners"
    seed        : int
    """
    strategies = {"random": sample_random, "lhs": sample_lhs, "corners": sample_corners}
    if strategy not in strategies:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose: {list(strategies)}")
    return strategies[strategy](resolutions, horizons, budget, seed)
