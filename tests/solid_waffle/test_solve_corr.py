"""Test functions for predicting the flat field correlation function."""

import numpy as np
from solid_waffle.ftsolve import solve_corr, solve_corr_many


def test_solve_corr_compare():
    """Compare solve_corr to solve_corr_many."""

    # make sample BFE kernel
    bfek = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            bfek[j, i] = 4.0e-7 * np.exp(-0.5 * np.hypot(i - 2.9, j - 3.05) ** 2)
    bfek[2, 2] -= np.sum(bfek)  # sums to zero

    # settings
    N = 21
    I_ = 210.0
    g = 1.56
    betas = np.array([5.0e-7, 2.0e-12])
    sigma_a = 0.0
    tslices = [1, 4, 5, 8]  # without tn
    avals = [0.012, 0.016, 0.002]

    cf_many = solve_corr_many(bfek, N, I_, g, betas, sigma_a, tslices + [1], avals, outsize=2)
    cf_single = solve_corr(bfek, N, I_, g, betas, sigma_a, tslices, avals, outsize=2)

    print(cf_many)
    print(np.amax(np.abs(cf_many - cf_single)))
    assert np.all(cf_many) > 1.0e9  # will fail
