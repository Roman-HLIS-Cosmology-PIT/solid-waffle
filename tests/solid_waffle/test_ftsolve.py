"""ftsolve test functions"""

import numpy as np
import pytest
from solid_waffle.ftsolve import op2_to_pars, p2kernel, pad_to_N, solve_corr, solve_corr_vis


def test_nopad():
    """Test pass-through for small N."""

    arr = np.linspace(5, 10, 25).reshape((5, 5))
    assert np.allclose(arr, pad_to_N(arr, 5))


def test_p2kernel():
    """Test function for p2kernel."""

    for i in range(4):
        s = 0.4 / 2**i
        cov = [s**2, 0.5 * s**2, s**2]
        pars = op2_to_pars(0.05 * p2kernel(cov, 2))
        assert 0.04999 < pars[0] / (1 + pars[0]) < 0.05001
        assert 0.999 < pars[1] / s**2 < 1.001
        assert 0.499 < pars[2] / s**2 < 0.501
        assert 0.999 < pars[3] / s**2 < 1.001
        cov = [1.1 * s**2, -0.8 * s**2, 0.9 * s**2]
        pars = op2_to_pars(0.05 * p2kernel(cov, 2))
        assert 0.04999 < pars[0] / (1 + pars[0]) < 0.05001
        assert 1.009 < pars[1] / s**2 < 1.101
        assert -0.801 < pars[2] / s**2 < -0.799
        assert 0.899 < pars[3] / s**2 < 0.901
        pars = op2_to_pars(0.025 * p2kernel(cov, 2) + 0.025 * p2kernel([s**2, 0, s**2], 2))
        print(i, (pars[0] / (1 + pars[0]) - 0.05) * 4**i)
        assert -0.001 < (pars[0] / (1 + pars[0]) - 0.05) * 4**i < -0.0006

    # test error
    with pytest.raises(ValueError):
        p2kernel(cov, 2, N_integ=4)


def test_solve_corr():
    """Test against configuration-space corrfn generated from known inputs/simulated flats."""

    N = 21
    I_ = 1487
    g = 2.06
    betas = 5.98e-6
    tslices = [3, 11, 13, 21]
    avals = [0, 0, 0]
    avals_nl = [0, 0, 0]

    test_bfek = 1.0e-6 * np.array(
        [
            [-0.01, 0.0020, -0.0210, -0.019, 0.028],
            [0.0040, 0.0490, 0.2480, 0.01, -0.0240],
            [-0.0170, 0.2990, -1.372, 0.2840, 0.0150],
            [0.0130, 0.0560, 0.2890, 0.0390, 0.02],
            [0.035, 0.0070, 0.0380, 0.0010, 0.026],
        ]
    )
    sigma_a = np.sum(test_bfek)

    c_abcd = solve_corr(test_bfek, N, I_, g, betas, sigma_a, tslices, avals, avals_nl)
    c_abcd_target = np.array(
        [
            [3.61947355e-01, 2.24557016e-02, 5.71593702e-01, 1.72169815e-01, 3.43336634e-01],
            [2.47704582e-01, 3.72551515e-01, 2.77608325e00, 7.17298813e-01, 2.56341501e-01],
            [2.69920609e-01, 2.53043051e00, -3.12431068e02, 2.77133768e00, -2.32598281e-01],
            [-3.34069588e-01, -2.76584704e-02, 2.11413629e00, 5.23629736e-01, -1.21615142e-02],
            [2.24029752e-01, -2.55616487e-01, -3.73920791e-01, 2.59359940e-02, -2.11130075e-01],
        ]
    )
    assert np.allclose(c_abcd, c_abcd_target, atol=0.01, rtol=0.0)

    # visible function
    c_abcd = solve_corr_vis(
        test_bfek, N, I_, g, betas, sigma_a, tslices, avals, avals_nl=avals_nl, omega=0, p2=0
    )
    assert np.allclose(c_abcd, c_abcd_target, atol=0.01, rtol=0.0)

    # visible function (other branch)
    c_abcd = solve_corr_vis(
        test_bfek,
        N,
        I_,
        g,
        betas,
        sigma_a,
        tslices,
        avals,
        avals_nl=avals_nl,
        omega=1.0e-12,
        p2=np.ones((3, 3)) / 9.0,
    )
    assert np.allclose(c_abcd, c_abcd_target, atol=0.01, rtol=0.0)

    # test flipping
    tslices_roll = [13, 21, 3, 11]
    c_abcd_swap = solve_corr(test_bfek, N, I_, g, betas, sigma_a, tslices_roll, avals, avals_nl)
    assert np.allclose(c_abcd_swap[::-1, ::-1], c_abcd_target, atol=0.01, rtol=0.0)

    # test error if not square
    with pytest.raises(ValueError):
        solve_corr(test_bfek[1:-1, :], N, I_, g, betas, sigma_a, tslices, avals, avals_nl)
    with pytest.raises(ValueError):
        c_abcd = solve_corr_vis(
            test_bfek[1:-1, :],
            N,
            I_,
            g,
            betas,
            sigma_a,
            tslices,
            avals,
            avals_nl=avals_nl,
            omega=1.0e-12,
            p2=np.ones((3, 3)) / 9.0,
        )
