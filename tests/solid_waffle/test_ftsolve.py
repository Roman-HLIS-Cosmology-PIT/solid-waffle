"""ftsolve test functions"""

from solid_waffle.ftsolve import op2_to_pars, p2kernel


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
