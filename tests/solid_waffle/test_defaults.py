"""Test of default options in detector_functions."""

import numpy as np
from solid_waffle.flat_simulator.detector_functions import (
    K2a,
    TestKernels,
    a_symmetric_avg,
    auto_convolve_kernel,
    ipc_kernel_HV,
)


def test_defaults():
    """Test for default options and manipulations."""

    kern = ipc_kernel_HV(0.0169, 0.0169)  # alphah,alphav for the sims
    # Convolve to get K^2
    kern2 = auto_convolve_kernel(kern)
    # print(kern2)
    assert np.abs(kern2[1, 2] - 3.151512e-02) < 1.0e-6

    # Some of this flipping might not be quite right, but since these are
    # symmetric averages it's ok for now
    input_bfe_a = 1.0e6 * np.fliplr(TestKernels.get_bfe_kernel_5x5_ir())
    K2a_out = K2a(kern2, input_bfe_a, round=4)
    # print("<0,0>, <1,0>, <1,1>, <2,0>, <2,1>, <2,2>:")
    s = a_symmetric_avg(K2a_out, round=4)
    # print(s)
    s_ref = np.array([-1.7228, 0.2815, 0.11, 0.0285, 0.0062, -0.0018], dtype=np.float64)
    assert np.all(np.abs(s - s_ref) < 1.0e-4)

    print(np.fliplr(input_bfe_a))
    assert np.abs(np.fliplr(input_bfe_a)[3, 0] + 0.0083) < 1.0e-4
