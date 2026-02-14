"""Test of default options in detector_functions."""

import numpy as np
from solid_waffle.flat_simulator.detector_functions import (
    a_symmetric_avg,
    auto_convolve_kernel,
    ipc_kernel_HV,
    K2a,
    TestKernels,
)


def test_defaults():
    """Test for default options."""

    kern = ipc_kernel_HV(0.0169, 0.0169)  # alphah,alphav for the sims
    # Convolve to get K^2
    kern2 = auto_convolve_kernel(kern)
    print(kern2)

    # Some of this flipping might not be quite right, but since these are
    # symmetric averages it's ok for now
    input_bfe_a = 1.0e6 * np.fliplr(TestKernels.get_bfe_kernel_5x5_ir())
    K2a_out = K2a(kern2, input_bfe_a, round=4)
    print("<0,0>, <1,0>, <1,1>, <2,0>, <2,1>, <2,2>:")
    print(a_symmetric_avg(K2a_out, round=4))
    print(np.around(np.fliplr(input_bfe_a), 4))

    assert np.all(kern2 == 0)  # will fail
