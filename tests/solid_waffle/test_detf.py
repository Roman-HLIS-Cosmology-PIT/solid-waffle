"""Tests for detector functions."""

import numpy as np
import pytest
from solid_waffle.flat_simulator import detector_functions as df


def test_calculate_ipc():
    """Test alternative versions in calculate_ipc."""

    q = np.zeros((1, 9, 9))
    q[0, 4, 4] = 1.0
    arr = df.calculate_ipc(q, [0.02, 0.01])
    q2 = np.zeros((9, 9))
    q2[4, 3] = q2[4, 5] = 0.01
    q2[3, 4] = q2[5, 4] = 0.02
    q2[4, 4] = 0.94
    assert np.allclose(q2[None, ...], arr)

    with pytest.raises(ValueError):
        df.calculate_ipc(q, [0.01, 0.02, -1])
