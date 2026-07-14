"""Tests for features in pyirc."""


import numpy as np
import pytest
from solid_waffle.pyirc import IndexDictionary, pyIRC_percentile


def test_indexerr():
    """Checks index raises an error if you add things in the wrong order."""

    d = IndexDictionary(0)
    d.addhnl(4)
    with pytest.raises(ValueError):
        d.addbfe(2)


def test_percentile_cont():
    """Checks the behavior of the percentile function if discrete option is off."""

    arr = np.linspace(0, 99, 100)
    mask = np.ones(100, dtype=bool)
    x = pyIRC_percentile(arr, mask, 75, disc=False)
    assert 74.24 < x < 74.26
