"""Tests for exceptions raised by the histogram functions."""

import pytest
from astropy.io import fits
from solid_waffle.histograms import main as hist_main


def test_noargs():
    """Not enough arguments."""

    with pytest.raises(ValueError):
        hist_main([None])


def test_nomatch(tmp_path):
    """Incorrect match."""

    with pytest.raises(ValueError):
        hist_main([None, "-f", 1, "-i", str(tmp_path) + "/a/b.fits", "-o", "/dev/null", "-n", 20])


def test_missing(tmp_path):
    """Missing files."""

    fmt = str(tmp_path) + "/99999999_test"
    for k in range(1, 5):
        fits.PrimaryHDU().writeto(fmt + f"_{k:03d}.fits")

    with pytest.raises(FileNotFoundError):
        hist_main([None, "-f", 1, "-i", fmt + "_001.fits", "-o", "/dev/null", "-n", 8])
