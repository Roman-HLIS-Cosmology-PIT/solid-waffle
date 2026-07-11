"""Test loading different file formats."""

import os

import asdf
import numpy as np
import pytest
from astropy.io import fits
from solid_waffle.pyirc import get_nside, get_num_slices, load_segment


def test_format2(tmp_path):
    """Test for format #2."""

    # set up file
    fn = str(tmp_path) + "/test2.fits"
    arr = np.zeros((5, 2048, 2048), dtype=np.uint16)
    for j in range(5):
        arr[j, 4:-4, 4:-4] = 1000 * (j + 1)
    arr[4, 300, 700] = 25000
    fits.PrimaryHDU(arr).writeto(fn, overwrite=True)
    fmt = 2

    # now the tests
    assert get_nside(fmt) == 2048
    assert get_num_slices(fmt, fn) == 5
    test = np.zeros((4, 4))
    test[:, :] = 60535
    test[2, 2] = 40535
    for use_fitsio in [True, False]:
        x = load_segment(fn, fmt, [698, 702, 298, 302], [5], True, use_fitsio=use_fitsio)
        assert np.allclose(x, test)

    os.remove(fn)


def test_format3(tmp_path):
    """Test for format #3."""

    # set up file
    fn = str(tmp_path) + "/test3.fits"
    arr = np.zeros((5, 4096, 4096), dtype=np.uint16)
    for j in range(5):
        arr[j, 4:-4, 4:-4] = 1000 * (j + 1)
    arr[4, 300, 700] = 25000
    hdulist = [fits.PrimaryHDU()]
    for t in range(5):
        hdulist.append(fits.ImageHDU(65535 - arr[t, :, :]))
    fits.HDUList(hdulist).writeto(fn, overwrite=True)
    fmt = 3

    # now the tests
    assert get_nside(fmt) == 4096
    assert get_num_slices(fmt, fn) == 5
    test = np.zeros((4, 4))
    test[:, :] = 60535
    test[2, 2] = 40535
    x = load_segment(fn, fmt, [698, 702, 298, 302], [5], True)
    assert np.allclose(x, test)
    with pytest.raises(ValueError):
        load_segment(fn, fmt, [698, 702, 298, 302], [5], True, use_fitsio=False)

    os.remove(fn)


def test_format4(tmp_path):
    """Test for format #4."""

    # set up file
    fn = str(tmp_path) + "/test4.fits"
    arr = np.zeros((5, 4096, 4096), dtype=np.uint16)
    for j in range(5):
        arr[j, 4:-4, 4:-4] = 1000 * (j + 1)
    arr[4, 300, 700] = 25000
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(65535 - arr[None, ...])]).writeto(fn, overwrite=True)
    fmt = 4

    # now the tests
    assert get_nside(fmt) == 4096
    assert get_num_slices(fmt, fn) == 5
    test = np.zeros((4, 4))
    test[:, :] = 60535
    test[2, 2] = 40535
    x = load_segment(fn, fmt, [698, 702, 298, 302], [5], True)
    assert np.allclose(x, test)

    os.remove(fn)


def test_format5(tmp_path):
    """Test for format #5."""

    # set up file
    fn = str(tmp_path) + "/test5.fits"
    arr = np.zeros((5, 4096, 4096), dtype=np.uint16)
    for j in range(5):
        arr[j, 4:-4, 4:-4] = 1000 * (j + 1)
    arr[4, 300, 700] = 25000
    fits.HDUList([fits.PrimaryHDU(arr)]).writeto(fn, overwrite=True)
    fmt = 5

    # now the tests
    assert get_nside(fmt) == 4096
    assert get_num_slices(fmt, fn) == 5
    test = np.zeros((4, 4))
    test[:, :] = 60535
    test[2, 2] = 40535
    x = load_segment(fn, fmt, [698, 702, 298, 302], [5], True)
    assert np.allclose(x, test)

    os.remove(fn)


def test_format7(tmp_path):
    """Test for format #7."""

    # set up file
    fn = str(tmp_path) + "/test3.fits"
    arr = np.zeros((5, 2048, 2048), dtype=np.uint16)
    for j in range(5):
        arr[j, 4:-4, 4:-4] = 1000 * (j + 1)
    arr[4, 300, 700] = 25000
    hdulist = [fits.PrimaryHDU()]
    for t in range(5):
        hdulist.append(fits.ImageHDU(65535 - arr[t, :, :]))
    fits.HDUList(hdulist).writeto(fn, overwrite=True)
    fmt = 7

    # now the tests
    assert get_nside(fmt) == 2048
    assert get_num_slices(fmt, fn) == 5
    test = np.zeros((4, 4))
    test[:, :] = 60535
    test[2, 2] = 40535
    x = load_segment(fn, fmt, [698, 702, 298, 302], [5], True)
    assert np.allclose(x, test)
    with pytest.raises(ValueError):
        load_segment(fn, fmt, [698, 702, 298, 302], [5], True, use_fitsio=False)

    os.remove(fn)


def test_format2001(tmp_path):
    """Test for format #2001."""

    # set up file
    fn = str(tmp_path) + "/test2001.asdf"
    arr = np.zeros((5, 4096, 4096), dtype=np.uint16)
    for j in range(5):
        arr[j, 4:-4, 4:-4] = 1000 * (j + 1)
    arr[4, 300, 700] = 25000
    asdf.AsdfFile({"roman": {"data": arr}}).write_to(fn)
    fmt = 2001

    # now the tests
    assert get_nside(fmt) == 4096
    assert get_num_slices(fmt, fn) == 5
    test = np.zeros((4, 4))
    test[:, :] = 60535
    test[2, 2] = 40535
    x = load_segment(fn, fmt, [698, 702, 298, 302], [5], True)
    assert np.allclose(x, test)

    os.remove(fn)
