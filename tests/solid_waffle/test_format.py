"""Test loading different file formats."""

import os

import asdf
import numpy as np
import pytest
from astropy.io import fits
from solid_waffle.pyirc import (
    IndexDictionary,
    gen_nl_cube,
    get_nside,
    get_num_slices,
    load_segment,
    pixel_data,
    ref_array,
    ref_array_block,
    ref_array_onerow,
)


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

    with pytest.raises(ValueError):
        load_segment(fn, fmt, [698, 702, 298, 302], [5], True, use_fitsio=False)

    # calling the same frame
    y = load_segment(fn, fmt, [698, 702, 298, 302], [5, 5], True)
    assert np.allclose(x, y[1])

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


def test_format6(tmp_path):
    """Test for format #6."""

    # set up file
    fn = str(tmp_path) + "/test6.fits"
    arr = np.zeros((5, 4096, 4096), dtype=np.uint16)
    for j in range(5):
        arr[j, 4:-4, 4:-4] = 1000 * (j + 1)
    arr[4, 300, 700] = 25000
    arr[4, :512, :4] = 50
    arr[:, 2560:2688, 1536:1664] = np.array([1000, 1490, 1960, 2410, 2840])[:, None, None]
    hdulist = [fits.PrimaryHDU(), fits.ImageHDU(arr[None, ...])]
    fits.HDUList(hdulist).writeto(fn, overwrite=True)
    fmt = 6

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

    # calling the same frame
    y = load_segment(fn, fmt, [698, 702, 298, 302], [5, 5], True)
    assert np.allclose(x, y[1])

    # reference test
    ra = ref_array([fn], fmt, 16, [3, 5], True)
    assert np.shape(ra) == (1, 16, 5)
    assert np.allclose(ra[0, :, 0], 65535)
    assert np.allclose(ra[0, 2:, 1], 65535)
    assert np.allclose(ra[0, :2, 1], 65510)
    assert np.allclose(ra[0, :, 2], 0)
    assert np.allclose(ra[0, :2, 3], 25)
    assert np.allclose(ra[0, 2:, 3], 0)
    assert np.allclose(ra[0, :, 4], 0)
    ra2 = ref_array_onerow([fn], fmt, 1, 16, [3, 5], True)
    print(np.shape(ra2))
    assert np.allclose(ra[:, 1, :], ra2[:, 1, :])
    ra3 = ref_array_block([fn], fmt, [256, 512], [3, 5], True)
    assert np.allclose(ra[:, 1, :], ra3)
    with pytest.raises(ValueError):
        ref_array_block([fn], fmt, [256], [3, 5], True)

    # polynomial cube
    swi = IndexDictionary(0)
    swi.addhnl(3)
    output_array, fit_array, deriv_array, coefs_array = gen_nl_cube(
        [fn], fmt, [1, 1, 5], (32, 32), 0.0, "abs", swi, True
    )
    assert np.shape(output_array) == (5, 32, 32)
    assert np.shape(fit_array) == (5, 32, 32)
    assert np.shape(deriv_array) == (5, 32, 32)
    assert np.shape(coefs_array) == (4, 32, 32)
    assert np.all(np.abs(coefs_array[:, 20, 12] - np.array([0, 500, -10, 0])) < 1.0e-5)
    assert np.all(np.abs(coefs_array[:, 21, 12] - np.array([0, 1000, 0, 0])) < 1.0e-5)

    # pixel data
    pixarr = pixel_data([fn], fmt, [1536, 1664, 2560, 2624], [2, 3], [], True)
    assert np.shape(pixarr) == (2, 2, 64, 128)
    assert np.all(pixarr[1] == 1)

    os.remove(fn)


def test_format7(tmp_path):
    """Test for format #7."""

    # set up file
    fn = str(tmp_path) + "/test7.fits"
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

    x = load_segment(fn, fmt, [698, 702, 298, 302], [5, 5], True)
    assert np.allclose(x[0], x[1])

    os.remove(fn)


def test_format_nonexistent(tmp_path):
    """Test for nonexistent format."""

    # set up file
    fn = str(tmp_path) + "/test7.fits"
    arr = np.zeros((5, 2048, 2048), dtype=np.uint16)
    for j in range(5):
        arr[j, 4:-4, 4:-4] = 1000 * (j + 1)
    arr[4, 300, 700] = 25000
    hdulist = [fits.PrimaryHDU()]
    for t in range(5):
        hdulist.append(fits.ImageHDU(65535 - arr[t, :, :]))
    fits.HDUList(hdulist).writeto(fn, overwrite=True)
    fmt = -1

    # now the tests
    with pytest.raises(ValueError):
        get_num_slices(fmt, fn)
    with pytest.raises(ValueError):
        load_segment(fn, fmt, [698, 702, 298, 302], [5], True, use_fitsio=False)

    os.remove(fn)
