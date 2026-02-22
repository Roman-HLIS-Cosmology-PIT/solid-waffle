"""Test for the histogram function."""

import numpy as np
from astropy.io import fits
from solid_waffle.histograms import main as histograms_main


def test_histogram(tmp_path):
    """
    Test function for making histograms.

    Parameters
    ----------
    tmp_path : str or pathlib.Path
        Directory in which to run the test.

    Returns
    -------
    None

    """

    # test parameters
    N = 4
    nread = 5

    # get coordinates
    _a = np.linspace(0, 4095, 4096)
    x, y = np.meshgrid(_a, _a)
    del _a

    # make the data cubes
    stamp = 20260221
    for i in range(1,N+1):
        data = np.zeros((nread, 4096, 4096), dtype=np.uint16)
        for j in range(nread):
            a = 1. + (x + y) / 8192.0
            sig = 2000. + 0.5 * y + 1000. * j + a * (20.0 * np.cos(x / 100.) + 15.0 * np.sin(y / 10.0))
            sfrac, _ = np.modf(sig / 128.0)
            sig -= np.where(sfrac < 0.01, sfrac, 0.01) * 128
            data[j, :, :] = np.clip(np.round(sig + 0.501), 0, 65535)
        fits.PrimaryHDU(data).writeto(str(tmp_path) + f"/{stamp:08d}T_{i:03d}.fits", overwrite=True)
        if i == 3:
            stamp += 1

    outfile = str(tmp_path) + "/hist.txt"
    histograms_main([None, "-f", "1", "-i", str(tmp_path) + "/20260220T_001.fits", "-o", outfile, "-n", str(N)])
    data = np.loadtxt(outfile).astype(np.int32)
    diff = data[2555:2566, 2] - np.array([1028, 1076, 884, 1016, 660, 0, 2308, 1068, 992, 1188, 1160], dtype=np.int32)
    assert np.all(np.abs(diff) <= 1)

def test_histogram2(tmp_path):
    """
    Test function for making histograms.

    Parameters
    ----------
    tmp_path : str or pathlib.Path
        Directory in which to run the test.

    Returns
    -------
    None

    """

    # test parameters
    N = 4
    nread = 5

    # get coordinates
    _a = np.linspace(0, 4095, 4096)
    x, y = np.meshgrid(_a, _a)
    del _a

    # make the data cubes
    for i in range(1,N+1):
        data = np.zeros((nread, 4096, 4096), dtype=np.uint16)
        for j in range(nread):
            a = 1. + (x + y) / 8192.0
            sig = 2000. + 0.5 * y + 1000. * j + a * (20.0 * np.cos(x / 100.) + 15.0 * np.sin(y / 10.0))
            sfrac, _ = np.modf(sig / 128.0)
            sig -= np.where(sfrac < 0.01, sfrac, 0.01) * 128
            data[j, :, :] = np.clip(np.round(sig + 0.501), 0, 65535)
        fits.PrimaryHDU(data).writeto(str(tmp_path) + f"/myfile_{i:03d}.fits", overwrite=True)

    outfile = str(tmp_path) + "/hist.txt"
    histograms_main([None, "-f", "1", "-i", str(tmp_path) + "/myfile_001.fits", "-o", outfile, "-n", str(N)])
    data = np.loadtxt(outfile).astype(np.int32)
    diff = data[2555:2566, 2] - np.array([1028, 1076, 884, 1016, 660, 0, 2308, 1068, 992, 1188, 1160], dtype=np.int32)
    assert np.all(np.abs(diff) <= 1)
