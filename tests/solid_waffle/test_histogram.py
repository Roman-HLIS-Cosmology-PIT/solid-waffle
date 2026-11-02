"""Tests for the histogram function."""

import numpy as np
import pytest
from astropy.io import fits
from solid_waffle.histograms import main as histograms_main

# Expected counts in bins 2555..2565 of channel 2 for the standard 4096x4096 test data.
SNAPSHOT = np.array([1028, 1076, 884, 1016, 660, 0, 2308, 1068, 992, 1188, 1160], dtype=np.int32)


def _make_cube(nside, nread):
    """Builds a test data cube of shape (nread, nside, nside) with a smooth varying signal."""

    _a = np.linspace(0, nside - 1, nside)
    x, y = np.meshgrid(_a, _a)
    data = np.zeros((nread, nside, nside), dtype=np.uint16)
    for j in range(nread):
        a = 1.0 + (x + y) / 8192.0
        sig = 2000.0 + 0.5 * y + 1000.0 * j + a * (20.0 * np.cos(x / 100.0) + 15.0 * np.sin(y / 10.0))
        sfrac, _ = np.modf(sig / 128.0)
        sig -= np.where(sfrac < 0.01, sfrac, 0.01) * 128
        data[j, :, :] = np.clip(np.round(sig + 0.501), 0, 65535)
    return data


def _expected_channel_counts(nside, nread, nfile):
    """Number of pixels that should be counted in each of the 32 channels."""

    chw = nside // 32
    widths = np.full(32, chw)
    widths[0] = widths[31] = chw - 4  # reference pixels trimmed on the left and right edges
    nrows = nside - 8  # reference pixels trimmed on the top and bottom
    return nrows * widths * (nread - 1) * nfile  # the time loop skips one read


def _run(tmp_path, fileformat, first_file, nfile):
    """Runs the histogram script and returns the output table."""

    outfile = str(tmp_path / "hist.txt")
    histograms_main(
        [None, "-f", str(fileformat), "-i", str(tmp_path / first_file), "-o", outfile, "-n", str(nfile)]
    )
    return np.loadtxt(outfile).astype(np.int64)


def test_histogram_dcl(tmp_path):
    """Checks the date-stamped file names, including a skipped and a changed date."""

    N = 4
    nread = 5
    data = _make_cube(4096, nread)
    stamp = 20260221  # the input file says 20260220, so the search must skip ahead
    for i in range(1, N + 1):
        fits.PrimaryHDU(data).writeto(tmp_path / f"{stamp:08d}T_{i:03d}.fits", overwrite=True)
        if i == 3:
            stamp += 1

    hist = _run(tmp_path, 1, "20260220T_001.fits", N)
    assert np.all(np.abs(hist[2555:2566, 2] - SNAPSHOT) <= 1)
    assert np.array_equal(hist.sum(axis=0), _expected_channel_counts(4096, nread, N))


def test_histogram_general(tmp_path):
    """Checks plain file names of the form body_001.fits."""

    N = 4
    nread = 5
    data = _make_cube(4096, nread)
    for i in range(1, N + 1):
        fits.PrimaryHDU(data).writeto(tmp_path / f"myfile_{i:03d}.fits", overwrite=True)

    hist = _run(tmp_path, 1, "myfile_001.fits", N)
    assert np.all(np.abs(hist[2555:2566, 2] - SNAPSHOT) <= 1)
    assert np.array_equal(hist.sum(axis=0), _expected_channel_counts(4096, nread, N))


def test_all_channels_2048(tmp_path):
    """Checks that a 2048x2048 detector (format 2) fills all 32 channels, not just the first 16."""

    N = 2
    nread = 3
    data = _make_cube(2048, nread)
    for i in range(1, N + 1):
        fits.PrimaryHDU(data).writeto(tmp_path / f"small_{i:03d}.fits", overwrite=True)

    hist = _run(tmp_path, 2, "small_001.fits", N)
    assert np.all(hist.sum(axis=0) > 0)
    assert np.array_equal(hist.sum(axis=0), _expected_channel_counts(2048, nread, N))


def test_every_file_is_read(tmp_path):
    """Checks that each input file is used once (each file is a different constant value)."""

    N = 3
    nread = 3
    nside = 2048
    for i in range(1, N + 1):
        data = np.full((nread, nside, nside), 1000 * i, dtype=np.uint16)
        fits.PrimaryHDU(data).writeto(tmp_path / f"const_{i:03d}.fits", overwrite=True)

    hist = _run(tmp_path, 2, "const_001.fits", N)
    per_file = _expected_channel_counts(nside, nread, 1)
    for i in range(1, N + 1):
        assert np.array_equal(hist[1000 * i, :], per_file)
    assert np.array_equal(hist.sum(axis=0), N * per_file)  # nothing landed in any other bin


def test_no_arguments():
    """Checks that calling with no arguments raises an error."""

    with pytest.raises(ValueError, match="histograms.py called with no arguments"):
        histograms_main(["oops"])


def test_bad_file_name(tmp_path):
    """Checks that a name not ending in _NNN.fits raises an error."""

    with pytest.raises(ValueError, match="Match failed"):
        histograms_main(
            [None, "-f", "1", "-i", str(tmp_path / "oops.doesnt.exist"), "-o", "x.txt", "-n", "1"]
        )


@pytest.mark.parametrize("first_file", ["20260220T_001.fits", "myfile_001.fits"])
def test_missing_file(tmp_path, first_file):
    """Checks that asking for more files than exist raises an error, for both naming styles."""

    data = np.zeros((2, 2048, 2048), dtype=np.uint16)
    fits.PrimaryHDU(data).writeto(tmp_path / first_file, overwrite=True)
    with pytest.raises(FileNotFoundError, match="Failed to find file 1"):
        _run(tmp_path, 2, first_file, 2)
