"""SPR reduction test."""

import os

import numpy as np
from astropy.io import fits
from solid_waffle.spr_reduce import run_spr_reduce


def test_spr(tmp_path):
    """Test for SPR measurement functionality."""
    stem = str(tmp_path) + "/sprt_"
    cleanup = []
    fmt = 5

    # make darks
    dk = np.zeros((11, 4096, 4096), dtype=np.uint16)
    dk[...] = 2000
    for k in range(2):
        fdark = stem + f"dark_{k+1:03d}.fits"
        fits.PrimaryHDU(dk).writeto(fdark)
        cleanup.append(fdark)

    # make reset grid
    rx = np.zeros(512, dtype=int)
    for j in range(16):
        rx[32 * j : 32 * j + 16] = np.arange(0, 128, 8) + 256 * j + 8
        rx[32 * j + 16 : 32 * j + 32] = 256 * j + 247 - np.arange(0, 128, 8)[::-1]
    ry = np.arange(7, 4091, 8)

    # reset kernel
    kernel = np.array([[0.001, 0.015, 0.001], [0.018, 0.0, 0.018], [0.001, 0.015, 0.001]])
    kernel[1, 1] = 1 - np.sum(kernel)
    for kx in range(512):
        for ky in range(511):
            target = dk[1, ry[ky] - 1 : ry[ky] + 2, rx[kx] - 1 : rx[kx] + 2]
            target[...] += np.round((4800 + 200.0 * np.cos(kx + 17 * ky)) * kernel).astype(np.uint16)
    for k in range(3):
        fspr = stem + f"spr_{k+1:03d}.fits"
        fits.PrimaryHDU(dk[:2, :, :]).writeto(fspr)
        cleanup.append(fspr)

    run_spr_reduce(
        [
            None,
            stem + "spr_001.fits",
            str(tmp_path) + "/SPRALL",
            f"-f={fmt:d}",
            "-n=3",
            "-p=1",
            "-d=" + stem + "dark_001.fits",
            "-nd=2",
            "-sca=999",
            "-i",
            "-a=1",
        ],
        verbose=True,
    )

    # now check the outputs
    alphamap = str(tmp_path) + "/SPRALL_alpha.fits"
    cleanup2 = [alphamap]
    target = [0.018, 0.015, 0.001, 0.018, 0.001, 0.015, 0.001, 0.018, 0.001, 0.015, 0.001, 0.0165, 0.0]
    with fits.open(alphamap) as amap:
        assert np.shape(amap[0].data) == (13, 512, 512)
        for j in range(13):
            assert np.all(amap[0].data[j] < target[j] + 0.0004)
            assert np.all(amap[0].data[j] > target[j] - 0.0004)
    cleanup2.append(str(tmp_path) + "/SPRALL_sprmean.fits")
    cleanup2.append(str(tmp_path) + "/SPRALL_sprdark.fits")

    # try again in dict mode
    for f in cleanup2:
        os.remove(f)

    run_spr_reduce(
        {
            "IN": stem + "spr_001.fits",
            "OUT": str(tmp_path) + "/SPRALL",
            "-f": fmt,
            "-n": 3,
            "-p": 1,
            "-d": stem + "dark_001.fits",
            "-nd": 2,
            "-sca": 999,
            "-i": True,
            "-a": 1,
        },
        verbose=True,
    )
    target = [0.018, 0.015, 0.001, 0.018, 0.001, 0.015, 0.001, 0.018, 0.001, 0.015, 0.001, 0.0165, 0.0]
    with fits.open(alphamap) as amap:
        assert np.shape(amap[0].data) == (13, 512, 512)
        for j in range(13):
            assert np.all(amap[0].data[j] < target[j] + 0.0004)
            assert np.all(amap[0].data[j] > target[j] - 0.0004)

    for f in cleanup2:
        os.remove(f)

    # this time extract non-linearity from a summary file
    # print to the summary file first
    stxt = (
        "# Columns:\n"
        "#   1, X (super pixel grid)\n"
        "#   2, Y (super pixel grid)\n"
        "#   3, number of good pixels\n"
        "#   4, raw gain (e/DN)\n"
        "#   5, alpha-corrected gain (e/DN)\n"
        "#   6, alpha,beta-corrected gain (e/DN)\n"
        "#   7, IPC alpha horizontal\n"
        "#   8, IPC alpha vertical\n"
        "#   9, nonlinearity beta (e^-1)\n"
        "#  10, charge per time slice (e)\n"
        "#  11, IPC alpha diagonal (if computed; otherwise all 0s)\n"
        "#  12, additional non-linearity coefficient, order 2 (DN^-1)\n"
        "#  13, additional non-linearity coefficient, order 3 (DN^-2)\n"
        "#\n"
    )
    ng = np.array([124] + [128] * 30 + [124])
    g = 1.6
    a = 0.01
    nl2 = -2.0e-5  # exaggerated for the test! want to see if SPR 'corrects' and returns a smaller alpha
    nl3 = -1.0e-11
    current = 1500.0
    for iy in range(32):
        for ix in range(32):
            stxt += (
                f"{ix:3d} {iy:3d} {ng[ix]*ng[iy]:15.7E} {g/(1-8*a):15.7E} {g:15.7E} {g:15.7E} "
                f"{a:15.7E} {a:15.7E} {0.:15.7E} {current:15.7E} {0.:15.7E} {nl2:15.7E} {nl3:15.7E}\n"
            )
    with open(str(tmp_path) + "/summary.txt", "w") as fi:
        fi.write(stxt)

    # the run
    run_spr_reduce(
        {
            "IN": stem + "spr_001.fits",
            "OUT": str(tmp_path) + "/SPRALL",
            "-f": fmt,
            "-n": 3,
            "-p": 1,
            "-d": stem + "dark_001.fits",
            "-nd": 2,
            "-sca": 999,
            "-i": True,
            "-a": 1,
            "-nl": str(tmp_path) + "/summary.txt",
        },
        verbose=True,
    )
    target = [0.018, 0.015, 0.001, 0.018, 0.001, 0.015, 0.001, 0.018, 0.001, 0.015, 0.001, 0.0165, 0.0]
    with fits.open(alphamap) as amap:
        assert np.shape(amap[0].data) == (13, 512, 512)

        # offsets
        q = np.median(amap[0].data, axis=(1, 2)) - target
        assert -0.002 < q[0] < -0.0012
        assert np.all(q[1::2] > -0.002)
        assert np.all(q[1::2] < -0.0012)
        assert np.all(np.abs(q[2:12:2]) < 1.0e-4)

    for f in cleanup + cleanup2:
        os.remove(f)
