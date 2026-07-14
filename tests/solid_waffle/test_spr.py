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
        ]
    )

    # now check the outputs
    alphamap = str(tmp_path) + "/SPRALL_alpha.fits"
    cleanup.append(alphamap)
    target = [0.018, 0.015, 0.001, 0.018, 0.001, 0.015, 0.001, 0.018, 0.001, 0.015, 0.001, 0.0165, 0.0]
    with fits.open(alphamap) as amap:
        assert np.shape(amap[0].data) == (13, 512, 512)
        for j in range(13):
            assert np.all(amap[0].data[j] < target[j] + 0.0004)
            assert np.all(amap[0].data[j] > target[j] - 0.0004)
    cleanup.append(str(tmp_path) + "/SPRALL_sprmean.fits")
    cleanup.append(str(tmp_path) + "/SPRALL_sprdark.fits")

    for f in cleanup:
        os.remove(f)
