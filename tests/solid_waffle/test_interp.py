"""Test routines for the interpolator."""

import numpy as np
from solid_waffle.interpolator import multimedsmooth


def test_interp():
    """Simple smoothed-interpolator test."""

    # build a simple function
    _s = np.linspace(0, 63, 64) * 16 + 7.5
    ix, iy = np.meshgrid(_s, _s)
    im = np.sin(np.pi / 100.0 * (ix + iy * np.sqrt(3.0)) / 2.0)  # sine wave, wavelength 200

    # first, just interpolate onto itself
    im2 = multimedsmooth(im, 1)
    assert np.amax(np.abs(im2 - im)) < 1.0e-9

    # now bin and put back -- check that this did something
    im2 = multimedsmooth(im, 2)
    assert 0.11 < np.amax(np.abs(im2 - im)) < 0.13
    assert 0.09 < np.amax(np.abs(im2 - 1.04 * im)) < 0.1

    # expansions
    _s = np.linspace(0, 127, 128) * 8 + 3.5
    ix2, iy2 = np.meshgrid(_s, _s)
    truth_im = np.sin(np.pi / 100.0 * (ix2 + iy2 * np.sqrt(3.0)) / 2.0)
    im2 = multimedsmooth(im, 1, expand=2)
    assert 0.015 < np.amax(np.abs(im2 - truth_im)) < 0.025
    im2 = multimedsmooth(im, 2, expand=2)
    assert 0.19 < np.amax(np.abs(im2 - truth_im)) < 0.21

    # now expand with bad values
    imc = np.copy(im)
    imc[17, 14] = np.nan
    im2 = multimedsmooth(imc, 1, expand=2)
    diff = np.abs(im2 - truth_im)
    assert 0.015 < np.amax(diff) < 0.025
    assert np.amax(diff[40:100, 40:100]) < 3.0e-4
    assert 0.005 < np.amax(diff[30:40, 24:34]) < 0.015

    # a corner region missing
    imc = im + 1.0
    imc[:2, -3:] = np.nan
    im2 = multimedsmooth(imc, 1, expand=2)
    diff = np.abs(im2 - truth_im - 1.0)
    assert 0.1 < diff[0, -1] < 0.13
    diff[0, -1] = 0.08  # prevent next test from failing
    assert np.all(diff[:3, -3:] > 0.015)
    assert np.all(diff[:3, -3:] < 0.1)
    assert np.all(diff[4:-1, 1:-1] < 0.018)
    assert np.all(diff[1:-1, 1:-5] < 0.018)

    # same but with 0's
    imc = im + 1.0
    imc[:2, -3:] = 0.0
    im2 = multimedsmooth(imc, 1, expand=2, ignore=0.0)
    diff = np.abs(im2 - truth_im - 1.0)
    assert 0.1 < diff[0, -1] < 0.13
    diff[0, -1] = 0.08  # prevent next test from failing
    assert np.all(diff[:3, -3:] > 0.015)
    assert np.all(diff[:3, -3:] < 0.1)
    assert np.all(diff[4:-1, 1:-1] < 0.018)
    assert np.all(diff[1:-1, 1:-5] < 0.018)

    # and with lists
    imc = im + 1.0
    imc[:2, -3:] = 1.0e4
    im2 = multimedsmooth(imc, 1, expand=2, ignore=[-2, 2])
    diff = np.abs(im2 - truth_im - 1.0)
    assert 0.1 < diff[0, -1] < 0.13
    diff[0, -1] = 0.08  # prevent next test from failing
    assert np.all(diff[:3, -3:] > 0.015)
    assert np.all(diff[:3, -3:] < 0.1)
    assert np.all(diff[4:-1, 1:-1] < 0.018)
    assert np.all(diff[1:-1, 1:-5] < 0.018)

    # and as 3D
    imc = im + 1.0
    imc[:2, -3:] = 1.0e4
    im2 = multimedsmooth(imc.reshape((1, 64, 64)), 1, expand=2, ignore=[-2, 2])
    assert np.shape(im2) == (1, 128, 128)
    diff = np.abs(im2[0] - truth_im - 1.0)
    assert 0.1 < diff[0, -1] < 0.13
    diff[0, -1] = 0.08  # prevent next test from failing
    assert np.all(diff[:3, -3:] > 0.015)
    assert np.all(diff[:3, -3:] < 0.1)
    assert np.all(diff[4:-1, 1:-1] < 0.018)
    assert np.all(diff[1:-1, 1:-5] < 0.018)

    # and with alternative distortion
    imc = im + 1.0
    imc[:2, -3:] = 1.0e4
    im2 = multimedsmooth(imc, 1, expand=2, ignore=[-2, 2], distort=1.0e-5)
    diff = np.abs(im2 - truth_im - 1.0)
    assert 0.1 < diff[0, -1] < 0.13
    diff[0, -1] = 0.08  # prevent next test from failing
    assert np.all(diff[:3, -3:] > 0.015)
    assert np.all(diff[:3, -3:] < 0.1)
    assert np.all(diff[4:-1, 1:-1] < 0.018)
    assert np.all(diff[1:-1, 1:-5] < 0.018)

    # now let's do a linearity test
    im = np.zeros((3, 64, 64))
    im[0] = np.sin(np.pi / 200.0 * (ix + iy * np.sqrt(3.0)) / 2.0) ** 3
    im[1] = np.cos(np.hypot(ix - 550, iy - 650) * np.pi / 200.0)
    im[2] = im[0] + 3 * im[1]
    im[:, 4, 19:24] = np.nan
    im2 = multimedsmooth(im, 1, expand=4)
    diff = im2[2] - (im2[0] + 3 * im2[1])
    assert np.all(np.abs(diff) < 1.0e-4)
