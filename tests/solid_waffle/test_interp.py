"""Test routines for the interpolator."""

import numpy as np
import pytest
from solid_waffle.interpolator import multimedsmooth, multimedsmooth_expand


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


def test_interp_err():
    """Checks errors are raised."""

    with pytest.raises(ValueError):
        multimedsmooth(np.zeros((16, 16)), 7)
    with pytest.raises(ValueError):
        multimedsmooth_expand(
            np.zeros(
                316,
            ),
            axis=-1,
        )
    with pytest.raises(ValueError):
        multimedsmooth_expand(np.zeros((16, 32)), 1, axis=(-2, -1))
    with pytest.raises(ValueError):
        multimedsmooth_expand(np.zeros((16, 32, 2)), 1, axis=(0, 1, 2))


def test_expand():
    """Tests expansion function."""

    data = np.zeros((3, 32, 32), dtype=np.float32)
    s = np.linspace(0, 31, 32) * 128 + 63.5
    x, y = np.meshgrid(s, s)
    data[0] = np.exp(-(np.hypot(x - 500, y - 1500) ** 2) / 2.0 / 400.0**2) + 1.0e-5
    data[1] = data[0]
    data[1, 5:10, 3] = np.nan
    data[2] = data[0]
    data[2, 5:10, 3] = 0.0

    bigdata = multimedsmooth_expand(data, axis=(-2, -1))
    assert bigdata.dtype == np.float32

    s = np.linspace(0, 4095, 4096)
    x, y = np.meshgrid(s, s)
    target = np.exp(-(np.hypot(x - 500, y - 1500) ** 2) / 2.0 / 400.0**2).astype(np.float32)
    target += 1.0e-5
    del x, y
    bd2 = bigdata - target[None, ...]
    assert np.all(np.abs(bd2[:2]) < 0.01)
    assert -1.5e-4 < bd2[0, 1170, 440] < 1.5e-4
    assert -0.004 < bd2[1, 1170, 440] < -0.003
    assert -0.8 < bd2[2, 1170, 440] < -0.7
    assert np.all(bd2[2, 1230:1250, 256:276] > 0.08)

    # test ignore
    b3 = multimedsmooth_expand(data, axis=(-2, -1), ignore=0.0) - target[None, ...]
    assert -0.004 < b3[2, 1170, 440] < -0.003
    del b3

    # now test the squashed version of this
    bigdata2 = multimedsmooth_expand(data.reshape((3, -1)), axis=-1)
    assert np.allclose(bigdata, bigdata2)
    del bigdata2

    # the permuted versions
    bigdata2 = multimedsmooth_expand(data.reshape((3, -1)).T, axis=0)
    assert np.shape(bigdata2) == (4096, 4096, 3)
    assert np.allclose(bigdata, np.transpose(bigdata2, axes=(2, 0, 1)))
    del bigdata2

    bigdata2 = multimedsmooth_expand(data.reshape((3, -1, 1)), axis=1)
    assert np.shape(bigdata2) == (3, 4096, 4096, 1)
    assert np.allclose(bigdata, bigdata2[..., 0])
    del bigdata2

    _d = np.zeros((2, 3, 32, 32, 2), np.float32)
    _d[1, ..., 0] = data
    _d[1, ..., 1] = 2 * data
    bigdata2 = multimedsmooth_expand(_d, axis=(2, 3))
    assert np.shape(bigdata2) == (2, 3, 4096, 4096, 2)
    assert np.allclose(bigdata, bigdata2[1, ..., 0])
    assert np.allclose(2 * bigdata, bigdata2[1, ..., 1], atol=4.0e-7)
    assert np.all(np.abs(bigdata2[0]) < 1.0e-7)
    del bigdata2, _d

    # grid down
    bigdata2 = multimedsmooth_expand(data.reshape((3, -1, 1)), n=1024, axis=1)
    assert np.shape(bigdata2) == (3, 1024, 1024, 1)
    bigdata_sq = np.mean(bigdata.reshape((3, 1024, 4, 1024, 4)), axis=(-3, -1))
    assert np.allclose(bigdata_sq, bigdata2[..., 0], atol=4.0e-4)
    del bigdata2, bigdata_sq

    # flattened
    bigdata2 = multimedsmooth_expand(data[2].ravel())
    assert np.shape(bigdata2) == (4096, 4096)
    assert np.allclose(bigdata[2], bigdata2)
    del bigdata2

    bigdata2 = multimedsmooth_expand(data[2].ravel(), axis=0)
    assert np.shape(bigdata2) == (4096, 4096)
    assert np.allclose(bigdata[2], bigdata2)
    del bigdata2

    # 2D, no extra axes
    bigdata2 = multimedsmooth_expand(data[2], axis=(0, 1))
    assert np.shape(bigdata2) == (4096, 4096)
    assert np.allclose(bigdata[2], bigdata2)
    del bigdata2
