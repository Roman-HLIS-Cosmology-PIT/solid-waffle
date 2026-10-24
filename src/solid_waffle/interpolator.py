"""Toolbox for interpolating solid-waffle outputs."""

import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator


def _medsmooth(im, dx, expand=1, ignore=None, distort=1.0e-6):
    """Takes an image, breaks it into superpixels with the bin size given,
    and then interpolates the median image back onto the original.

    Parameters
    ----------
    im : np.npdarray of float
        The 2D image to interpolate.
    dx : int
        The size of block to use for the interpolation.
    expand : int, optional
        If provided, expands the grid by this amount (e.g.: if `im` is 32x32 and `expand=2`,
        then the output is a 64x64 array).
    ignore : float or list, optional
        If a float, ignore exactly that value (useful for integers that represent exactly
        in float, e.g., 0).
        If a list, ignores values outside that range.
    distort : float, optional
        Distortion to force the Voronoi decomposition to always break degeneracies from
        discretely sampled 2D data in the same way.

    Returns
    -------
    np.npdarray of float
        A 2D image.
        The interpolated image with bad values removed, expanded if requested.

    See Also
    --------
    multimedsmooth
        A version that works in 3D as well.

    """

    (ny, nx) = np.shape(im)
    if ny % dx != 0 or nx % dx != 0:
        raise ValueError(f"can't slice ({ny},{nx}) into size {dx}")

    # ignore values
    imx = np.copy(im)
    if ignore is not None:
        if isinstance(ignore, float):
            imx[im == ignore] = np.nan
        if isinstance(ignore, list):
            imx[im < ignore[0]] = np.nan
            imx[im > ignore[1]] = np.nan

    # make subimage
    subim = np.nanmedian(imx.reshape((ny // dx, dx, nx // dx, dx)), axis=(1, 3))
    iy, ix = np.where(np.isfinite(subim))
    vals = subim[iy, ix]
    _y = (iy + 0.5) * dx - 0.5
    _x = (ix + 0.5) * dx - 0.5

    # make a circumscribing edge
    rad = np.hypot(ny, nx)
    nedge = 16
    theta = 2 * np.pi * np.linspace(0, 1 - 1 / nedge, nedge)
    _x = np.concatenate((_x, (nx - 1) / 2.0 + rad * np.cos(theta)))
    _y = np.concatenate((_y, (ny - 1) / 2.0 + rad * np.sin(theta)))
    _y += distort * _x
    vals = np.concatenate((vals, np.full((nedge,), np.nanmedian(subim))))

    interp = CloughTocher2DInterpolator(np.vstack((_x, _y)).T, vals)
    x1d = np.linspace(-0.5 * (1 - 1 / expand), nx - 0.5 * (1 + 1 / expand), nx * expand)
    y1d = np.linspace(-0.5 * (1 - 1 / expand), ny - 0.5 * (1 + 1 / expand), ny * expand)
    x, y = np.meshgrid(x1d, y1d)
    y += distort * x
    return interp(x, y).astype(im.dtype, copy=False)


def multimedsmooth(im, dx, expand=1, ignore=None, distort=1.0e-6):
    """Takes an image, breaks it into superpixels with the bin size given,
    and then interpolates the median image back onto the original.

    Parameters
    ----------
    im : np.npdarray of float
        The 2D or 3D image to interpolate.
    dx : int
        The size of block to use for the interpolation.
    expand : int, optional
        If provided, expands the grid by this amount (e.g.: if `im` is 32x32 and `expand=2`,
        then the output is a 64x64 array).
    ignore : float or list, optional
        If a float, ignore exactly that value (useful for integers that represent exactly
        in float, e.g., 0).
        If a list, ignores values outside that range.
    distort : float, optional
        Distortion to force the Voronoi decomposition to always break degeneracies from
        discretely sampled 2D data in the same way.

    Returns
    -------
    np.npdarray of float
        A 2D or 3D image.
        The interpolated image with bad values removed, expanded if requested.
        Interpolation is done independently on each slice if 3D.
    """

    # allow 2D arrays
    if len(np.shape(im)) == 2:
        return _medsmooth(im, dx, expand=expand, ignore=ignore, distort=distort)

    # this part for 3D
    (n, ny, nx) = np.shape(im)
    im_out = np.zeros((n, ny * expand, nx * expand), dtype=im.dtype)
    for k in range(n):
        im_out[k] = _medsmooth(im[k], dx, expand=expand, ignore=ignore, distort=distort)
    return im_out
