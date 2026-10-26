"""Toolbox for interpolating solid-waffle outputs."""

import math

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


def multimedsmooth_expand(data, n=4096, axis=-1, ignore=None):
    """Expands a possibly-flattened data cube into (4096, 4096).

    Parameters
    ----------
    data : np.ndarray
        Data cube, including one or more axes that need to be shortened.
    n : int, optional
        Size of grid onto which to expand; default is 4096 (so expands that direction
        onto a 4096 x 4096 array).
    axis : int or (int, int), optional
        The axis that corresponds to the super-pixel directions. If an int, takes that
        axis; if a tuple, then has 2 super-pixel axes (y-direction first, then x).
    ignore : float or list, optional
        If a float, ignore exactly that value (useful for integers that represent exactly
        in float, e.g., 0).
        If a list, ignores values outside that range.

    Returns
    -------
    np.ndarray
        Data cube with the indicated axis/axes expanded. If it is 2 axes (`axis` is of type
        (int, int)) then expands both; if 1 axis, then also reshapes that axis into two axes
        (so the dimensionality increases by 1).

    Notes
    -----
    The expanded array has the pixel axes in the same place as the original super-pixel axes.
    If the original super-pixel axis was flattened, then new array has two axes at that location
    (with y first).

    """

    # convert the '1 axis' picture into 2 axes
    if isinstance(axis, int):
        ax_y = axis if axis >= 0 else len(np.shape(data)) + axis
        sh_old = np.shape(data)
        nside2 = sh_old[ax_y]
        ny = nx = math.isqrt(nside2)
        if ny * nx != nside2:
            raise ValueError(f"cannot reshape axis of length {nside2} into a square array")
        sh = sh_old[:ax_y] + (ny, nx) + sh_old[ax_y + 1 :]
        data = data.reshape(sh)
        ax_x = ax_y + 1
    else:
        # if a tuple, just extract the information
        sh = np.shape(data)
        if len(axis) != 2:
            raise ValueError(f"axis must be either int or (int, int); {axis} not allowed.")
        (ax_y, ax_x) = axis
        ny = sh[ax_y]
        nx = sh[ax_x]

    if ny != nx:
        raise ValueError("Can't operate on non-square array.")

    # Now permute these to the end and flatten the rest.
    _d = np.moveaxis(data, [ax_y, ax_x], [-2, -1])
    shd = np.shape(_d)
    newdata = multimedsmooth(_d.reshape((-1, ny, nx)), 1, expand=n // ny, ignore=ignore).reshape(
        shd[:-2] + (n, n)
    )

    return np.moveaxis(newdata, [-2, -1], [ax_y, ax_x])
