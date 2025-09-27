"""
Single pixel reset reduction script.

Functions
---------
run_spr_reduce
    Runs the single pixel reset reduction.

"""

import re
import sys
import time

import numpy as np
from astropy.io import fits

from . import pyirc

thisversion = 5


def run_spr_reduce(arglist, verbose=False):
    """
    Runs the single pixel reset reduction.

    Parameters
    ----------
    arglist : list or dict
        The parameters for the SPR reduction (see Notes for formats).
    verbose : bool, optional
        Whether to talk a lot.

    Returns
    -------
    None

    Notes
    -----
    In *list* format, the arguments are command-line options:

    * `arglist`[0] = not used

    * `arglist`[1] = input file name

    * `arglist`[2] = ouput stemp

    * `arglist`[3:] = options.

    The options are:

    * ``-f=<format>`` : controls input format

    * ``-n=<nfile>`` : controls number of files used (assumed sequential numbering if n>=2)

    * ``-p=<pattern>`` : controls pattern number

    * ``-d=<dark file>`` : dark data cube, same format, used for masking

    * ``-nd=<ndarks>`` : controls number of dark files used (assumed sequential numbering if n>=2)

    * ``-sca=<scanum>`` : SCA number (for output)

    * ``-sd`` : if set, subtract dark (not recommended at this time, hasn't worked as well as we had hoped)

    * ``-i`` : if set, interpolate masked pixels

    * ``-a=<parameter>`` : alternate file naming scheme. The options are:

      * ``-a=0`` : most general name. In this case, the input file name is treated as a regular expression,
        i.e., ``myfile_{:03d}.fits^5`` will be interpreted as the first file being ``myfile_005.fits``, then
        the second is ``myfile_006.fits``, and so on.

      * ``-a=1`` : default DCL numbering scheme, in which the first file is ``..._001.fits``, and then the
        subsequent files replace the ending: ``..._002.fits``, ``..._003.fits``, and so on.

      * ``-a=2`` : alternative including "_1_" in the file name (only needed for some cases).

    * ``-nl=<summary file>`` : summary file for NL information

    In *dictionary* format, the options are specified as: ``arglist["IN"]`` for the input file;
    ``arglist["OUT"]`` for the output stem; and e.g. ``arglist["-nd"]`` for the remaining options (for
    options that don't have an "=" value in the command line, you can set e.g. ``arglist["-i"]=True``).

    """

    # if a dictionary is provided, convert it to a list
    if isinstance(arglist, dict):
        argdict = arglist
        arglist = [None, argdict["IN"], argdict["OUT"]]
        for k in argdict:
            if k not in ["IN", "OUT"]:
                if isinstance(argdict[k], bool):
                    if argdict[k]:
                        arglist.append(k)
                else:
                    arglist.append(k + f"={argdict[k]}")

    if verbose:
        print("argument list ->", arglist)

    Narg = len(arglist)

    nfile = 1
    formatpars = 4
    tdark = 10
    outstem = arglist[2]
    ipc_pattern = 1
    usedark = False
    ndark = 1
    darkfile = ""
    subtr_dark = False
    interp_alpha = False
    sca = "xxxxx"
    name_scheme = 1
    use_nl = False

    for k in range(3, Narg):
        m = re.search(r"^-f=(\d+)$", arglist[k])
        if m:
            formatpars = int(m.group(1))
        m = re.search(r"^-n=(\d+)$", arglist[k])
        if m:
            nfile = int(m.group(1))
        m = re.search(r"^-p=(\d+)$", arglist[k])
        if m:
            ipc_pattern = int(m.group(1))
        m = re.search(r"^-d=(\S+)$", arglist[k])
        if m:
            usedark = True
            darkfile = m.group(1)
        m = re.search(r"^-nd=(\d+)$", arglist[k])
        if m:
            ndark = int(m.group(1))
        m = re.search(r"^-sd$", arglist[k])
        if m:
            subtr_dark = True
        m = re.search(r"^-i$", arglist[k])
        if m:
            interp_alpha = True  # plan to add functionality # noqa: F841
        m = re.search(r"^-sca=(\d+)", arglist[k])
        if m:
            sca = m.group(1)
        m = re.search(r"^-a=(\d+)", arglist[k])
        if m:
            name_scheme = int(m.group(1))
        m = re.search(r"^-nl=(\S+)$", arglist[k])
        if m:
            use_nl = True
            nlfile = m.group(1)

    N = pyirc.get_nside(formatpars)
    dmap = np.zeros((nfile, N, N))

    # Pull down information from NL file
    if use_nl:
        summaryinfo = np.loadtxt(nlfile)
        sum_nx = int(np.amax(summaryinfo[:, 0])) + 1
        sum_ny = int(np.amax(summaryinfo[:, 1])) + 1
        omax = 6
        colindex = [0] * (omax + 1)
        nlcoefs = np.zeros((omax + 1, sum_ny, sum_nx))
        with open(nlfile, "r") as f:
            for x in f:
                m = re.search(r"^\# +(\d+), additional non-linearity coefficient, order (\d+) ", x)
                if m:
                    thiscol = int(m.group(1))
                    thisord = int(m.group(2))
                    colindex[thisord] = thiscol
                    if thisord > omax:
                        raise ValueError(f"Error: increase omax={omax:d} to {thisord:d}")
            if verbose:
                print("NL information ->", sum_ny, sum_nx, omax, colindex)
            for p in range(1, omax + 1):
                if colindex[p] > 0:
                    nlcoefs[p, :, :] = summaryinfo[:, colindex[p] - 1].reshape(sum_ny, sum_nx)

    # IPC patterns:
    #
    # pattern 1 = original dx=8,dy=8, start at (8,7)
    # pattern 2 = original dx=8,dy=8, start at (6,7)
    #
    dx = dy = 8
    if ipc_pattern == 1 or ipc_pattern == 2:
        dx = dy = 8
    # < alternate dx,dy would go here >

    nx = N // dx
    ny = N // dy
    rx = np.zeros(nx, dtype=int)
    ry = np.zeros(ny, dtype=int)

    signals = np.zeros((nfile, 13, ny, nx))
    dmask = np.zeros((ny, nx))

    # List of IPC patterns (actually a list)
    if ipc_pattern == 1:
        for j in range(16):
            rx[32 * j : 32 * j + 16] = np.arange(0, 128, 8) + 256 * j + 8
            rx[32 * j + 16 : 32 * j + 32] = 256 * j + 247 - np.arange(0, 128, 8)[::-1]
        ry = np.arange(dy - 1, N, dy)
    if ipc_pattern == 2:
        for j in range(16):
            rx[32 * j : 32 * j + 16] = np.arange(0, 128, 8) + 256 * j + 6
            rx[32 * j + 16 : 32 * j + 32] = 256 * j + 249 - np.arange(0, 128, 8)[::-1]
        ry = np.arange(dy - 1, N, dy)

    # Make dark map
    if usedark:
        # Dark map
        D = np.zeros((ndark, N, N))
        for j in range(ndark):
            thisfile = darkfile + ""
            if j > 0:
                m = re.search(r"(.+_)(\d+)\.fits$", darkfile)
                if m:
                    new_index = int(m.group(2)) + j
                    thisfile = m.group(1) + f"{new_index:03d}.fits"
                else:
                    raise ValueError("Error: can't construct new file name.")
            thisframe = pyirc.load_segment(thisfile, formatpars, [0, N, 0, N], [1, 1 + tdark], True)
            D[j, :, :] = thisframe[0, :, :] - thisframe[1, :, :]
        darkframe = np.median(D, axis=0) / float(tdark)
        del D

        # Make FITS output of dark map
        hdu = fits.PrimaryHDU(darkframe)
        hdu.header["DATE"] = format(time.asctime(time.localtime(time.time())))
        hdu.header["SCA"] = sca
        hdu.header["ORIGIN"] = "spr_reduce.py"
        hdu.header["VERSION"] = f"{thisversion:d}"
        hdu.header["FILETYPE"] = "dark map for IPC masking"
        hdul = fits.HDUList([hdu])
        hdul.writeto(outstem + "_sprdark.fits", overwrite=True)

    filelist = []
    for j in range(nfile):
        thisfile = arglist[1]
        if name_scheme == 0:
            s = arglist[1].split("^")
            thisfile = s[0].format(j + int(s[1]))
        elif j > 0:
            if name_scheme == 1:
                m = re.search(r"(.+_)(\d+)\.fits$", arglist[1])
                if m:
                    new_index = int(m.group(2)) + j
                    thisfile = m.group(1) + f"{new_index:03d}.fits"
                else:
                    raise ValueError("Error: can't construct new file name.")
            if name_scheme == 2:
                m = re.search(r"(.+)_1_(.+\.fits)$", arglist[1])
                if m:
                    thisfile = m.group(1) + f"_{j+1:d}_" + m.group(2)
                else:
                    raise ValueError("Error: can't construct new file name.")

        filelist.append(thisfile)
        thisframe = pyirc.load_segment(thisfile, formatpars, [0, N, 0, N], [1, 2], True)
        dmap[j, :, :] = thisframe[0, :, :] - thisframe[1, :, :]
        if subtr_dark:
            dmap[j, :, :] -= darkframe

        # Non-linearity correction, if used
        if use_nl:
            sys.stdout.write(f"Non-linearity corrections: super-rows (of {sum_ny:d}): ")
            sys.stdout.flush()
            for iys in range(sum_ny):
                sys.stdout.write(f"{iys:d} ")
                sys.stdout.flush()
                ysmin = iys * (N // sum_ny)
                ysmax = (iys + 1) * (N // sum_ny)
                for ixs in range(sum_nx):
                    xsmin = ixs * (N // sum_nx)
                    xsmax = (ixs + 1) * (N // sum_nx)
                    S = dmap[j, ysmin:ysmax, xsmin:xsmax]  # makes subarray
                    Sf = np.copy(S)
                    # want to solve Sf = S + c_2 S^2 + c_3 S^3 + ...
                    for _ in range(20):
                        # iterative solution
                        Sp = np.copy(S)
                        for p in range(2, omax + 1):
                            Sp += nlcoefs[p, iys, ixs] * S**p
                        S += Sf - Sp
            if verbose:
                print("Done.")

        # subtractions to get "signals" (background, center, horiz, vert, diag)
        for iy in range(ny):
            yc = ry[iy]
            if yc >= 5 and yc < N - 5:
                for ix in range(nx):
                    xc = rx[ix]
                    if xc >= 5 and xc < N - 5:
                        signals[j, 0, iy, ix] = np.median(
                            np.concatenate(
                                (
                                    dmap[j, yc - 3 : yc - 1, xc - 3 : xc + 4].flatten(),
                                    dmap[j, yc + 2 : yc + 4, xc - 3 : xc + 4].flatten(),
                                    dmap[j, yc - 1 : yc + 2, xc - 3 : xc - 1].flatten(),
                                    dmap[j, yc - 1 : yc + 2, xc + 2 : xc + 4].flatten(),
                                )
                            )
                        )
                        signals[j, 1, iy, ix] = dmap[j, yc, xc] - signals[j, 0, iy, ix]
                        # horizontal
                        signals[j, 2, iy, ix] = (dmap[j, yc, xc - 1] + dmap[j, yc, xc + 1]) / 2.0 - signals[
                            j, 0, iy, ix
                        ]
                        # vertical
                        signals[j, 3, iy, ix] = (dmap[j, yc - 1, xc] + dmap[j, yc + 1, xc]) / 2.0 - signals[
                            j, 0, iy, ix
                        ]
                        # diagonal
                        signals[j, 4, iy, ix] = (
                            dmap[j, yc - 1, xc - 1]
                            + dmap[j, yc + 1, xc - 1]
                            + dmap[j, yc - 1, xc + 1]
                            + dmap[j, yc + 1, xc + 1]
                        ) / 4.0 - signals[j, 0, iy, ix]
                        # individual pixels, going from "right" to "upper-right" and then around
                        signals[j, 5, iy, ix] = dmap[j, yc, xc + 1] - signals[j, 0, iy, ix]
                        signals[j, 6, iy, ix] = dmap[j, yc + 1, xc + 1] - signals[j, 0, iy, ix]
                        signals[j, 7, iy, ix] = dmap[j, yc + 1, xc] - signals[j, 0, iy, ix]
                        signals[j, 8, iy, ix] = dmap[j, yc + 1, xc - 1] - signals[j, 0, iy, ix]
                        signals[j, 9, iy, ix] = dmap[j, yc, xc - 1] - signals[j, 0, iy, ix]
                        signals[j, 10, iy, ix] = dmap[j, yc - 1, xc - 1] - signals[j, 0, iy, ix]
                        signals[j, 11, iy, ix] = dmap[j, yc - 1, xc] - signals[j, 0, iy, ix]
                        signals[j, 12, iy, ix] = dmap[j, yc - 1, xc + 1] - signals[j, 0, iy, ix]

    # Make FITS output of difference map
    hdu = fits.PrimaryHDU(np.mean(dmap, axis=0))
    hdu.header["DATE"] = format(time.asctime(time.localtime(time.time())))
    hdu.header["SCA"] = sca
    hdu.header["ORIGIN"] = "spr_reduce.py"
    hdu.header["VERSION"] = f"{thisversion:d}"
    hdu.header["FILETYPE"] = "SPR difference map"
    hdul = fits.HDUList([hdu])
    hdul.writeto(outstem + "_sprmean.fits", overwrite=True)

    medsignals = np.median(signals, axis=0)
    if verbose:
        print(f"median signal = {np.median(medsignals[1,:,:]):8.6f} DN")
    alpha = np.zeros((13, ny, nx))

    # Masking based on the dark
    if usedark:
        for iy in range(ny):
            yc = ry[iy]
            if yc >= 5 and yc < N - 5:
                for ix in range(nx):
                    xc = rx[ix]
                    if (
                        xc >= 5
                        and xc < N - 5
                        and (
                            np.amax(darkframe[yc - 1 : yc + 2, xc - 1 : xc + 2])
                            > 1e-3 * tdark * medsignals[1, iy, ix]
                        )
                    ):
                        dmask[iy, ix] = 1

    # mask if central pixel isn't the brightest by at least a factor of 10
    # -- would usually indicate a problem
    for iy in range(ny):
        for ix in range(nx):
            if 0.1 * medsignals[1, iy, ix] < np.amax(medsignals[5:13, iy, ix]):
                dmask[iy, ix] = 1

    # alpha map
    den = (
        medsignals[1, :, :]
        + 2 * medsignals[2, :, :]
        + 2 * medsignals[3, :, :]
        + 4 * medsignals[4, :, :]
        + 1e-99
    )
    # (the 1e-99 ensures that 0's are passed through)
    alpha[:11, :, :] = medsignals[2:, :, :] / den
    alpha[11, :, :] = (alpha[0, :, :] + alpha[1, :, :]) / 2.0
    alpha[12, :, :] = dmask[:, :]

    # filling in
    if ipc_pattern == 1:
        alpha[:, -1, :] = alpha[:, -2, :]
        for j in range(16):
            alpha[:, :, 15 + 32 * j] = alpha[:, :, 14 + 32 * j]
            alpha[:, :, 16 + 32 * j] = alpha[:, :, 17 + 32 * j]

    # interpolation
    for iy in range(ny):
        for ix in range(nx):
            if dmask[iy, ix] > 0.5:
                # first try 8 nearest neighbors
                aDen = np.sum(1 - dmask[iy - 1 : iy + 2, ix - 1 : ix + 2])
                if aDen > 0:
                    for k in range(12):
                        alpha[k, iy, ix] = (
                            np.sum(
                                (1 - dmask[iy - 1 : iy + 2, ix - 1 : ix + 2])
                                * alpha[k, iy - 1 : iy + 2, ix - 1 : ix + 2]
                            )
                            / aDen
                        )

    # Make FITS output of IPC map
    hdu = fits.PrimaryHDU(alpha)
    hdu.header["DATE"] = format(time.asctime(time.localtime(time.time())))
    hdu.header["SCA"] = sca
    hdu.header["ORIGIN"] = "spr_reduce.py"
    hdu.header["VERSION"] = f"{thisversion:d}"
    hdu.header["FILETYPE"] = "IPC data cube"
    hdu.header["DX"] = str(dx)
    hdu.header["DY"] = str(dy)
    hdu.header["SLICE01"] = "alpha_horizontal"
    hdu.header["SLICE02"] = "alpha_vertical"
    hdu.header["SLICE03"] = "alpha_diagonal"
    hdu.header["SLICE04"] = "alpha dx=+1, dy= 0"
    hdu.header["SLICE05"] = "alpha dx=+1, dy=+1"
    hdu.header["SLICE06"] = "alpha dx= 0, dy=+1"
    hdu.header["SLICE07"] = "alpha dx=-1, dy=+1"
    hdu.header["SLICE08"] = "alpha dx=-1, dy= 0"
    hdu.header["SLICE09"] = "alpha dx=-1, dy=-1"
    hdu.header["SLICE10"] = "alpha dx= 0, dy=-1"
    hdu.header["SLICE11"] = "alpha dx=+1, dy=-1"
    hdu.header["SLICE12"] = "alpha (average of 4 nearest neighbors)"
    hdu.header["SLICE13"] = "mask (0 = normal, 1 = masked), not foolproof"
    for k in range(Narg):
        keyword = f"ARGV{k:02d}"
        hdu.header[keyword] = arglist[k]
    hdu.header["MASKSIZE"] = f"{int(np.sum(dmask)):d}/{nx*ny:d}"
    hdu.header["MEDSIG"] = (f"{np.median(medsignals[1,:,:]):8.2f}", "Median signal in central pixel")
    for k in range(len(filelist)):
        keyword = f"INF{k:02d}"
        hdu.header[keyword] = filelist[k]
    hdul = fits.HDUList([hdu])
    hdul.writeto(outstem + "_alpha.fits", overwrite=True)

    if verbose:
        print("median alpha information ->")
        print(np.median(alpha, axis=[1, 2]))
        print(f"Number of masked pixels = {int(np.sum(dmask)):d}/{nx*ny:d}")


if __name__ == "__main__":
    run_spr_reduce(sys.argv, verbose=True)
