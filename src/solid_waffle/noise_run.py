"""
Noise analysis script.

Functions
---------
run_noise
    Runs the noise reduction.

"""

import re
import sys
import time
from os import path

import numpy as np
from astropy.io import fits
from scipy import linalg
from scipy.signal import convolve

from . import pyirc


def run_noise(arglist, verbose=False):
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
    The following arguments are permitted:

    * ``-f <fileformat>`` : input file format

    * ``-i <infile>`` : input file

    * ``-o <outfile>`` : output file

    * ``-n <nfile>`` : number of input noise files to use

    * ``-t <t_init>`` : starting time slice

    * ``-cd <cds_cut>`` : cut for low CDS pixels

    * ``-rh <rowh>`` : row overhead in samples

    * ``-tn <tnoiseframe>`` : number of total noise frames to include

    * ``-ro`` : turn on reference output

    """

    # if a dictionary is provided, convert it to a list
    if isinstance(arglist, dict):
        argdict = arglist
        arglist = [None]
        for k in argdict:
            if isinstance(argdict[k], bool):
                if argdict[k]:
                    arglist.append(k)
            else:
                arglist.append(k)
                arglist.append(str(argdict[k]))

    if verbose:
        print("argument list ->", arglist)

    # Get command-line arguments
    fileformat = 1
    nfile = 2
    infile = outfile = ""
    t_init = 2
    rowh = 7
    cds_cut = 10.0
    tnoiseframe = 55
    refout = False
    nch = 32
    for i in range(1, len(arglist)):
        if arglist[i] == "-f":
            fileformat = int(arglist[i + 1])
        if arglist[i] == "-i":
            infile = arglist[i + 1]
        if arglist[i] == "-o":
            outfile = arglist[i + 1]
        if arglist[i] == "-n":
            nfile = int(arglist[i + 1])
        if arglist[i] == "-t":
            t_init = int(arglist[i + 1])
        if arglist[i] == "-cd":
            cds_cut = float(arglist[i + 1])
        if arglist[i] == "-rh":
            rowh = int(arglist[i + 1])
        if arglist[i] == "-tn":
            tnoiseframe = int(arglist[i + 1])
        if arglist[i] == "-nch":
            nch = int(arglist[i + 1])
        if arglist[i] == "-ro":
            refout = True  # has reference output

    if infile == "":
        raise ValueError("Error: no infile. Use -i option.")

    if outfile == "":
        raise ValueError("Error: no outfile. Use -o option.")

    if verbose:
        print("Input:", infile, " +", nfile)
        print("  ... Format =", fileformat)
        print("Output:", outfile)

    # Get size of array
    N = pyirc.get_nside(fileformat)

    # Get array of new files:
    m = re.search(r"^(.+)/(\d\d\d\d\d\d\d\d)([^/]+)\_(\d+).fits", infile)
    if m:
        prefix = m.group(1)
        stamp = int(m.group(2))
        body = m.group(3)
    else:
        raise ValueError("Match failed.")
    filelist = []
    for k in range(nfile):
        thisname = prefix + f"/{stamp:d}" + body + f"_{k+1:03d}.fits"
        count = 0
        while not path.exists(thisname):
            count += 1
            thisname = prefix + f"/{stamp+count:d}" + body + f"_{k+1:03d}.fits"
            if count == 10000:
                raise ValueError("Failed to find file", k)
        filelist.append(thisname)

    # get group time
    with fits.open(filelist[0]) as G:
        tgroup = float(G[0].header["TGROUP"])
        if tgroup < 0.01:
            tgroup = float(G[0].header["TFRAME"])

    # Now get the median of the first stable frames
    nside = pyirc.get_nside(fileformat)
    ntslice = pyirc.get_num_slices(fileformat, filelist[k])
    firstframe = np.zeros((nfile, nside, nside), dtype=np.uint16)
    if verbose:
        print(np.shape(firstframe))
    for k in range(nfile):
        firstframe[k, :, :] = pyirc.load_segment(
            filelist[k], fileformat, [0, nside, 0, nside], [t_init], False
        )

    # Median and IQR
    medarray = np.percentile(firstframe, 50, axis=0, method="linear")
    iqrarray = np.percentile(firstframe, 75, axis=0, method="linear") - np.percentile(
        firstframe, 25, axis=0, method="linear"
    )

    del firstframe

    # Dark maps
    darkcds = np.zeros((nfile, nside, nside), dtype=np.float32)
    for k in range(nfile):
        darkcds[k, :, :] = (
            pyirc.load_segment(filelist[k], fileformat, [0, nside, 0, nside], [t_init], False).astype(
                np.float32
            )
            - pyirc.load_segment(filelist[k], fileformat, [0, nside, 0, nside], [t_init + 1], False).astype(
                np.float32
            )
        ) + ((k + 0.5) / nfile - 0.5)
    dark1 = np.median(darkcds, axis=0) / tgroup
    dark1err = (
        (np.percentile(darkcds, 75, axis=0) - np.percentile(darkcds, 25, axis=0))
        / tgroup
        * 0.9291
        / np.sqrt(nfile - 1)
    )
    # comment: the "sigma on the median" of a Gaussian distribution is 0.9291 IQR
    # (where IQR is the inter-quartile range)
    for k in range(nfile):
        darkcds[k, :, :] = (
            pyirc.load_segment(filelist[k], fileformat, [0, nside, 0, nside], [t_init], False).astype(
                np.float32
            )
            - pyirc.load_segment(filelist[k], fileformat, [0, nside, 0, nside], [ntslice], False).astype(
                np.float32
            )
        ) + ((k + 0.5) / nfile - 0.5)
    dark2 = np.median(darkcds, axis=0) / tgroup / (ntslice - t_init)
    dark2err = (
        (np.percentile(darkcds, 75, axis=0) - np.percentile(darkcds, 25, axis=0))
        / tgroup
        / (ntslice - t_init)
        * 0.9291
        / np.sqrt(nfile - 1)
    )
    del darkcds

    cdsnoise = np.zeros((nside, nside))
    nband = 32
    nstep = ntslice // 2 - 2
    if verbose:
        print("nstep =", nstep)
    for j in range(nband):
        bandData = np.zeros((nfile, nstep, nside // nband, nside), dtype=np.float32)
        ymin = j * nside // nband
        ymax = ymin + nside // nband
        for k in range(nfile):
            if verbose:
                print("CDS band", j, "file", k)
                sys.stdout.flush()
            for kb in range(nstep):
                bandData[k, kb, :, :] = (
                    pyirc.load_segment(
                        filelist[k], fileformat, [0, nside, ymin, ymax], [t_init + 2 * kb], False
                    ).astype(np.float32)
                    - pyirc.load_segment(
                        filelist[k], fileformat, [0, nside, ymin, ymax], [t_init + 2 * kb + 1], False
                    ).astype(np.float32)
                    + ((k + 0.5) / nfile + (kb + 0.5) / nstep - 1.0)
                )  # uniformly distribute to avoid quantization
        cdsnoise[ymin:ymax, :] = np.percentile(bandData, 75, axis=(0, 1), method="linear") - np.percentile(
            bandData, 25, axis=(0, 1), method="linear"
        )
        del bandData
    cdsnoise /= 1.34896

    # total noise
    tnoise = np.zeros((nside, nside))
    nband = 32
    for j in range(nband):
        bandData = np.zeros((nfile, nside // nband, nside))
        ymin = j * nside // nband
        ymax = ymin + nside // nband
        for k in range(nfile):
            if verbose:
                print("TN band", j, "file", k)
                sys.stdout.flush()
            for kb in range(tnoiseframe):
                bandData[k, :, :] = bandData[k, :, :] + (
                    pyirc.load_segment(filelist[k], fileformat, [0, nside, ymin, ymax], [t_init + kb], False)
                    + ((k + 0.5) / nfile - 0.5)
                ) * (kb - (tnoiseframe - 1) / 2.0)  # weight by how far from center
        tnoise[ymin:ymax, :] = np.percentile(bandData, 75, axis=0, method="linear") - np.percentile(
            bandData, 25, axis=0, method="linear"
        )
        del bandData
    tnoise /= (
        tnoiseframe * (tnoiseframe + 1.0) / 12.0
    )  # normalize so -1/2 --> +1/2 DN (change of 1) corresponds to one unit
    # this is sum (kb-(tnoiseframe-1)/2.) [ (kb-(tnoiseframe-1)/2.) / (tnoiseframe-1) ]
    tnoise /= 1.34896

    # ACN noise information
    nband = 32  # number of channels
    nfile2 = min(nfile, 16)
    ntslice = pyirc.get_num_slices(fileformat, filelist[k])
    nstep = ntslice // 2 - 2
    if verbose:
        print("nstep =", nstep)
    acnband = np.zeros(nband)
    aclip = 3.0 * 1.34896 * np.median(cdsnoise)
    for j in range(nband):
        if verbose:
            print("ACN, band", j)
        bandData = np.zeros((nfile2, nstep, nside, nside // nband))
        xmin = j * nside // nband
        xmax = xmin + nside // nband
        for k in range(nfile2):
            for kb in range(nstep):
                bandData[k, kb, :, :] = pyirc.load_segment(
                    filelist[k], fileformat, [xmin, xmax, 0, nside], [t_init + 2 * kb], False
                ).astype(np.float64) - pyirc.load_segment(
                    filelist[k], fileformat, [xmin, xmax, 0, nside], [t_init + 2 * kb + 1], False
                ).astype(np.float64)
        for kb in range(nstep):
            bref = np.median(bandData[:, kb, :, :], axis=0)
            for k in range(nfile2):
                bandData[k, kb, :, :] -= bref
        # clip
        bandData = np.where(bandData > aclip, aclip, bandData)
        bandData = np.where(bandData < -aclip, -aclip, bandData)

        # get power spectrum
        bandPower = np.absolute(np.fft.fft(bandData, axis=3, norm="ortho")) ** 2
        if verbose:
            print("shape of bandPower:", np.shape(bandPower))
        bandPowerDiff = bandPower[:, :, :, 64] - np.mean(bandPower[:, :, :, 60:64], axis=3)
        if verbose:
            print("shape of bandPowerDiff:", np.shape(bandPowerDiff))
        # for i in range(N//nch): print('{:3d} {:12.6f}'.format(i, np.mean(bandPower[:,:,:,i])))
        acnband[j] = (
            np.mean(bandPowerDiff) / (N // nch) / 2.0
        )  # divide by length; then by 2 to get per sample
        if verbose:
            print("band", j, ": ACN =", acnband[j], "DN**2")
            sys.stdout.flush()
        del bandData
    acnnoise = np.median(acnband) * nfile2 / (nfile2 - 1.0)
    acnnoise = 0 if acnnoise < 0 else np.sqrt(acnnoise)
    if verbose:
        print("overall ACN", acnnoise)

    # 1/f noise information
    nband = 32  # number of channels
    extra_power_spectra = 1
    if refout:
        extra_power_spectra = 3  # how many additional power spectra we compute

    ntslice = pyirc.get_num_slices(fileformat, filelist[k])
    nstep = ntslice // 2 - 2
    if verbose:
        print("nstep =", nstep)
    acnband = np.zeros(nband)
    aclip = 3.0 * 1.34896 * np.median(cdsnoise)
    avebandData = np.zeros((nfile2, nstep, nside, nside // nband))
    slength = N * (N // nband + rowh)
    PS = np.zeros((nband + extra_power_spectra, slength))
    for j in range(nband + extra_power_spectra):
        if verbose:
            print("1/f, band", j)
            sys.stdout.flush()
        bandData = np.zeros((nfile2, nstep, nside, nside // nband))
        if j < nband:
            xmin = j * nside // nband
            xmax = xmin + nside // nband
            for k in range(nfile2):
                for kb in range(nstep):
                    bandData[k, kb, :, :] = pyirc.load_segment(
                        filelist[k], fileformat, [xmin, xmax, 0, nside], [t_init + 2 * kb], False
                    ).astype(np.float64) - pyirc.load_segment(
                        filelist[k], fileformat, [xmin, xmax, 0, nside], [t_init + 2 * kb + 1], False
                    ).astype(np.float64)
            for kb in range(nstep):
                bref = np.median(bandData[:, kb, :, :], axis=0)
                for k in range(nfile2):
                    bandData[k, kb, :, :] -= bref
            # clip
            bandData = np.where(bandData > aclip, aclip, bandData)
            bandData = np.where(bandData < -aclip, -aclip, bandData)

            # flip even channels
            if j % 2 == 1:
                bandData = np.flip(bandData, axis=3)

            avebandData += bandData / nband
        elif j == nband:
            bandData[:, :, :, :] = avebandData
        else:
            # now we are working with the reference output.
            # get reference +/- average of the other channels
            xmin = nside
            xmax = xmin + nside // nband
            for k in range(nfile2):
                for kb in range(nstep):
                    bandData[k, kb, :, :] = pyirc.load_segment(
                        filelist[k], fileformat, [xmin, xmax, 0, nside], [t_init + 2 * kb], False
                    ).astype(np.float64) - pyirc.load_segment(
                        filelist[k], fileformat, [xmin, xmax, 0, nside], [t_init + 2 * kb + 1], False
                    ).astype(np.float64)
            for kb in range(nstep):
                bref = np.median(bandData[:, kb, :, :], axis=0)
                for k in range(nfile2):
                    bandData[k, kb, :, :] -= bref
            # clip
            bandData = np.clip(bandData, -aclip, aclip)

            if j == nband + 1:
                bandData[:, :, :, :] += avebandData
            else:
                bandData[:, :, :, :] -= avebandData

        # get power spectrum
        mySeries = np.zeros((nfile2, nstep, slength))
        for r in range(N):
            mySeries[:, :, r * (N // nband + rowh) : r * (N // nband + rowh) + N // nband] = bandData[
                :, :, r, :
            ]

        bandPower = np.mean(np.absolute(np.fft.fft(mySeries, axis=2, norm="ortho")) ** 2, axis=(0, 1))
        bandPower *= (
            (1.0 + rowh * nband / N) / slength / 2.0
        )  # divide by length; then by 2 to get per sample instead of CDS
        if verbose:
            print("shape of bandPower:", np.shape(bandPower))
        PS[j, :] = bandPower

        del bandData
        del bandPower
        del mySeries

    if verbose:
        for i in range(1024):
            print(f"{i:6d} {PS[nband,i]:11.5E} {np.median(PS[:nband,i])-PS[nband,i]:12.5E}")

    ncut = N
    c_pink = np.sum(PS[nband, 1 : ncut + 1]) - ncut * np.median(PS[nband, :])
    c_pink = 0.0 if c_pink < 0 else np.sqrt(c_pink)
    u_pink = np.sum(PS[:nband, 1 : ncut + 1]) / nband - ncut * np.median(PS[:nband, :]) - c_pink
    u_pink = 0.0 if u_pink < 0 else np.sqrt(u_pink)

    # reference output information
    if refout:
        PS_plus = np.sum(PS[nband + 1, 1 : ncut + 1]) - ncut * np.median(PS[nband + 1, :])
        PS_minus = np.sum(PS[nband + 2, 1 : ncut + 1]) - ncut * np.median(PS[nband + 2, :])
        # now need to get:
        # r_pink = amplitude of reference output 1/f noise
        # rho = correlation coefficient with common noise
        r_pink = np.sqrt(np.clip(PS_plus + PS_minus - 2 * c_pink**2, 0, None) / 2.0)
        rho = (PS_plus - PS_minus) / (4 * r_pink * c_pink + 1e-24)  # prevent division by zero
        rho = np.clip(rho, -1.0, 1.0)
        #
        # and now the slope m (factor by which to multiply the correlated noise
        # to get what we see in the reference output)
        m_pink = r_pink * rho / (c_pink + 1e-24)
        ru_pink = r_pink * (1 - rho**2)

    # normalize to variance (DN^2) per ln frequency
    scale = np.sqrt(2 / np.log(N))
    u_pink *= scale
    c_pink *= scale
    if verbose:
        print("c_pink =", c_pink, "u_pink =", u_pink)
    if refout:
        ru_pink *= scale
        if verbose:
            print("m_pink = ", m_pink, "ru_pink =", ru_pink)
            print(">>", c_pink**2, PS_plus, PS_minus)
            for jj in range(np.shape(PS)[0]):
                print(np.sum(PS[jj, 1 : ncut + 1]) - ncut * np.median(PS[jj, :]))

    # Get PCAs
    # split into this many channels
    #
    cdsFL = np.zeros((nfile, nside, nside), dtype=np.float32)
    if verbose:
        print(np.shape(cdsFL))
    for k in range(nfile):
        if verbose:
            print("build CDS ", k)
            sys.stdout.flush()
        cdsFL[k, :, :] = pyirc.load_segment(
            filelist[k], fileformat, [0, nside, 0, nside], [t_init], False
        ).astype(np.float32) - pyirc.load_segment(
            filelist[k], fileformat, [0, nside, 0, nside], [ntslice - 2], False
        ).astype(np.float32)
        # reference pixel removal -- first sides, then top
        for j in range(nside):
            cdsFL[k, j, :] -= np.median(np.concatenate((cdsFL[k, j, :4], cdsFL[k, j, -4:])))
        for i in range(nch):
            xmin = i * nside // nch
            xmax = xmin + nside // nch
            cdsFL[k, :, xmin:xmax] -= np.median(cdsFL[k, -4:, xmin:xmax])
    med_cdsFL = np.percentile(cdsFL, 50, axis=0, method="linear").astype(np.float32)
    iqr_cdsFL = (
        np.percentile(cdsFL, 75, axis=0, method="linear") - np.percentile(cdsFL, 25, axis=0, method="linear")
    ).astype(np.float32)
    iqr_cdsFL_med = np.median(iqr_cdsFL)
    niqr_clip = 3.0
    for k in range(nfile):
        cdsFL[k, :, :] -= med_cdsFL
        cdsFL[k, :, :] = np.where(
            cdsFL[k, :, :] > niqr_clip * iqr_cdsFL_med, niqr_clip * iqr_cdsFL_med, cdsFL[k, :, :]
        )
        cdsFL[k, :, :] = np.where(
            cdsFL[k, :, :] < -niqr_clip * iqr_cdsFL_med, -niqr_clip * iqr_cdsFL_med, cdsFL[k, :, :]
        )
    #
    # get covariance
    C = np.zeros((nside, nside))
    for i in range(nfile):
        if verbose:
            print("cdsFL[i,:,:] =", cdsFL[i, :, :])
        for j in range(nfile):
            C[j, i] = np.sum(cdsFL[j, :, :].astype(np.float64) * cdsFL[i, :, :].astype(np.float64))
    C /= float(nfile - 1)
    w, v = linalg.eigh(C)
    max_eigenval = w[-1]
    pca0_eigenvec = v[:, -1]
    if verbose:
        print("max eigenvalue =", max_eigenval)
        print("eigenvec =", pca0_eigenvec)
        print("normalization of eigenvec =", np.sum(pca0_eigenvec**2))
    PCA0 = np.zeros((nside, nside))
    for k in range(nfile):
        PCA0 += pca0_eigenvec[k] * cdsFL[k, :, :].astype(np.float64)
    PCA0 /= np.sqrt(np.mean(PCA0**2))
    #
    pca0_stdev = np.sqrt(max_eigenval) / float(nside)
    if verbose:
        print("stdev of pca0 =", pca0_stdev)
    # clean up memory
    del med_cdsFL
    del iqr_cdsFL
    del cdsFL

    # Generate output cube
    outcube = np.zeros((10, nside, nside), dtype=np.float32)
    outcube[0, :, :] = 65535 - medarray
    outcube[1, :, :] = iqrarray / 1.34896
    outcube[2, :, :] = cdsnoise
    outcube[3, :, :] = PCA0
    outcube[4, :, :] = dark1
    outcube[5, :, :] = dark2
    outcube[6, :, :] = tnoise
    # low CDS, high total noise pixels
    outcube[7, :, :] = np.where(np.logical_and(cdsnoise < cds_cut, tnoise > np.median(tnoise)), 1, 0)
    outcube[7, :4, :] = 0.0
    outcube[7, -4:, :] = 0.0
    outcube[7, :, :4] = 0.0
    outcube[7, :, -4:] = 0.0
    outcube[8, :, :] = dark1err
    outcube[9, :, :] = dark2err
    # label the slices
    outcubeslices = [
        "BIAS",
        "RESET",
        "CDS",
        "PCA0",
        "DARK1",
        "DARK2",
        "TNOISE",
        "LCDSHTN",
        "DARK1ERR",
        "DARK2ERR",
    ]

    # array information on low CDS, high total noise pixels
    badmap = outcube[7, 4:-4, 4:-4]
    nbad1 = np.sum(badmap > 0.5)
    nbad3 = np.sum(convolve(badmap, np.ones((3, 3)), mode="same") > 0.5)
    nbad5 = np.sum(convolve(badmap, np.ones((5, 5)), mode="same") > 0.5)

    primary_hdu = fits.PrimaryHDU()
    out_hdu = fits.ImageHDU(outcube)
    out_hdu.header["EXTNAME"] = "NOISE"
    out_hdu.header["TGROUP"] = tgroup
    out_hdu.header["TDARK1"] = tgroup
    out_hdu.header["TDARK2"] = tgroup * (ntslice - t_init)
    out_hdu.header["ACN"] = acnnoise
    out_hdu.header["C_PINK"] = c_pink
    out_hdu.header["U_PINK"] = u_pink
    out_hdu.header["PCA0_AMP"] = pca0_stdev
    for k in range(nfile):
        out_hdu.header[f"NR_MF{k+1:03d}"] = filelist[k].strip()
    # median information
    out_hdu.header["CDS_CUT"] = (cds_cut, "DN")
    out_hdu.header["CDS_MED"] = (np.median(cdsnoise), "DN")
    out_hdu.header["TOT_MED"] = (np.median(tnoise), "DN")
    out_hdu.header["LCHTN1"] = (nbad1, "low CDS, high total noise pixels")
    out_hdu.header["LCHTN3"] = (nbad3, "low CDS, high total noise pixels, 3x3 expanded")
    out_hdu.header["LCHTN5"] = (nbad5, "low CDS, high total noise pixels, 5x5 expanded")
    out_hdu.header["NR_DATE"] = time.asctime(time.localtime(time.time()))

    for idx in range(np.shape(outcube)[0]):
        out_hdu.header[outcubeslices[idx]] = (idx, "Index of " + outcubeslices[idx])

    ps_hdu = fits.ImageHDU(PS.astype(np.float32))
    ps_hdu.header["EXTNAME"] = "PS"

    # CDS vs total noise 2D plot
    d_np = 0.25
    N_np = 80
    ctnplot = np.zeros((N_np, N_np))
    for j in range(N_np):
        tnmin = j * d_np
        tnmax = tnmin + d_np
        if j == 0:
            tnmin = -1e49
        if j == N_np - 1:
            tnmax = 1e49
        for i in range(N_np):
            cnmin = i * d_np
            cnmax = cnmin + d_np
            if i == 0:
                cnmin = -1e49
            if i == N_np - 1:
                cnmax = 1e49
            ctnplot[j, i] = np.count_nonzero(
                np.logical_and(
                    np.logical_and(tnoise >= tnmin, tnoise < tnmax),
                    np.logical_and(cdsnoise >= cnmin, cdsnoise < cnmax),
                )
            )
    plot_hdu = fits.ImageHDU(ctnplot)
    plot_hdu.header["EXTNAME"] = "NOISEHIST"
    plot_hdu.header["INFO"] = "2D plot, CDS vs total noise"
    plot_hdu.header["XDEF"] = ("CDS noise", "DN")
    plot_hdu.header["YDEF"] = ("total noise", "DN")
    plot_hdu.header["DNOISE"] = (d_np, "DN")
    plot_hdu.header["MAXNOISE"] = (d_np * N_np, "DN")

    outhdus = [primary_hdu, out_hdu, ps_hdu, plot_hdu]

    # extra HDU if we asked for the reference output
    if refout:
        extra = np.zeros((2, nside, nside // nch))

        # median
        xmin = nside
        xmax = nside + nside // nch
        nstep = ntslice - 2
        bandData = np.zeros((nfile2, nstep, nside, nside // nch), dtype=np.float32)
        for k in range(nfile2):
            for kb in range(nstep):
                bandData[k, kb, :, :] = pyirc.load_segment(
                    filelist[k], fileformat, [xmin, xmax, 0, nside], [t_init + kb], False
                ).astype(np.float64)
        extra[0, :, :] = 65535 - np.median(bandData, axis=(0, 1))

        # noise
        nstep = ntslice // 2 - 2
        bandData = np.zeros((nfile, nstep, nside, nside // nband), dtype=np.float32)
        for k in range(nfile):
            for kb in range(nstep):
                bandData[k, kb, :, :] = (
                    pyirc.load_segment(
                        filelist[k], fileformat, [xmin, xmax, 0, nside], [t_init + 2 * kb], False
                    ).astype(np.float32)
                    - pyirc.load_segment(
                        filelist[k], fileformat, [xmin, xmax, 0, nside], [t_init + 2 * kb + 1], False
                    ).astype(np.float32)
                    + ((k + 0.5) / nfile + (kb + 0.5) / nstep - 1.0)
                )  # uniformly distribute to avoid quantization
        extra[1, :, :] = (
            (
                np.percentile(bandData, 75, axis=(0, 1), method="linear")
                - np.percentile(bandData, 25, axis=(0, 1), method="linear")
            )
            / 1.34896
            / np.sqrt(2.0)
        )
        # effective read noise of the reference channel in DN rms

        amp33_hdu = fits.ImageHDU(extra.astype(np.float32))
        amp33_hdu.header["M_PINK"] = m_pink
        amp33_hdu.header["RU_PINK"] = ru_pink
        amp33_hdu.header["EXTNAME"] = "AMP33"
        outhdus.append(amp33_hdu)

    hdul = fits.HDUList(outhdus)
    hdul.writeto(outfile, overwrite=True)


if __name__ == "__main__":
    run_noise(sys.argv, verbose=True)
