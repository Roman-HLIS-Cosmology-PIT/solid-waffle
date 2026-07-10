"""Integrated noise & linearity test."""

import os

import numpy as np
from astropy.io import fits
from numpy.random import RandomState
from solid_waffle.noise_run import run_noise


def test_intlin(tmp_path):
    """Generate some sample files and fit them."""

    tmp_path = str(tmp_path)
    rng = RandomState(20160217)

    # parameters
    nch = 32  # number of channels
    nside = 4096  # side length of array
    nb = 4  # number of border reference pixels
    ndark = 4  # number of darks
    n_loflat = 3  # number of low-intensity flats
    n_hiflat = 3  # number of high-intensity flats
    nt = 11  # number of time slices
    g = 1.5  # gain in e/DN
    refsig = 28000  # reference level in DN
    amp_1f = 0.8
    amp_1f_uncorr = 0.3
    m_ref = 0.6
    c_ref = 0.2
    tfr = 3.16  # time per frame

    # make "truth" information
    _s = np.linspace(0, nside - 1, nside).astype(np.float32)
    x, y = np.meshgrid(_s, _s)
    del _s
    reset_level = (5000.0 + 20.0 * np.sin(np.pi * x / 128) + rng.normal(0.0, 10.0, (nside, nside))).astype(
        np.float32
    )  # DN
    read = rng.uniform(6.0, 9.0, (nside, nside)).astype(np.float32)  # DN, per read
    refread = rng.uniform(2.0, 3.0, (nside, nside // nch)).astype(np.float32)  # DN, per read, ref signal
    intensity = (
        5000.0 * np.exp(-(np.hypot(x - 600, y - 3100) ** 2) / 1.0e8) + rng.uniform(0.0, 25.0, (nside, nside))
    ).astype(np.float32)  # linearized DN per read
    dark = 1.0e-6 * 1.0e8 ** (rng.uniform(0.0, 1.0, (nside, nside)) ** 32)
    scale = (
        8.0e4 * np.exp((np.hypot(x - 2200, y - 3800) ** 2) / 1.0e8) + rng.uniform(0.0, 150.0, (nside, nside))
    ).astype(np.float32)  # scale for linearity curve

    # turn off sensitivity of reference pixels
    for x in [intensity, dark]:
        x[:nb, :] = x[-nb:, :] = 0.0
        x[:, :nb] = x[:, -nb:] = 0.0

    # the linearity mapping is:
    # S_obs = Reset + scale * (1 - exp(-S_lin / scale))
    # This way there is an analytic expectation for how it gets inverted.

    # Now build everything:
    cleanuplist = []
    for x in [("dark", ndark, 0.0), ("loflat", n_loflat, 0.1), ("hiflat", n_hiflat, 1.0)]:
        for sq in range(x[1]):
            # signal
            data = np.zeros((1, nt, nside, nside + nside // nch), dtype=np.uint16)
            data[0, :, :, nside:] = refsig
            q__g = rng.normal(0.0, 25.0, (nside, nside)).astype(np.float32)  # charge/gain in DN_lin
            for t in range(nt):
                q__g[:, :] += rng.poisson(intensity * x[2] + dark) / g  # add new signal
                signal = reset_level + scale * (1.0 - np.exp(-q__g / scale))

                # add some read and 1/f noise
                w = nside // nch
                noise = np.zeros((nside, nside + w), dtype=np.float32)
                noise[:, :nside] = read * rng.normal(0.0, 1.0, (nside, nside))
                noise[:, nside:] = refread * rng.normal(0.0, 1.0, (nside, w))
                for i in range(nch + 2):
                    # first make the 1/f noise realizations
                    le = nside * w
                    _x = np.zeros(2 * le, dtype=np.complex128)
                    _x[1:le] = (
                        rng.normal(0.0, 1.0, le - 1) + 1j * rng.normal(0.0, 1.0, le - 1)
                    ) / np.linspace(1, le - 1, le - 1) ** 0.5
                    s = np.fft.fft(_x)[:le].real.reshape((nside, w))
                    sflip = s[:, ::-1]

                    # uncorrelated across channels
                    if i < nch:
                        noise[:, i * w : (i + 1) * w] += amp_1f_uncorr * (sflip if i % 2 else s)
                    # correlated across channels
                    if i == nch:
                        for j in range(nch):
                            noise[:, j * w : (j + 1) * w] += amp_1f * (sflip if j % 2 else s)
                        noise[:, nside:] += amp_1f * m_ref * s
                    # reference output only
                    if i == nch + 1:
                        noise[:, nside:] += amp_1f * c_ref * s

                # now fill in both the science and reference channels
                data[0, t, :, :nside] = np.clip(signal + noise[:, :nside], 0, 65535).astype(np.uint16)
                data[0, t, :, nside:] = np.clip(refsig + noise[:, nside:], 0, 65535).astype(np.uint16)

            fn = tmp_path + f"/99999999_data_{x[0]:s}_{sq+1:03d}.fits"
            fitsfile = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data)])
            fitsfile[0].header["TGROUP"] = tfr
            fitsfile.writeto(fn, overwrite=True)
            print(
                "-->",
                fn,
                np.mean(data[0, :, :, :nside], axis=(-2, -1)),
                np.std(data[0, :, :, :nside], axis=(-2, -1)),
            )
            cleanuplist.append(f"data_{x[0]:s}_{sq+1:03d}")

    # ... and now we can run the noise script
    run_noise(
        {
            "-f": 6,
            "-i": tmp_path + "/99999999_data_dark_001.fits",
            "-o": tmp_path + "/noiseprop.fits",
            "-n": 4,
            "-t": 1,
            "-cd": 6.0,
            "-rh": 7,
            "-tn": 11,
            "-ro": True,
        },
        verbose=True,
    )

    # now let's look at the file
    with fits.open(tmp_path + "/noiseprop.fits") as nf:
        # bias/reset stuff
        bias = nf["NOISE"].data[nf["NOISE"].header["BIAS"]]
        assert 4910.0 < np.percentile(bias, 1.0) < 4980.0
        assert 5020.0 < np.percentile(bias, 99.0) < 5090.0
        assert 15.0 < np.median(nf["NOISE"].data[nf["NOISE"].header["RESET"]]) < 30.0
        assert 9.0 < np.median(nf["NOISE"].data[nf["NOISE"].header["CDS"]]) < 11.0
        assert 4.0 < np.median(nf["NOISE"].data[nf["NOISE"].header["TNOISE"]]) < 6.0

        # because of the small number of realizations, there is spurious low CDS / high total noise
        assert 500000 < np.count_nonzero(nf["NOISE"].data[nf["NOISE"].header["LCDSHTN"]]) < 1000000

        # darks
        assert 15.0 < np.percentile(dark / tfr, 99.9) < 20.0
        assert -1.0e-3 < np.percentile(dark / tfr, 0.1) < 1.0e-3

        nh = nf["NOISE"].header
        dk1 = nf["NOISE"].data[nh["DARK1"]]
        print("DARK1")
        assert 5.0 < np.percentile(dark / tfr - dk1, 99.9) < 10.0
        assert 1 < np.median(nf["NOISE"].data[nh["DARK1ERR"]]) < 2
        assert -7.0 < np.percentile(dark / tfr - dk1, 0.1) < -5.0
        print("DARK2")
        dk2 = nf["NOISE"].data[nh["DARK2"]]
        for x in [dk2, dark / tfr, dark / tfr - dk2]:
            print(np.percentile(x, 0.1), np.percentile(x, 99.9))
        assert 0.1 < np.median(nf["NOISE"].data[nh["DARK2ERR"]]) < 0.2
        assert 4.0 < np.percentile(dark / tfr - dk2, 99.9) < 7.0
        assert -0.8 < np.percentile(dark / tfr - dk2, 0.1) < -0.5

        # some assertions
        assert 3.15 < nh["TGROUP"] < 3.17
        assert 3.15 < nh["TDARK1"] < 3.17
        assert 31.5 < nh["TDARK2"] < 31.7
        assert 0.001 < nh["PCA0_AMP"] < 10
        assert 0.4 < nh["C_PINK"] < 1.0
        assert 0.3 < nh["U_PINK"] / nh["C_PINK"] < 0.8

        # power spectrum
        assert np.shape(nf["PS"].data) == (35, 552960)
        var = np.sum(nf["PS"].data, axis=1)
        assert np.all(var[:32] > 52.0)
        assert np.all(var[:32] < 55.0)
        assert 6.0 < var[32] < 10.0
        assert 20.0 < var[33] < 30.0
        assert 6.0 < var[34] < 10.0

        # check the histogram adds up
        assert np.sum(nf["NOISEHIST"].data) == nside**2

        # AMP33
        assert np.all(np.abs(nf["AMP33"].data[0] - refsig) < 5.0)
        assert np.shape(nf["AMP33"].data) == (2, 4096, 128)
        assert 0.45 < nf["AMP33"].header["M_PINK"] < 0.75
        assert nf["AMP33"].header["RU_PINK"] < 0.3
        assert 2.0 < np.median(nf["AMP33"].data[1]) < 3.0

    # cleanup
    for f in cleanuplist:
        print("<<", f)
        os.remove(tmp_path + "/99999999_" + f + ".fits")


# if __name__ == "__main__":
#     test_intlin("out")
