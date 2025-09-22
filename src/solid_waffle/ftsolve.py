"""
Fourier-domain flat correlation prediction code for solid-waffle.
"""

import warnings

import numpy as np
from numpy.fft import fft2, ifft2


def center(arr):
    """
    Transforms kernel so that it looks as expected to the eye.

    Parameters
    ----------
    arr : np.array
        2D array kernel, square with odd side length.

    Returns
    -------
    np.array
        Returns version of kernel with (0,0) in center, y=-1 down, x=-1 left, etc.
        Centered arrays should *NOT* be used for calculations! For display only.

    See Also
    --------
    decenter : Inverse function.

    """

    size = (len(arr) + 1) // 2
    return np.roll(np.roll(arr, -size, axis=0), -size, axis=1)


def decenter(arr):
    """
    Transforms kernel from human-readable to numpy-readable.

    Only decentered arrays should be used for calculations.

    Parameters
    ----------
    arr : np.array
        2D array kernel, square with odd side length, human-readable.

    Returns
    -------
    np.array
        Decentered array.

    See Also
    --------
    center : Inverse function.

    """

    size = (len(arr) + 1) // 2
    return np.roll(np.roll(arr, size, axis=0), size, axis=1)


def flip(arr):
    """
    Transforms decentered a_i,j -> decentered a_-i,-j.

    The function is its own inverse.

    Parameters
    ----------
    arr : np.array
        2D array kernel, square with odd side length, human-readable.

    Returns
    -------
    np.array
        Flipped array.

    """

    arrc = center(arr)
    arrc_flipped = np.flip(arrc.flatten(), axis=0).reshape(arrc.shape)
    return decenter(arrc_flipped)


def pad_to_N(arr, N):
    """
    Zero-pads array out to size NxN (if arr is smaller than this).

    Parameters
    ----------
    arr : np.array
        2D input array, odd size, square, assumed to be centered.
    N : int
        Size to pad `arr` to. Must be odd. Does nothing if `arr` is already this big.

    Returns
    -------
    np.array
        The padded array.

    """

    if not arr.shape[0] > N:
        pad_size = (N - arr.shape[0]) // 2
        return np.pad(arr, pad_size, mode="constant")
    else:
        return arr


def p2kernel(cov, np2, N_integ=256):
    """
    Constructs a pairwise correlation kernel from a 2x2 covariance matrix.

    Parameters
    ----------
    cov : list or np.array of float
        Length 3; [Cxx, Cxy, Cyy].
    np2: int
        Kernel radius to generate.
    N_integ : int, optional
        Number of integration steps.

    Returns
    -------
    p2_output : np.array of float
        The p2 kernel, of shape (2*`np2`+1, 2*`np2`+1)
        so that ``p2_output[np2+j, np2+i]`` is the probability that if one charge
        generated in a flat field lands in (0,0), the other lands in (i,j).

    """

    use_extrule = True  # turn off only for de-bugging

    NN_integ = 2 * N_integ + 1  # dimension of integration region

    # Integration weights -- 2D array
    w = np.zeros(NN_integ)
    if use_extrule:
        if N_integ < 8:
            print("Error: N_integ in p2kernel must be at least 8.")
            exit()
        for i in range(1, N_integ + 1):
            w[i] = i / N_integ**2
        w[N_integ] *= 3.0 / 4.0
        w[1] *= 7.0 / 6.0
        w[N_integ - 1] *= 7.0 / 6.0
        w[2] *= 23.0 / 24.0
        w[N_integ - 2] *= 23.0 / 24.0
        w[N_integ + 1 :] = np.flip(w[:N_integ])
        w[N_integ + 1 :] = np.flip(w[:N_integ])
    else:
        for i in range(N_integ + 1):
            w[2 * N_integ - i] = w[i] = i / N_integ**2

    ww = np.outer(w, w)

    # get inverse covariance
    # note we actually want 2C
    detC = 4 * (cov[0] * cov[2] - cov[1] ** 2)
    iCov_xx = 2 * cov[2] / detC
    iCov_xy = -2 * cov[1] / detC
    iCov_yy = 2 * cov[0] / detC

    p2_output = np.zeros((2 * np2 + 1, 2 * np2 + 1))
    for j in range(-np2, 1):
        z2 = np.tile(np.linspace(j - 1, j + 1, NN_integ), (NN_integ, 1)).transpose()
        for i in range(-np2, np2 + 1):
            z1 = np.tile(np.linspace(i - 1, i + 1, 2 * N_integ + 1), (NN_integ, 1))
            integrand = np.exp(-0.5 * iCov_xx * z1**2 - iCov_xy * z1 * z2 - 0.5 * iCov_yy * z2**2)
            p2_output[np2 + j, np2 + i] = np.sum(integrand * ww)
    # use symmetry to not re-do a calculation
    for j in range(1, np2 + 1):
        p2_output[np2 + j, :] = np.flip(p2_output[np2 - j, :])

    p2_output /= 2 * np.pi * np.sqrt(detC)
    return p2_output


def op2_to_pars(op2, cmin=0.01):
    """
    Fits a quantum yield model to a Phi-kernel.

    Parameters
    ----------
    op2 : np.array
        The Phi-kernel, omega/(1+omega)*p2 (where omega is the 2-charge probability and p2 is the
        pairwise charge diffusion probability kernel). This is the combination that can be extracted from
        a correlation function.
    cmin : float, optional
        Minimum semi-minor axis of the covariance. (Prevents regularity problems.)

    Returns
    -------
    list
       Entries are [omega, cxx, cxy, cyy, change in last step, number of iterations].

    """

    cf = 1.0
    np2 = np.shape(op2)[0] // 2
    omegabar = np.sum(op2)
    cxx = cyy = 2 * cmin**2
    cxy = 0
    eps = 1
    j_iter = 0
    N = 96  # low resolution at first, upgrade when we get close
    this_np2 = 1
    this_op2 = op2[np2 - 1 : np2 + 2, np2 - 1 : np2 + 2]  # extract 3x3 for initial fitting
    dstep = 0.1
    while (eps > 1e-8 and j_iter < 256) or N < 256:
        # flag to go to full fitting
        if eps < 1e-5 or j_iter == 496:
            N = 256
            this_np2 = np2
            this_op2 = op2
        omegabar_old = omegabar
        cxx_old = cxx
        cxy_old = cxy
        cyy_old = cyy

        # update omegabar
        p2 = p2kernel([cxx, cxy, cyy], this_np2, N)
        err = this_op2 - omegabar * p2
        derr = -p2
        omegabar -= cf * np.sum(err * derr) / np.sum(derr**2)
        # update cxx
        # p2 doesn't need to be updated when we change omegabar
        err = this_op2 - omegabar * p2
        derr = -omegabar * (p2kernel([(1 + dstep) * cxx, cxy, cyy], this_np2, N) - p2) / (dstep * cxx)
        cxx -= cf * np.sum(err * derr) / np.sum(derr**2)
        cxxmin = cxy**2 / (cyy - cmin**2) + cmin**2
        if cxx < cxxmin:
            cxx = cxxmin * 1.000000001
        # update cyy
        p2 = p2kernel([cxx, cxy, cyy], this_np2, N)
        err = this_op2 - omegabar * p2
        derr = -omegabar * (p2kernel([cxx, cxy, (1 + dstep) * cyy], this_np2) - p2) / (dstep * cyy)
        cyy -= cf * np.sum(err * derr) / np.sum(derr**2)
        cyymin = cxy**2 / (cxx - cmin**2) + cmin**2
        if cyy < cyymin:
            cyy = cyymin * 1.000000001
        # update cxy
        p2 = p2kernel([cxx, cxy, cyy], this_np2, N)
        err = this_op2 - omegabar * p2
        dcxy = dstep * np.sqrt(cxx * cyy)
        cxylim = np.sqrt((cxx - cmin**2) * (cyy - cmin**2)) / 1.000000001
        if dcxy > np.abs(cxylim - np.abs(cxy)):
            dcxy = np.abs(cxylim - np.abs(cxy))
        derr = (
            -omegabar
            * (
                p2kernel([cxx, cxy + dcxy / 2, cyy], this_np2, N)
                - p2kernel([cxx, cxy - dcxy / 2, cyy], this_np2, N)
            )
            / dcxy
        )
        cxy -= cf * np.sum(err * derr) / np.sum(derr**2)
        if cxy < -cxylim:
            cxy = -cxylim
        if cxy > cxylim:
            cxy = cxylim

        j_iter += 1
        eps = np.max(
            np.abs(np.asarray([omegabar - omegabar_old, cxx - cxx_old, cxy - cxy_old, cyy - cyy_old]))
        )
        # lambda1 = (cxx + cyy - np.sqrt((cxx - cyy) ** 2 + (2 * cxy) ** 2)) / 2.0
        # print(omegabar, cxx, cxy, cyy, lambda1, eps, j_iter)

    if j_iter == 256:
        warnings.warn("op2_to_pars: failed to converge")
    omega = omegabar / (1 - omegabar)
    return [omega, cxx, cxy, cyy, eps, j_iter]


def p2kernel_test():
    """
    Test function for p2kernel.
    """

    for i in range(4):
        s = 0.4 / 2**i
        cov = [s**2, 0.5 * s**2, s**2]
        print(i, cov)
        print(op2_to_pars(0.05 * p2kernel(cov, 2)))
        cov = [1.1 * s**2, -0.8 * s**2, 0.9 * s**2]
        print(i, cov)
        print(op2_to_pars(0.05 * p2kernel(cov, 2)))
        print(op2_to_pars(0.025 * p2kernel(cov, 2) + 0.025 * p2kernel([s**2, 0, s**2], 2)))


def solve_corr(bfek, N, I_, g, betas, sigma_a, tslices, avals, avals_nl=[0, 0, 0], outsize=2):  # noqa: B006
    """
    Predicts the unequal-time correlation function C_{abcd}(Delta i, Delta j).

    Parameters
    ----------
    bfek : np.array
        Compound kernel [K^2 a+KK*] (assumed to be centered).
    N : int
        Size to use for boudary conditions (must be odd, larger is more accurate).
    I_ : float
        Current (elementary charges per pixel per frame).
    g : float
        Gain in e/DN.
    betas : np.array of float
        Array of classical non-linearity coefficients [beta_2...beta_n].
    sigma_a : float
        Sum of the BFE kernel, in e^-1.
    tslices : list of int
        List of time slices [ta, tb, tc, td].
    avals : list or tuple of float
        The alpha values for the linear IPC kernel [aV, aH, aD]. Dimensionless.
    avals_nl : list or tuple of float, optional
        The alpha values for NL-IPC kernel [aV_nl, aH_nl, aD_nl]. Units 1/e..
    outsize : int, optional
        The "radius" of output (so the BFE kernel has size (2*outsize+1, 2*outsize+1).

    Returns
    -------
    np.array
        An array of size (N, N) describing C_{abcd}(Delta i, Delta j) in "decentered" mode.

    """

    ta, tb, tc, td = tslices
    aV, aH, aD = avals
    aV_nl, aH_nl, aD_nl = avals_nl

    # convert betas to an array if it isn't already
    if not isinstance(betas, np.ndarray):
        betas = np.array([betas])

    if bfek.shape[1] != bfek.shape[0]:
        warnings.warn("WARNING: convolved BFE kernel (BFEK) not square.")

    assert N == 2 * (N // 2) + 1

    # Calculate K and K* from given alphas
    cent = slice(N // 2 - outsize, N // 2 + outsize + 1)

    k = decenter(pad_to_N(np.array([[aD, aV, aD], [aH, 1 - 4 * aD - 2 * aV - 2 * aH, aH], [aD, aV, aD]]), N))

    knl = decenter(
        pad_to_N(
            np.array(
                [
                    [aD_nl, aV_nl, aD_nl],
                    [aH_nl, -4 * aD_nl - 2 * aV_nl - 2 * aH_nl, aH_nl],
                    [aD_nl, aV_nl, aD_nl],
                ]
            ),
            N,
        )
    )

    # solve Fourier version for asq: F(BFEK) = Ksq^2*asq + Ksq*Knl_sq
    bfek = decenter(pad_to_N(bfek, N))
    ksq = fft2(k)
    knl_sq = fft2(knl)
    asq = (fft2(bfek) - ksq * knl_sq) / ksq**2
    a = ifft2(asq)

    a_flipped = flip(a)
    afsq = fft2(a_flipped)
    afsq_p = flip(afsq)

    ksq_p = flip(ksq)
    knl_sq_p = flip(knl_sq)

    # Calculate Cov(qsq(t),qsq(t')) (see eqn 38)
    qqs = []

    for ts in [(ta, tc), (ta, td), (tb, tc), (tb, td)]:
        t1 = min(ts)
        t = max(ts)

        # qq = (1/(afsq+afsq_p-sigma_a) * np.exp(I*afsq*(t-t1)) *
        #   (np.exp(I*(afsq+afsq_p)*t1)-np.exp(I*sigma_a*t1)))

        X = I_ * t1 * (afsq + afsq_p - sigma_a)
        qq = (
            (
                np.where(
                    np.abs(X) > 1e-4,
                    (np.exp(X) - 1) / np.where(np.abs(X) > 1e-5, X, X + 1),
                    1 + X / 2.0 + X**2 / 6.0 + X**3 / 24.0,
                )
            )
            * I_
            * t1
            * np.exp(I_ * afsq * (t - t1))
            * np.exp(I_ * sigma_a * t1)
        )
        if ts[1] < ts[0]:
            qq = np.conjugate(qq)

        qqs.append(qq)

    # Plug into correlation function (see eqn 51)
    csq_abcd = (
        1
        / g**2
        * (
            eval_cnl(betas, I_, ta)
            * eval_cnl(betas, I_, tc)
            * (ksq + knl_sq * I_ * ta)
            * (ksq_p + knl_sq_p * I_ * tc)
            * qqs[0]
            - eval_cnl(betas, I_, ta)
            * eval_cnl(betas, I_, td)
            * (ksq + knl_sq * I_ * ta)
            * (ksq_p + knl_sq_p * I_ * td)
            * qqs[1]
            - eval_cnl(betas, I_, tb)
            * eval_cnl(betas, I_, tc)
            * (ksq + knl_sq * I_ * tb)
            * (ksq_p + knl_sq_p * I_ * tc)
            * qqs[2]
            + eval_cnl(betas, I_, tb)
            * eval_cnl(betas, I_, td)
            * (ksq + knl_sq * I_ * tb)
            * (ksq_p + knl_sq_p * I_ * td)
            * qqs[3]
        )
    )

    return center(np.real(ifft2(csq_abcd)))[cent][:, cent]


def eval_cnl(betas, I_, t):
    """
    Evaluates the derivative of a non-linearity polynomial.

    Parameters
    ----------
    betas : np.array
        The coefficients, in order starting from 2nd order (beta_2), then 3rd order (beta_3), etc.
    I_ : float
        The current in electrons per pixel per frame.
    t : float
        The time in frames since reset.

    Returns
    -------
    float
        The derivative g*dS/dQ (where g is the gain in e/DN; S is the signal in DN; and Q is the
        charge in e).

    """

    nu = np.arange(2, len(betas) + 2)
    return 1 - np.sum(nu * betas * (I_ * t) ** (nu - 1))


def solve_corr_many(
    bfek,
    N,
    I_,
    g,
    betas,
    sigma_a,
    tslices,
    avals,
    avals_nl=[0, 0, 0],  # noqa: B006
    outsize=2,
):
    """
    Predicts a sequence of similar unequal-time correlation functions.

    Parameters
    ----------
    bfek : np.array
        Compound kernel [K^2 a+KK*] (assumed to be centered).
    N : int
        Size to use for boudary conditions (must be odd, larger is more accurate).
    I_ : float
        Current (elementary charges per pixel per frame).
    g : float
        Gain in e/DN.
    betas : np.array of float
        Array of classical non-linearity coefficients [beta_2...beta_n].
    sigma_a : float
        Sum of the BFE kernel, in e^-1.
    tslices : list of int
        List of time slices [ta, tb, tc, td, tn]. Should have tn>=1.
    avals : list or tuple of float
        The alpha values for the linear IPC kernel [aV, aH, aD]. Dimensionless.
    avals_nl : list or tuple of float, optional
        The alpha values for NL-IPC kernel [aV_nl, aH_nl, aD_nl]. Units 1/e..
    outsize : int, optional
        The "radius" of output (so the BFE kernel has size (2*outsize+1, 2*outsize+1).

    Returns
    -------
    np.array
        The mean correlation function, sum_{k=0}^{tn-1} C_{ta+k,tb+k,tc+k,td+k} / tn,
        as a shape (N, N) array, decentered.

    See Also
    --------
    solve_corr : equivalent version with tn=1.

    """

    this_t = tslices[:-1]
    tn = tslices[-1]
    cf = solve_corr(bfek, N, I_, g, betas, sigma_a, this_t, avals, avals_nl, outsize)
    for _ in range(tn - 1):
        for k in range(4):
            this_t[k] += 1
        cf += solve_corr(bfek, N, I_, g, betas, sigma_a, this_t, avals, avals_nl, outsize)
    cf /= tn + 0.0
    return cf


# Make a new function for visible wavelengths that returns the default
# behavior of solve_corr if omega = 0. Otherwise, it takes in p2 and omega != 0.
# input p2 is *centered*
def solve_corr_vis(
    bfek,
    N,
    I_,
    g,
    betas,
    sigma_a,
    tslices,
    avals,
    avals_nl=[0, 0, 0],  # noqa: B006
    outsize=2,
    omega=0,
    p2=0,
):
    """
    Predicts the unequal-time correlation function C_{abcd}(Delta i, Delta j) for visible light.

    Parameters
    ----------
    bfek : np.array
        Compound kernel [K^2 a+KK*] (assumed to be centered).
    N : int
        Size to use for boudary conditions (must be odd, larger is more accurate).
    I_ : float
        Current (elementary charges per pixel per frame).
    g : float
        Gain in e/DN.
    betas : np.array of float
        Array of classical non-linearity coefficients [beta_2...beta_n].
    sigma_a : float
        Sum of the BFE kernel, in e^-1.
    tslices : list of int
        List of time slices [ta, tb, tc, td].
    avals : list or tuple of float
        The alpha values for the linear IPC kernel [aV, aH, aD]. Dimensionless.
    avals_nl : list or tuple of float, optional
        The alpha values for NL-IPC kernel [aV_nl, aH_nl, aD_nl]. Units 1/e..
    outsize : int, optional
        The "radius" of output (so the BFE kernel has size (2*outsize+1, 2*outsize+1).
    omega : float, optional
        The probability of getting 2 charges.
    p2 : np.array, optional
        The pairwise separation probability for 2 charges generated at the same point in a flat field.

    Returns
    -------
    np.array
        An array of size (N, N) describing C_{abcd}(Delta i, Delta j) in "decentered" mode.

    See Also
    --------
    solve_corr : Similar but for IR flats.

    """

    if omega == 0:
        return solve_corr(bfek, N, I_, g, betas, sigma_a, tslices, avals, avals_nl, outsize)
    else:
        p2_sq = fft2(decenter(pad_to_N(p2, N)))
        ta, tb, tc, td = tslices
        aV, aH, aD = avals
        aV_nl, aH_nl, aD_nl = avals_nl

        # convert betas to an array if it isn't already
        if not isinstance(betas, np.ndarray):
            betas = np.array([betas])

        if bfek.shape[1] != bfek.shape[0]:
            warnings.warn("WARNING: convolved BFE kernel (BFEK) not square.")

        assert N == 2 * (N // 2) + 1

        # Calculate K and K* from given alphas
        cent = slice(N // 2 - outsize, N // 2 + outsize + 1)

        k = decenter(
            pad_to_N(np.array([[aD, aV, aD], [aH, 1 - 4 * aD - 2 * aV - 2 * aH, aH], [aD, aV, aD]]), N)
        )

        knl = decenter(
            pad_to_N(
                np.array(
                    [
                        [aD_nl, aV_nl, aD_nl],
                        [aH_nl, -4 * aD_nl - 2 * aV_nl - 2 * aH_nl, aH_nl],
                        [aD_nl, aV_nl, aD_nl],
                    ]
                ),
                N,
            )
        )

        # solve Fourier version for asq: F(BFEK) = Ksq^2*asq + Ksq*Knl_sq
        bfek = decenter(pad_to_N(bfek, N))
        ksq = fft2(k)
        knl_sq = fft2(knl)
        asq = (fft2(bfek) - ksq * knl_sq) / ksq**2
        a = ifft2(asq)

        a_flipped = flip(a)
        afsq = fft2(a_flipped)
        afsq_p = flip(afsq)

        ksq_p = flip(ksq)
        knl_sq_p = flip(knl_sq)

        # Calculate Cov(qsq(t),qsq(t')) (see eqn 38)
        qqs = []

        for ts in [(ta, tc), (ta, td), (tb, tc), (tb, td)]:
            t1 = min(ts)
            t = max(ts)

            # qq = (1/(afsq+afsq_p-sigma_a) * np.exp(I*afsq*(t-t1)) *
            #   (np.exp(I*(afsq+afsq_p)*t1)-np.exp(I*sigma_a*t1)))
            # Incorporate visible parameters into charge correlation function

            X = I_ * t1 * (afsq + afsq_p - sigma_a)
            qq = (
                ((2 * omega * p2_sq + 1 + omega) / (1 + omega))
                * (
                    np.where(
                        np.abs(X) > 1e-4,
                        (np.exp(X) - 1) / np.where(np.abs(X) > 1e-5, X, X + 1),
                        1 + X / 2.0 + X**2 / 6.0 + X**3 / 24.0,
                    )
                )
                * I_
                * t1
                * np.exp(I_ * afsq * (t - t1))
                * np.exp(I_ * sigma_a * t1)
            )
            if ts[1] < ts[0]:
                qq = np.conjugate(qq)

            qqs.append(qq)

        # Plug into correlation function (see eqn 51)
        csq_abcd = (
            1
            / g**2
            * (
                eval_cnl(betas, I_, ta)
                * eval_cnl(betas, I_, tc)
                * (ksq + knl_sq * I_ * ta)
                * (ksq_p + knl_sq_p * I_ * tc)
                * qqs[0]
                - eval_cnl(betas, I_, ta)
                * eval_cnl(betas, I_, td)
                * (ksq + knl_sq * I_ * ta)
                * (ksq_p + knl_sq_p * I_ * td)
                * qqs[1]
                - eval_cnl(betas, I_, tb)
                * eval_cnl(betas, I_, tc)
                * (ksq + knl_sq * I_ * tb)
                * (ksq_p + knl_sq_p * I_ * tc)
                * qqs[2]
                + eval_cnl(betas, I_, tb)
                * eval_cnl(betas, I_, td)
                * (ksq + knl_sq * I_ * tb)
                * (ksq_p + knl_sq_p * I_ * td)
                * qqs[3]
            )
        )

        return center(np.real(ifft2(csq_abcd)))[cent][:, cent]


# Like solve_corr_many but designed for handling charge diffusion
def solve_corr_vis_many(
    bfek,
    N,
    I_,
    g,
    betas,
    sigma_a,
    tslices,
    avals,
    avals_nl=[0, 0, 0],  # legacy interface # noqa: B006
    outsize=2,
    omega=0,
    p2=0,
):
    """
    Predicts a sequence of similar unequal-time correlation functions, for visible light.

    Parameters
    ----------
    bfek : np.array
        Compound kernel [K^2 a+KK*] (assumed to be centered).
    N : int
        Size to use for boudary conditions (must be odd, larger is more accurate).
    I_ : float
        Current (elementary charges per pixel per frame).
    g : float
        Gain in e/DN.
    betas : np.array of float
        Array of classical non-linearity coefficients [beta_2...beta_n].
    sigma_a : float
        Sum of the BFE kernel, in e^-1.
    tslices : list of int
        List of time slices [ta, tb, tc, td, tn]. Should have tn>=1.
    avals : list or tuple of float
        The alpha values for the linear IPC kernel [aV, aH, aD]. Dimensionless.
    avals_nl : list or tuple of float, optional
        The alpha values for NL-IPC kernel [aV_nl, aH_nl, aD_nl]. Units 1/e..
    outsize : int, optional
        The "radius" of output (so the BFE kernel has size (2*outsize+1, 2*outsize+1).
    omega : float, optional
        The probability of getting 2 charges.
    p2 : np.array, optional
        The pairwise separation probability for 2 charges generated at the same point in a flat field.

    Returns
    -------
    np.array
        The mean correlation function, sum_{k=0}^{tn-1} C_{ta+k,tb+k,tc+k,td+k} / tn,
        as a shape (N, N) array, decentered.

    See Also
    --------
    solve_corr_vis : equivalent version with tn=1.

    """

    this_t = tslices[:-1]
    tn = tslices[-1]
    cf = solve_corr_vis(bfek, N, I_, g, betas, sigma_a, this_t, avals, avals_nl, outsize, omega, p2)
    for _ in range(tn - 1):
        for k in range(4):
            this_t[k] += 1
        cf += solve_corr_vis(bfek, N, I_, g, betas, sigma_a, this_t, avals, avals_nl, outsize, omega, p2)
    cf /= tn + 0.0
    return cf


if __name__ == "__main__":
    """
   Test against configuration-space corrfn generated from known inputs/simulated flats.
   """

    N = 21
    I_ = 1487
    g = 2.06
    betas = np.array([1e-3, 5e-4])
    tslices = [3, 11, 13, 21]
    avals = [0, 0, 0]
    avals_nl = [0, 0, 0]

    test_bfek = 1.0e-6 * np.array(
        [
            [-0.01, 0.0020, -0.0210, -0.019, 0.028],
            [0.0040, 0.0490, 0.2480, 0.01, -0.0240],
            [-0.0170, 0.2990, -1.372, 0.2840, 0.0150],
            [0.0130, 0.0560, 0.2890, 0.0390, 0.02],
            [0.035, 0.0070, 0.0380, 0.0010, 0.026],
        ]
    )

    # test_bfek = np.load('/users/PCON0003/cond0088/Projects/detectors/solid-waffle/'
    #                     'testBFEK_flatsim_matcheddark_bfeonly18237sim_10files_sub20.npy')
    sigma_a = np.sum(test_bfek)

    # Test against BFEK values in run of test_run.py with input config.18237.sample1
    # N = 21
    # I_ = 1378
    # g = 2.26
    # beta = 5.98e-7
    # sigma_a = 0.0
    # tslices = [3, 11, 13, 21]
    # avals = [0.014,0.023,0]
    # avals_nl = [0,0,0]
    # test_bfek = np.load('test_bfek.npy')

    c_abcd = solve_corr(test_bfek, N, I_, g, betas, sigma_a, tslices, avals, avals_nl)
    print(c_abcd)
