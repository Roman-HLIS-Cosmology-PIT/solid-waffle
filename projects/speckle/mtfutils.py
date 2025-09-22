import numpy


# functions to fit two triangles to a power spectrum
# (array, 0 element corresponding to 0 wavenumber)
#
# as a function of offset (cpp0) and width (cppw)
#
# output = best amplitude1, best amplitude 2, SSR
def get_ps_err(ps_in, cpp0, cppw):
    """
    Fit two triangles to a power spectrum.

    Parameters
    ----------
    ps_in : np.array
        The power spectrum array (indexed starting at 0 cycles/pixel).
    cpp0 : float
        Offset of the triangle peak in cycles/pixel.
    cppw : float
        The half-width of the triangle peak in cycles/pixel.

    Returns
    -------
    bestamp1 : float
        Amplitude of the peak at wavenumber 0.
    bestamp2 : float
        Amplitude of the peak at `cpp0`.
    ssr : float
        Sum of the squared residuals.

    """

    cpp0 += 1.0
    n = numpy.size(ps_in)
    ps_inS = ps_in - numpy.median(ps_in)
    template1 = numpy.zeros((n,))
    template2 = numpy.zeros((n,))
    dx = cppw * n
    for j in range(1 + int(dx)):
        template1[j] = template1[-j] = 1.0 - j / dx
    xc = cpp0 * n
    xmin = xc - dx
    xmax = xc + dx
    for j in range(int(xmin), int(xmax) + 1):
        amp = 0.5 * (1.0 - numpy.abs(j - xc) / dx)
        template2[j % n] += amp
        template2[-(j % n)] += amp
    F11 = numpy.sum(template1[1:] ** 2)
    F12 = numpy.sum(template1[1:] * template2[1:])
    F22 = numpy.sum(template2[1:] ** 2)
    S1 = numpy.sum(ps_inS[1:] * template1[1:])
    S2 = numpy.sum(ps_inS[1:] * template2[1:])
    bestamp1 = (F22 * S1 - F12 * S2) / (F11 * F22 - F12**2)
    bestamp2 = (F11 * S2 - F12 * S1) / (F11 * F22 - F12**2)
    ssr = numpy.sum((ps_inS[1:] - bestamp1 * template1[1:] - bestamp2 * template2[1:]) ** 2)
    return bestamp1, bestamp2, ssr


# return "best" power spectrum from triangel fit
# and normalized residuals:
#
# cpp0 best, cppw best, amplitude 1, amplitude 2, SSR/SS
def get_triangle_from_ps(ps_in, cppguess):
    """
    Fit two triangles to a power spectrum.

    Parameters
    ----------
    ps_in : np.array
        The power spectrum array (indexed starting at 0 cycles/pixel).
    cppguess : float
        Offset of the triangle peak in cycles/pixel.

    Returns
    -------
    best_cpp0 : float
        The best-fit wavenumber of the fringe pattern in cycles/pixel.
    best_cppw : float
        The best-fit width of the triangle peak in cycles/pixel.
    a1 : float
        The best-fit amplitude of the peak at wavenumber 0.
    a2 : float
        The best-fit amplitude of the peak at `best_cpp0`.
    ssr_norm : float
        Sum of the squared residuals, normalized by the input power spectrum.

    """

    gr = 33
    sp = 8

    nn = numpy.size(ps_in)
    ssr_array = numpy.zeros((gr, gr))
    cpp0 = numpy.linspace(1.05 / nn, 0.5, gr)
    if cppguess > 0.4:
        cpp0 = numpy.linspace(0.2, 0.5, gr)
    if cppguess > 0.5:
        cpp0 = numpy.linspace(0.5, 0.8, gr)
    cppw = numpy.linspace(1.05 / nn, 0.255, gr)
    for _ in range(24):
        for i in range(gr):
            for j in range(gr):
                a1, a2, ssr = get_ps_err(ps_in, cpp0[i], cppw[j])
                ssr_array[i, j] = ssr
        ind = numpy.unravel_index(numpy.argmin(ssr_array), numpy.shape(ssr_array))
        ii = ind[0]
        jj = ind[1]
        best_cpp0 = cpp0[ii]
        best_cppw = cppw[jj]

        # make next grid
        if ii < sp:
            ii = sp
        if jj < sp:
            jj = sp
        if ii >= gr - sp:
            ii = gr - sp - 1
        if jj >= gr - sp:
            jj = gr - sp - 1
        cpp0 = numpy.linspace(cpp0[ii - sp], cpp0[ii + sp], gr)
        cppw = numpy.linspace(cppw[jj - sp], cppw[jj + sp], gr)

    # flip if guess on other side
    a1, a2, ssr = get_ps_err(ps_in, best_cpp0, best_cppw)
    ssr_norm = ssr / numpy.sum(ps_in[1:] ** 2)

    return best_cpp0, best_cppw, a1, a2, ssr_norm
