"""
Provide functions for creating lineshapes suitable for plotting.

For non-DNMR calculations, inputs are lists of (frequency, intensity) tuples,
which then have Lorentzian distributions applied to them.

For DNMR calculations, the lineshapes are directly computed. Currently,
non-quantum mechanical formulas for two uncoupled spins and for two coupled
spins are used.
"""
import numpy as np

from .nmrmath import dnmr_AB, d2s_func


def lorentz(v, v0, I, w):
    """
    A lorentz function that takes linewidth at half intensity (w) as a
    parameter.

    When `v` = `v0`, and `w` = 0.5 (Hz), the function returns intensity I.

    Arguments
    ---------
    v : float
        The frequency (x coordinate) at which to evaluate intensity (y
        coordinate).
    v0 : float
        The center of the distribution.
    I : float
        the relative intensity of the signal
    w : float
        the peak width at half maximum intensity

    Returns
    -------
    float
        the intensity (y coordinate) for the Lorentzian distribution
        evaluated at frequency `v`.
    """
    # Adding a height scaling factor so that peak intensities are lowered as
    # they are more broad. If I is the intensity with a default w of 0.5 Hz:
    scaling_factor = 0.5 / w  # i.e. a 1 Hz wide peak will be half as high
    return scaling_factor * I * (
            (0.5 * w) ** 2 / ((0.5 * w) ** 2 + (v - v0) ** 2))


def add_signals(linspace, peaklist, w):
    """
    Given a numpy linspace, a spectrum as a list of (frequency, intensity)
    tuples, and a linewidth, returns an array of y coordinates for the
    total line shape.

    Arguments
    ---------
    linspace : array-like
        normally a numpy.linspace of x coordinates corresponding to frequency
        in Hz.
    peaklist : [(float, float)...]
        a list of (frequency, intensity) tuples.
    w : float
        peak width at half maximum intensity.

    Returns
    -------
    [float...]
        an array of y coordinates corresponding to intensity.
    """
    result = lorentz(linspace, peaklist[0][0], peaklist[0][1], w)
    for v, i in peaklist[1:]:
        result += lorentz(linspace, v, i, w)
    return result


def nmrplot(spectrum, y=1):
    """
    A no-frills routine that plots spectral simulation data.

    Arguments
    ---------
    spectrum : [(float, float)...]
        a list of (frequency, intensity) tuples.
    y : float
        maximum intensity for the plot.
    """
    """Oddball function. This is really a function for an application, 
    not a library. TODO: revise or eliminate."""
    import matplotlib.pyplot as plt

    spectrum.sort()  # Could become costly with larger spectra
    l_limit = spectrum[0][0] - 50
    r_limit = spectrum[-1][0] + 50
    x = np.linspace(l_limit, r_limit, 800)
    plt.ylim(-0.1, y)
    plt.gca().invert_xaxis()  # reverses the x axis
    # noinspection PyTypeChecker
    plt.plot(x, add_signals(x, spectrum, w=1))

    plt.show()
    return


def tkplot(spectrum, w=0.5):
    """Generate linspaces of x and y coordinates suitable for plotting on a
    matplotlib tkinter canvas.

    TODO: this is not tk specific. rename?

    Arguments
    ---------
    spectrum : [(float, float)...]
        a list of (frequency, intensity) tuples.
    w : float
        peak width at half height

    Returns
    -------
    (ndarray, ndarray)
        a tuple of numpy.ndarrays for x and y coordinates
    """
    spectrum.sort()
    r_limit = spectrum[-1][0] + 50
    l_limit = spectrum[0][0] - 50
    x = np.linspace(l_limit, r_limit, 2400)
    y = add_signals(x, spectrum, w)
    return x, y


def tkplot_nmrmint(spectrum, w=0.5, spectrometer_frequency=300):
    """Generate linspaces of x and y coordinates suitable for plotting on a
    matplotlib tkinter canvas.

    Hard-coding a -1 to 15 ppm linspace, with resolution such that a 1 GHz
    spectrometer has 10 points per Hz.
    :param spectrum: A list of (frequency, intensity) tuples
    :param w: peak width at half height
    :param spectrometer_frequency: the frequency of the spectrometer (i.e
    frequency in MHz that 1H nuclei resonate at)
    :return: a tuple of x and y numpy.ndarrays
    """
    """This is a port of nmrmint's version of tkplot.
    TODO: think about what nmrmint should offer to all users.
    Should eliminate this from library.
    """
    x = np.linspace(-1 * spectrometer_frequency,
                    15 * spectrometer_frequency,
                    160000)  # 0.01 Hz resolution on 1 GHz spectrometer
    y = add_signals(x, spectrum, w)
    return x, y


def dnmrplot_2spin(va, vb, ka, Wa, Wb, pa):
    """Create a lineshape for the DNMR spectrum of two uncoupled nuclei
    undergoing exchange.

    Arguments
    ---------
    va : float
        the frequency of nucleus 'a' at the slow exchange limit
    vb : float
        the frequency of nucleus 'b' at the slow exchange limit
    ka : float
        the rate of nuclear exchange
    Wa : float
        the width at half heigh of the signal for nucleus a (at the slow
        exchange limit).
    Wb : float
        the width at half height of the signal for nucleus b (at the slow
        exchange limit).
    pa : float
        the fraction of the population in state a (vs. state b)

    Returns
    -------
    ([float...], [float...])
        a tuple of numpy arrays for frequencies (x coordinate) and
        corresponding intensities (y coordinate). Hard-coded for 800 data
        points and a frequency range from vb-50 to va+50.

    TODO: throughout nmrtools there is hard-coding of defaults, based on
    needs of applications. Consider the needs of other users and make more
    universal.
    """
    """A decision needs to be made on the final function to apply along the 
    linspace."""
    if vb > va:
        va, vb = vb, va
        Wa, Wb = Wb, Wa
        pa = 1 - pa
    l_limit = vb - 50
    r_limit = va + 50
    x = np.linspace(l_limit, r_limit, 800)
    # y = dnmr_2spin(x, va, vb, ka, Wa, Wb, pa)

    # OR:

    dfunc = d2s_func(va, vb, ka, Wa, Wb, pa)
    y = dfunc(x)

    # OR:
    # y = reich(x, va, vb, ka, Wa, Wb, pa)

    return x, y


def dnmrplot_AB(v1, v2, J, k, W):
    """
    Create a lineshape for the DNMR spectrum of two uncoupled nuclei
    undergoing exchange.

    Arguments
    ---------
    v1 : float
        the frequency of nucleus 'a' at the slow exchange limit.
    v2 : float
        the frequency of nucleus 'b' at the slow exchange limit.
    J : float
        the coupling constant between nuclei a and b.
    k : float
        the rate of two-site exchange of nuclei a and b.
    W : float
        the line width at the slow exchange limit.

    Returns
    -------
    (numpyp.array, numpy.array)
        a tuple of numpy arrays for frequencies (x coordinate) and
        corresponding intensities (y coordinate).
        Hard-coded for 800 data points and a frequency range
        from `vb` - 50 to `va` + 50.
    """
    if v2 > v1:
        v1, v2 = v2, v1
    l_limit = v2 - 50
    r_limit = v1 + 50
    x = np.linspace(l_limit, r_limit, 800)
    y = dnmr_AB(x, v1, v2, J, k, W)
    return x, y
