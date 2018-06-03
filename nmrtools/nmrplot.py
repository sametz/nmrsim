"""
Provide functions for creating lineshapes suitable for plotting.

For non-DNMR calculations, inputs are lists of (frequency, intensity) tuples,
which then have Lorentzian distributions applied to them.

For DNMR calculations, the lineshapes are directly computed. Currently,
non-quantum mechanical formulas for two uncoupled spins and for two coupled
spins are used.
"""
import numpy as np

from uw_dnmr.model.nmrmath import dnmr_AB, d2s_func


def lorentz(v, v0, I, w):
    """
    A lorentz function that takes linewidth at half intensity (w) as a
    parameter.
    :param v: Array of values at which to evaluate distribution.
    :param v0: Center of the distribution.
    :param I: relative intensity of the signal
    :param w: Peak width at half max intensity

    :returns: Distribution evaluated at points in x.
    """
    return I * ((0.5 * w) ** 2 / ((0.5 * w) ** 2 + (v - v0) ** 2))


def add_signals(linspace, peaklist, w):
    """
    Given a numpy linspace, a spectrum as a list of (frequency, intensity)
    tuples, and a linewidth, returns an array of y coordinates for the
    lineshape.

    :param linspace: a numpy linspace of x coordinates for the lineshape.
    :param peaklist: a list of (frequency, intensity) tuples
    :param w: peak width at half maximum intensity
    :returns: array of y coordinates for the lineshape
    """
    result = lorentz(linspace, peaklist[0][0], peaklist[0][1], w)
    for v, i in peaklist[1:]:
        result += lorentz(linspace, v, i, w)
    return result


# add_signals should supercede the adder function below--
# schedule for deletion
def adder(x, plist, Q=2):
    """
    :param x: the x coordinate (relative frequency in Hz)
    :param plist: a list of tuples of peak data (frequency, intensity)
    :param Q: the line width "fudge factor" used by lorentz2
    returns: the sum of the peak Lorentzian functions at x
    """
    total = 0
    for v, i in plist:
        total += lorentz2(x, v, i, Q)
    return total


def nmrplot(spectrum, y=1):
    """
    A no-frills routine that plots spectral simulation data.
    :param spectrum: A list of (frequency, intensity) tuples
    :param y: max intensity
    """
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
    :param spectrum: A list of (frequency, intensity) tuples
    :param w: peak width at half height
    :return: a tuple of x and y coordinate linspaces
    """
    spectrum.sort()
    r_limit = spectrum[-1][0] + 50
    l_limit = spectrum[0][0] - 50
    x = np.linspace(l_limit, r_limit, 2400)
    y = add_signals(x, spectrum, w)
    return x, y


def dnmrplot_2spin(va, vb, ka, Wa, Wb, pa):
    """Create a lineshape for the DNMR spectrum of two uncoupled nuclei
    undergoing exchange.

    :param va: The frequency of nucleus 'a' at the slow exchange limit
    :param vb: The frequency of nucleus 'b' at the slow exchange limit
    :param ka: The rate of nuclear exchange
    :param Wa: The width at half heigh of the signal for nucleus a (at the slow
    exchange limit).
    :param Wb: The width at half height of the signal for nucleus b (at the slow
    exchange limit).
    :param pa: The fraction of the population in state a (vs. state b)

    :return: a tuple of numpy arrays for frequencies (x coordinate) and
    corresponding intensities (y coordinate). Hard-coded for 800 data points
    and a frequency range from vb-50 to va+50.
    """

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

    :param v1: The frequency of nucleus 'a' at the slow exchange limit
    :param v2: The frequency of nucleus 'b' at the slow exchange limit
    :param J: The coupling constant between nuclei a and b
    :param k: The rate of two-site exchange of nuclei a and b
    :param W: The line width at the slow exchange limit

    :return: a tuple of numpy arrays for frequencies (x coordinate) and
    corresponding intensities (y coordinate). Hard-coded for 800 data points
    and a frequency range from vb-50 to va+50.
    """
    if v2 > v1:
        v1, v2 = v2, v1
    l_limit = v2 - 50
    r_limit = v1 + 50
    x = np.linspace(l_limit, r_limit, 800)
    y = dnmr_AB(x, v1, v2, J, k, W)
    return x, y


if __name__ == '__main__':

    reichdefault = (165.00, 135.00, 1.50, 0.50, 0.50, 50.00)
    x, y = dnmrplot_2spin(*reichdefault)

    def testplot(spectrum):
        """Used to test the spectral data generated by a simulation"""
        x, y = spectrum
        l_limit = x[0] - 50
        r_limit = x[-1] + 50
        lower_limit = min(y)
        upper_limit = max(y)
        plt.ylim(lower_limit, upper_limit)
        plt.gca().invert_xaxis()  # reverses x-axis "NMR-Style"
        plt.plot(x, y)

    testplot((x, y))
    plt.show()
