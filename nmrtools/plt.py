import matplotlib.pyplot as plt
import numpy as np

from nmrtools.math import lorentz

"""TODO: rethink plot routines. There are two main flavors:
    * plot peaklists, either as stick plots or with Lorentzians
    * plot lineshapes
    
Different routes:
    peaklist -> stick plot
    peaklist -> Lorentzian lineshape
    DNMR simulation -> lineshape
    
Consider function factories that will crank out the desired plot object.

Peaklists may be np.arrays, or lists of tuples, depending on origin. Either
use a consistent form throughout (e.g. np.array, converting users array-like
objects as needed) or refactor to allow multiple inputs.
"""


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
    # TODO: consider naming, and confusion with .math.add_peaks
    # TODO: function looks clunky. Refactor?
    result = lorentz(linspace, peaklist[0][0], peaklist[0][1], w)
    for v, i in peaklist[1:]:
        result += lorentz(linspace, v, i, w)
    return result


def mplplot(peaklist, w=1, y_min=-0.01, y_max=1, points=800, limits=None):
    """
    A no-frills routine that plots spectral simulation data.

    Arguments
    ---------
    peaklist : [(float, float)...]
        a list of (frequency, intensity) tuples.
    w : float
        peak width at half height
    y_max : float or int
        maximum intensity for the plot.
    points : int
        number of data points
    limits : (float, float)
        frequency limits for the plot
    """
    # import matplotlib.pyplot as plt

    peaklist.sort()  # Could become costly with larger spectra
    if limits:
        try:
            l_limit, r_limit = limits
            l_limit = float(l_limit)
            r_limit = float(r_limit)
        except Exception as e:
            print(e)
            print('limits must be a tuple of two numbers')
            # return None
            raise
        if l_limit > r_limit:
            l_limit, r_limit = r_limit, l_limit
    else:
        l_limit = peaklist[0][0] - 50
        r_limit = peaklist[-1][0] + 50
    x = np.linspace(l_limit, r_limit, points)
    plt.ylim(y_min, y_max)
    plt.gca().invert_xaxis()  # reverses the x axis
    y = add_signals(x, peaklist, w)
    # noinspection PyTypeChecker
    plt.plot(x, y)
    plt.show()
    return x, y
    # TODO: or return plt? Decide behavior


def mplplot_stick(peaklist, y_min=-0.01, y_max=1, limits=None):
    """TODO: description below incorrect. x, y must be numpy.ndarray.
    Decide on a consistent interface (e.g. vs. mplplot).
    Also: setting limits by adding small peaks is hacky, and also doesn't work
    if peaklist isn't ordered.
    """
    """
    matplotlib plot a spectrum in "stick" (stem) style.

    Arguments
    ---------
    x : [float...]
        a list of frequencies
    y : [float...]
        a list of intensities corresponding with x
    max_y : float
        maximum intensity for the plot.
    """
    # import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    if limits:
        try:
            l_limit, r_limit = limits
            l_limit = float(l_limit)
            r_limit = float(r_limit)
        except Exception as e:
            print(e)
            print('limits must be a tuple of two numbers')
            raise
        if l_limit > r_limit:
            l_limit, r_limit = r_limit, l_limit
    else:
        l_limit = peaklist[0][0] - 50
        r_limit = peaklist[-1][0] + 50
    x, y = zip(*peaklist)
    x = np.append(x, [l_limit, r_limit])
    y = np.append(y, [0.001, 0.001])
    plt.xlim(r_limit, l_limit)
    plt.ylim(y_min, y_max)
    ax.stem(x, y, markerfmt=' ', basefmt='C0-')
    ax.invert_xaxis()
    plt.show()
    return x, y


def mplplot_lineshape(x, y, y_min=None, y_max=None, limits=None):
    # fig, ax = plt.subplots()

    if limits:
        try:
            l_limit, r_limit = limits
            l_limit = float(l_limit)
            r_limit = float(r_limit)
        except Exception as e:
            print(e)
            print('limits must be a tuple of two numbers')
            raise
        if l_limit > r_limit:
            l_limit, r_limit = r_limit, l_limit
    else:
        l_limit = x[0]  # assumes x already sorted low->high
        r_limit = x[-1]

    if y_min is None or y_max is None:  # must test vs None so that 0 = True
        margin = max(y) * 0.1
        if y_min is None:
            y_min = min(y) - margin
        if y_max is None:
            y_max = max(y) + margin
    plt.xlim(r_limit, l_limit)  # should invert x axis
    plt.ylim(y_min, y_max)
    plt.plot(x, y)
    plt.show()
    return x, y
