"""A collection of functions for processing simulated NMR spectra.

Terms used:
signal: a pair (e.g. tuple) of frequency, intensity values
peaklist: a list (or 1D-array-like) of signals.

Provides the following functions:

* add_peaks: combines a list of signals into one signal of average frequency
  and summed intensity.

* reduce_peaks: processes a peaklist so that signals within a frequency
  tolerance are added together.

* normalize_peaklist: scales a peaklist so that intensities add to a specific
  value.
* lorentz: given a frequency, a signal and a linewidth, calculates an
  intensity. Used to calculate Lorentzian lineshapes for signals.

* get_intensity: given a lineshape and a frequency, find the intensity at the
  datapoint closest to that frequency.
"""
import numpy as np


def add_peaks(plist):
    """
    Reduces a list of (frequency, intensity) tuples to an
    (average frequency, total intensity) tuple.

    Parameters
    ----------
    plist: [(float, float)...]
        a list of (frequency, intensity) tuples

    Returns
    -------
    (float, float)
        a tuple of (average frequency, total intensity)
    """
    v_total = 0
    i_total = 0
    for v, i in plist:
        v_total += v
        i_total += i
    return v_total / len(plist), i_total


def reduce_peaks(plist_, tolerance=0):
    """
    Takes a list of (x, y) tuples and adds together tuples whose values are
    within a certain tolerance limit.

    Parameters
    ---------
    plist_ : [(float, float)...]
        A list of (x, y) tuples
    tolerance : float
        tuples that differ in x by <= tolerance are combined using `add_peaks`

    Returns
    -------
    [(float, float)...]
        a list of (x, y) tuples where all x values differ by > `tolerance`
    """
    res = []
    work = []  # an accumulator of peaks to be added
    plist = sorted(plist_)
    for peak in plist:
        if not work:
            work.append(peak)
            continue
        if peak[0] - work[-1][0] <= tolerance:
            work.append(peak)
            continue
        else:
            res.append(add_peaks(work))
            work = [peak]
    if work:  # process any remaining work after for loop
        res.append(add_peaks(work))

    return res


def _normalize(intensities, n=1):
    """
    Scale a list of intensities so that they sum to the total number of
    nuclei.

    Parameters
    ---------
    intensities : [float...]
        A list of intensities.
    n : int (optional)
        Number of nuclei (default = 1).
    """
    factor = n / sum(intensities)
    intensities[:] = [factor * i for i in intensities]


def normalize_peaklist(peaklist, n=1):
    """
    Normalize the intensities in a peaklist so that total intensity equals
    value n (nominally the number of nuclei giving rise to the signal).

    Parameters
    ---------
    peaklist : [(float, float)...]
        a list of (frequency, intensity) tuples.
    n : int or float (optional)
        total intensity to normalize to (default = 1).
    """
    freq, int_ = [x for x, y in peaklist], [y for x, y in peaklist]
    _normalize(int_, n)
    return list(zip(freq, int_))


def lorentz(v, v0, I, w):
    """
    A lorentz function that takes linewidth at half intensity (w) as a
    parameter.

    When `v` = `v0`, and `w` = 0.5 (Hz), the function returns intensity I.

    Arguments
    ---------
    v : float
        The frequency (x coordinate) in Hz at which to evaluate intensity (y
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
    return scaling_factor * I * ((0.5 * w) ** 2 / ((0.5 * w) ** 2 + (v - v0) ** 2))


def add_lorentzians(linspace, peaklist, w):
    """
    Given a numpy linspace, a peaklist of (frequency, intensity)
    tuples, and a linewidth, returns an array of y coordinates for the
    total line shape.

    Arguments
    ---------
    linspace : array-like
        Normally a numpy.linspace of x coordinates corresponding to frequency
        in Hz.
    peaklist : [(float, float)...]
        A list of (frequency, intensity) tuples.
    w : float
        Peak width at half maximum intensity.

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


def get_intensity(lineshape, x):
    """
    A crude method to find the intensity of data point closest to
    frequency x. Better: interpolate between two data points if match isn't
    exact (TODO?)

    Parameters
    ----------
    lineshape : tuple of (x, y) arrays for frequency, intensity data
    x : frequency lookup

    Returns
    -------
    float : the intensity at that frequency
    """
    nearest_x_index = np.abs(lineshape[0] - x).argmin()
    return lineshape[1][nearest_x_index]


def get_maxima(lineshape):
    """
    Crude function that returns maxima in the lineshape.

    Parameters
    ----------
    lineshape : tuple of frequency, intensity arrays

    Returns
    -------
    a list of (frequency, intensity) tuples for individual maxima.
    """
    res = []
    for n, val in enumerate(lineshape[1][1:-2]):
        index = n + 1  # start at lineshape[1][1]
        lastvalue = lineshape[1][index - 1]
        nextvalue = lineshape[1][index + 1]

        if lastvalue < val and nextvalue < val:
            print("MAXIMUM FOUND AT: ")
            print((lineshape[0][index], val))
            res.append((lineshape[0][index], val))
    return res
