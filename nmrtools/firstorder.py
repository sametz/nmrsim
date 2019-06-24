""""Functions for calculating first-order spectra will appear here."""


def doublet(plist, J):
    """
    Applies a *J* coupling to each signal in a list of (frequency, intensity)
    signals, creating two half-intensity signals at +/- *J*/2.

    Parameters
    ---------
    plist : [(float, float)...]
        a list of (frequency{Hz}, intensity) tuples.
    J : float
        The coupling constant in Hz.

    Returns
    -------
    [(float, float)...]
        a list of (frequency, intensity) tuples.
    """
    res = []
    for v, i in plist:
        res.append((v - J / 2, i / 2))
        res.append((v + J / 2, i / 2))
    return res


def multiplet(signal, couplings):
    """
    Splits a set of signals into first-order multiplets.

    Parameters
    ---------
    signal : (float, float)
        a (frequency{Hz}, intensity) tuple;
    couplings : [(float, int)...]
        A list of (*J*, # of nuclei) tuples. The order of the tuples in
        couplings does not matter.
        e.g. to split a signal into a *dt, J* = 8, 5 Hz, use:
        ``couplings = [(8, 2), (5, 3)]``

    Returns
    -------
    [(float, float)...]
        a plist of the multiplet that results from splitting the plist
        signal(s) by each J.

    """
    res = [signal]
    for coupling in couplings:
        for i in range(coupling[1]):
            res = doublet(res, coupling[0])
    return res


def add_peaks(plist):
    """
    Reduces a list of (frequency, intensity) tuples to an
    (average frequency, total intensity) tuple.

    Parameter
    --------
    plist: [(float, float)...]
        a list of (frequency, intensity) tuples

    Returns
    -------
    (float, float)
        a tuple of (average frequency, total intensity)
    """
    # TODO: Is this if statement necessary?
    if len(plist) == 1:
        return plist[0]  # nothing to add
    v_total = 0
    i_total = 0
    for v, i in plist:
        v_total += v
        i_total += i
    return v_total / len(plist), i_total


def reduce_peaks(plist, tolerance=0):
    """
    Takes an ordered list of (x, y) tuples and adds together tuples whose first
    values are within a certain tolerance limit.

    Parameters
    ---------
    plist : [(float, float)...]
        A *sorted* list of (x, y) tuples (sorted by x)
    tolerance : float
        tuples that differ in x by <= tolerance are combined using ``add_peaks``

    Returns
    -------
    [(float, float)...]
        a list of (x, y) tuples where all x values differ by > `tolerance`
    """
    res = []
    work = [plist[0]]  # an accumulator of peaks to be processed
    for i in range(1, len(plist)):
        if not work:
            work.append(plist)
            continue
        if plist[i][0] - work[-1][0] <= tolerance:
            work.append(plist[i])  # accumulate close peaks
            continue
        else:
            res.append(add_peaks(work))
            work = [plist[i]]
    if work:
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
    n : int
        Number of nuclei.
    """
    factor = n / sum(intensities)
    for index, intensity in enumerate(intensities):
        intensities[index] = intensity * factor


def first_order(signal, couplings):  # Wa, RightHz, WdthHz not implemented yet
    """
    Splits a signal into a first-order multiplet.

    Parameters
    ---------
    signal : (float, float)
        a (frequency, intensity) tuple.
    couplings : [(float, int)...]
        a list of (J, # of nuclei) tuples.

    Returns
    -------
    [(float, float)...}
        a plist-style spectrum (list of (frequency, intensity) tuples)
    """
    return reduce_peaks(sorted(multiplet(signal, couplings)))


def normalize_spectrum(spectrum, n=1):
    """
    Normalize the intensities in a spectrum so that total intensity equals
    value n (nominally the number of nuclei giving rise to the signal).

    Parameters
    ---------
    spectrum : [(float, float)...]
        a list of (frequency, intensity) tuples.
    n : int or float
        total intensity to normalize to.
    """
    freq, int_ = [x for x, y in spectrum], [y for x, y in spectrum]
    _normalize(int_, n)
    return list(zip(freq, int_))
