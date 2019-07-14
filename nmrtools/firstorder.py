""""Functions for calculating first-order spectra will appear here."""

from nmrtools.math import reduce_peaks


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


def first_order_spin_system(v, J):
    """
    Can create a first-order peaklist from the same v/J arguments used for
    qm calculations.

    Parameters
    ----------
    v : array-like [float...]
        an array of frequencies
    J : 2D array-like (square)
        a matrix of J coupling constants

    Returns
    -------

    """
    result = []
    for i, v_ in enumerate(v):
        couplings = ((j, 1) for j in J[i] if j != 0)
        signal = multiplet((v_, 1), couplings)
        result += signal
    return reduce_peaks(sorted(result))
