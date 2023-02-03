""""Functions for calculating first-order spectra.

The nmrsim.firstorder module provides the following functions:

* multiplet: performs first-order splitting of a signal into multiple signals.

* first_order_spin_system: provides a peaklist for several nuclei, using the
    same v/J parameters that are used for second-order spin systems.
    See nmrsim.qm for details on these parameters.
"""

from nmrsim.math import reduce_peaks


def _doublet(plist, J):
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


# TODO: consider making the multiplet and nmrsim.Multiplet arguments similar
def multiplet(signal, couplings):
    """
    Splits a set of signals into first-order multiplets.

    Parameters
    ---------
    signal : (float, float)
        a (frequency (Hz), intensity) tuple;
    couplings : [(float, int)...]
        A list of (*J*, # of nuclei) tuples. The order of the tuples in
        couplings does not matter.
        e.g. to split a signal into a *dt, J* = 8, 5 Hz, use:
        ``couplings = [(8, 2), (5, 3)]``

    Returns
    -------
    [(float, float)...]
        a sorted peaklist for the multiplet that results from splitting the
        signal by each J.
    """
    res = [signal]
    for coupling in couplings:
        for i in range(coupling[1]):
            res = _doublet(res, coupling[0])
    return sorted(reduce_peaks(res))


def first_order_spin_system(v, J):
    """
    Create a first-order peaklist of several multiplets from the same v/J
    arguments used for qm calculations.

    This allows a user to model several multiplets at once, rather than
    creating each multiplet individually. It also provides a "toggle" where
    the user, or a higher-level function/class (such as nmrsim.SpinSystem)
    can decide whether a spin system is modeled as first order or second order.

    Parameters
    ----------
    v : array-like [float...]
        an array of frequencies
    J : 2D array-like (square)
        a matrix of J coupling constants

    Returns
    -------
    [(float, float)...]
        a combined peaklist of signals for all the multiplets in the spin
        system.
    """
    result = []
    for i, v_ in enumerate(v):
        couplings = ((j, 1) for j in J[i] if j != 0)
        signal = multiplet((v_, 1), couplings)
        result += signal
    return reduce_peaks(sorted(result))
