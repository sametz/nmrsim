""""Functions for calculating first-order spectra.

The nmrsim.firstorder module provides the following functions:

* multiplet: performs first-order splitting of a signal into multiple signals.

* first_order_spin_system: provides a peaklist for several nuclei, using the
    same v/J parameters that are used for second-order spin systems.
    See nmrsim.qm for details on these parameters.
"""
from math import comb

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
        a list of (*J*, # of nuclei) tuples. The order of the tuples in
        couplings does not matter.
        e.g. to split a signal into a *dt, J* = 8, 5 Hz, use:
        ``couplings = [(8, 1), (5, 2)]``

    Returns
    -------
    [(float, float)...]
        a sorted peaklist for the multiplet that results from splitting the
        signal by each J.
    """
    res = [signal]
    for coupling in couplings:
        for _ in range(coupling[1]):
            res = _doublet(res, coupling[0])
    return sorted(reduce_peaks(res))


def _multiplet(signal, coupling):
    """
    Splits a single signal into a first-order multiplet using binomial coefficients.

    Parameters
    ---------
    signal : (float, float)
        a (frequency (Hz), intensity) tuple;
    couplings : (float, int)
        a (*J*, # of nuclei) tuple.
        e.g. to split a signal into a *t* with *J* = 8 Hz, use:
        ``couplings = (8, 2)``

    Returns
    -------
    [(float, float)...]
        a sorted peaklist for the multiplet that results from splitting the
        signal by each J.
    """

    shift, intensity = signal
    J, nsplit = coupling

    # Calculate the chemical shift of the first peak of the multiplet
    # s -> no offset
    # d -> offset by J/2
    # t -> offset by J
    # ...
    first_peak = shift - J * nsplit / 2

    new_signal = []

    # Precalculate binomial coefficients
    coeffs = [comb(nsplit, i) for i in range(nsplit + 1)]
    normalization_constant = intensity / sum(coeffs)
    for i in range(nsplit + 1):
        new_signal.append((first_peak + i * J, coeffs[i] * normalization_constant))

    return new_signal


def binomial_multiplet(signal, couplings):
    """
    Splits a set of signals into first-order multiplets using Pascal's triangle/binomial coefficients.
    Equivalent to ```nmrsim.firstorder.multiplet```

    Parameters
    ---------
    signal : (float, float)
        a (frequency (Hz), intensity) tuple;
    couplings : [(float, int)...]
        a list of (*J*, # of nuclei) tuples. The order of the tuples in
        couplings does not matter.
        e.g. to split a signal into a *dt, J* = 8, 5 Hz, use:
        ``couplings = [(8, 1), (5, 2)]``

    Returns
    -------
    [(float, float)...]
        a sorted peaklist for the multiplet that results from splitting the
        signal by each J.
    """
    res = [signal]
    for coupling in couplings:
        new_res = []
        for sig in res:
            new_res += _multiplet(sig, coupling)
        res = new_res

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
