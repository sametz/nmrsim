""""Functions for calculating first-order spectra will appear here."""
from dataclasses import dataclass


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


# def new_add_peaks(plist):
#     """Refactoring of add_peaks: more elegant, but also slower by ca 15x."""
#     from statistics import mean
#     v, i = zip(*plist)
#     average_v = mean(v)
#     total_i = sum(i)
#     return average_v, total_i


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


def first_order_spin_system(v, J):
    result = []
    for i, v_ in enumerate(v):
        couplings = ((j, 1) for j in J[i] if j != 0)
        signal = multiplet((v_, 1), couplings)
        result += signal
    return reduce_peaks(sorted(result))


# work in progress
# def spectrum_from_signals(signals):
#     spectrum = []
#     for signal in signals:
#         spectrum += multiplet(signals)
#     return reduce_peaks(spectrum)


# https://realpython.com/python-type-checking/
# https://blog.florimond.dev/reconciling-dataclasses-and-properties-in-python
@dataclass
class Multiplet:
    """This is a stub for now. Considering either a class or a dataclass for
    multiplets, spin systems, and spectra. Also consider naming:
    there is both function 'multiplet' and class 'Multiplet' right now!
    """
    v: float
    I: float
    couplings: list


""" API ideas:
    firstorder.multiplet for one signal
    firstorder.spectrum for multiple signals
    firstorder.spinsystem takes qm-style v and J arguments, and returns a 
        firstorder.spectrum?
    thinking bigger picture: an nmrtools.spectrum class that holds all data 
    needed for a complete spectrum, including spectrometer frequency and 
    whether to calculate it as first-order or second-order.
    
    Convenience functions for how users will supply arguments? In particular,
    J couplings.
    J = [(7.1, 3), (1.0, 2)]
    J = {'J12': 7.1,
         'J13': 7.1,
         'J14': 1.0} and parse into a matrix?
"""