import numpy as np
from math import sqrt


def dnmr_AB(v, v1, v2, J, k, W):
    """
    A translation of the equation from Weil's JCE paper (NOTE: Reich pointed
    out that it has a sign typo!).
    p. 14, for the uncoupled 2-site exchange simulation.
    v: frequency whose amplitude is to be calculated
    va, vb: frequencies of a and b nuclei (slow exchange limit, no coupling;
    va > vb)
    ka: rate constant for state A--> state B
    pa: fraction of population in state Adv: frequency difference (va - vb)
    between a and b singlets (slow exchange)
    T2a, T2b: T2 (transverse relaxation time) for each nuclei
    returns: amplitude at frequency v
    """
    pi = np.pi
    vo = (v1 + v2) / 2
    tau = 1 / k
    tau2 = 1 / (pi * W)
    a1_plus = 4 * pi ** 2 * (vo - v + J / 2) ** 2
    a1_minus = 4 * pi ** 2 * (vo - v - J / 2) ** 2
    a2 = - ((1 / tau) + (1 / tau2)) ** 2
    a3 = - pi ** 2 * (v1 - v2) ** 2
    a4 = - pi ** 2 * J ** 2 + (1 / tau ** 2)
    a_plus = a1_plus + a2 + a3 + a4
    a_minus = a1_minus + a2 + a3 + a4

    b_plus = 4 * pi * (vo - v + J / 2) * (
        (1 / tau) + (1 / tau2)) - 2 * pi * J / tau
    b_minus = 4 * pi * (vo - v - J / 2) * (
        (1 / tau) + (1 / tau2)) + 2 * pi * J / tau

    r_plus = 2 * pi * (vo - v + J)
    r_minus = 2 * pi * (vo - v - J)

    s = (2 / tau) + (1 / tau2)

    n1 = r_plus * b_plus - s * a_plus
    d1 = a_plus ** 2 + b_plus ** 2
    n2 = r_minus * b_minus - s * a_minus
    d2 = a_minus ** 2 + b_minus ** 2

    I = (n1 / d1) + (n2 / d2)
    return I


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

    # Function dnmr_ab is applied across np.linspace x:
    y = dnmr_AB(x, v1, v2, J, k, W)
    return x, y


if __name__ == '__main__':
    args = (165, 135, 12, 12, 0.5)
    lineshape = dnmrplot_AB(*args)
    print(lineshape)