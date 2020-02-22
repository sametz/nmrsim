"""Non-quantum mechanical solutions for specific second-order patterns.

These are adapted from the routines from WINDNMR [1]_ by Hans Reich,
U. Wisconsin, and equations from Pople, Schneider and Bernstein [2]_. Note that
many of the names for arguments, etc. are non-Pythonic but chosen to match the
WINDNMR interface and source code (for now).

The `partials` module provides the following functions:

* AB: simulates an AB quartet.

* AB2: simulates an AB2 system.

* ABX: simulates an ABX system.

* ABX3: simulates an ABX3 system.

* AAXX: simulates an AA'XX' system.

* AABB: simulates an AA'BB' system.

References
----------
.. [1] WINDNMR-Pro home page: https://www.chem.wisc.edu/areas/reich/plt/windnmr.htm
.. [2] Pople, J.A.; Schneider, W.G.; Bernstein, H.J. *High-Resolution Nuclear Magnetic Resonance.* New York:
   McGraw-Hill, 1959.

"""
# TODO: refactor away from legacy argument/variable names to user-friendly,
# pythonic ones.
from math import sqrt

import numpy as np

from nmrsim.firstorder import multiplet
from nmrsim.math import _normalize


def AB(Jab, Vab, Vcentr, normalize=True):
    """
    Calculates the signal frequencies and intensities for two strongly
    coupled protons (Ha and Hb).

    Parameters
    ---------
    Jab : float
        The coupling constant (Hz) between Ha and Hb
    Vab : float
        The chemical shift difference (Hz) between Ha and Hb in the absence
        of coupling.
    Vcentr : float
        The frequency (Hz) for the center of the AB quartet.
    normalize: bool
        Whether the signal intensity should be normalized.

    Returns
    -------
    [(float, float)...]
        A list of four (frequency, intensity) tuples.
    """
    J = Jab
    dv = Vab
    c = ((dv ** 2 + J ** 2) ** 0.5) / 2
    center = Vcentr
    v1 = center - c - (J / 2)
    v2 = v1 + J
    v3 = center + c - (J / 2)
    v4 = v3 + J
    dI = J / (2 * c)
    I1 = 1 - dI
    I2 = 1 + dI
    I3 = I2
    I4 = I1
    vList = [v1, v2, v3, v4]
    IList = [I1, I2, I3, I4]
    if normalize:
        _normalize(IList, 2)
    return list(zip(vList, IList))


def AB2(Jab, Vab, Vcentr, normalize=True):
    """
    Calculates signal frequencies and intensities for an AB2 spin system.

    Parameters
    ---------
    Jab : float
        the Ha-Hb coupling constant (Hz).
    Vab : float
        the difference in the frequencies (Hz). A positive value means vb > va;
        negative means va > vb.
    Vcentr : float
        the frequency (Hz) for the center of the AB2 signal.
    normalize: bool
        whether the signal intensity should be normalized.

    Returns
    -------
    [(float, float)...]
        a list of (frequency, intensity) tuples.
    """
    # There is a disconnect between the variable names in the WINDNMR GUI and
    # the variable names in this function.
    # The following code provides a temporary interface until this is
    # refactored.
    J, dV, Vab = Jab, Vab, Vcentr

    # Also, inconsistencies in WINDNMR GUI, internal WINDNMR code, and Pople
    # equations require a conversion.
    dV = - dV
    va = Vab + (dV / 2)
    vb = va - dV

    Jmod = J * (3 / 4)
    C_plus = sqrt(dV ** 2 + dV * J + (9 / 4) * (J ** 2)) / 2
    C_minus = sqrt(dV ** 2 - dV * J + (9 / 4) * (J ** 2)) / 2
    cos2theta_plus = (dV / 2 + J / 4) / C_plus
    cos2theta_minus = (dV / 2 - J / 4) / C_minus
    sintheta_plus = sqrt((1 - cos2theta_plus) / 2)
    sintheta_minus = sqrt((1 - cos2theta_minus) / 2)
    costheta_plus = sqrt((1 + cos2theta_plus) / 2)
    costheta_minus = sqrt((1 + cos2theta_minus) / 2)
    sin_dtheta = sintheta_plus * costheta_minus - costheta_plus * sintheta_minus
    cos_dtheta = costheta_plus * costheta_minus + sintheta_plus * sintheta_minus

    # In Pople, Schneider and Bernstein, Table 6-8:
    # V1-V4 are "Origin: A";
    # V5-V8 are "Origin: B";
    # V9 is "Origin: Comb."
    V1 = Vab + Jmod + C_plus
    V2 = vb + C_plus + C_minus
    V3 = va
    V4 = Vab - Jmod + C_minus
    V5 = vb + C_plus - C_minus
    V6 = Vab + Jmod - C_plus
    V7 = vb - C_plus + C_minus
    V8 = Vab - Jmod - C_minus
    V9 = vb - C_plus - C_minus

    I1 = (sqrt(2) * sintheta_plus - costheta_plus) ** 2
    I2 = (sqrt(2) * sin_dtheta + costheta_plus * costheta_minus) ** 2
    I3 = 1
    I4 = (sqrt(2) * sintheta_minus + costheta_minus) ** 2
    I5 = (sqrt(2) * cos_dtheta + costheta_plus * sintheta_minus) ** 2
    I6 = (sqrt(2) * costheta_plus + sintheta_plus) ** 2
    I7 = (sqrt(2) * cos_dtheta - sintheta_plus * costheta_minus) ** 2
    I8 = (sqrt(2) * costheta_minus - sintheta_minus) ** 2
    I9 = (sqrt(2) * sin_dtheta + sintheta_plus * sintheta_minus) ** 2
    vList = [V1, V2, V3, V4, V5, V6, V7, V8, V9]
    IList = [I1, I2, I3, I4, I5, I6, I7, I8, I9]

    if normalize:
        _normalize(IList, 3)
    return list(zip(vList, IList))


def ABX(Jab, Jax, Jbx, Vab, Vcentr, vx, normalize=True):
    """
    Non-QM approximation for an ABX spin system. The approximation assumes
    that Hx is very far away in chemical shift from Ha/Hb.

    Parameters
    ---------
    Jab : float
        The Ha-Hb coupling constant (Hz).
    Jax : float
        The Ha-Hx coupling constant (Hz).
    Jbx : float
        The Hb-Hx coupling constant (Hz).
    Vab : float
        The difference in the frequencies (in the absence of
        coupling) of Ha and Hb (Hz).
    Vcentr : float
        The frequency (Hz) for the center of the AB signal.
    vx : float
        The frequency (Hz) for Hx in the absence of coupling.

    normalize: bool (optional)
        whether the signal intensity should be normalized. If false, the total
        signal intensity happens to be ~12.

    Returns
    -------
    [(float, float)...]
        a list of (frequency, intensity) tuples.
    """
    # Contradictions in naming between WINDNMR's interface, internal code, and
    # Pople/Schneider/Bernstein "fixed" with these reassignments:
    dVab = Vab
    Vab = Vcentr

    dJx = Jax - Jbx
    cm = dJx / 2
    cp = Jax + Jbx
    M = dVab + cm
    L = dVab - cm
    D_plus = sqrt(M ** 2 + Jab ** 2) / 2
    D_minus = sqrt(L ** 2 + Jab ** 2) / 2
    sin2phi_plus = Jab / (2 * D_plus)
    sin2phi_minus = Jab / (2 * D_minus)
    cos2phi_plus = M / (2 * D_plus)
    cos2phi_minus = L / (2 * D_minus)
    m = (cp + 2 * Jab) / 4
    n = (cp - 2 * Jab) / 4
    t = cos2phi_plus * cos2phi_minus + sin2phi_plus * sin2phi_minus

    # In Pople, Schneider and Bernstein, Table 6-15:
    # V1-V4 are "Origin: B" (PSB Table 6-15);
    # V5-V8 are "Origin: A";
    # V9-V12 are "Origin: X";
    # V13-14 are "Origin: Comb. (X)"
    V1 = Vab - m - D_minus
    V2 = Vab + n - D_plus
    V3 = Vab - n - D_minus
    V4 = Vab + m - D_plus
    V5 = Vab - m + D_minus
    V6 = Vab + n + D_plus
    V7 = Vab - n + D_minus
    V8 = Vab + m + D_plus
    V9 = vx - cp / 2
    V10 = vx + D_plus - D_minus
    V11 = vx - D_plus + D_minus
    V12 = vx + cp / 2
    V13 = vx - D_plus - D_minus
    V14 = vx + D_plus + D_minus
    I1 = 1 - sin2phi_minus
    I2 = 1 - sin2phi_plus
    I3 = 1 + sin2phi_minus
    I4 = 1 + sin2phi_plus
    I5 = I3
    I6 = I4
    I7 = I1
    I8 = I2
    I9 = 1
    I10 = (1 + t) / 2
    I11 = I10
    I12 = 1
    I13 = (1 - t) / 2
    I14 = I13
    VList = [V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14]
    IList = [I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11, I12, I13, I14]
    if normalize:
        _normalize(IList, 3)
    return list(zip(VList, IList))


def ABX3(Jab, Jax, Jbx, Vab, Vcentr):
    """
    Simulation of the AB part of an ABX3 spin system.

    Parameters
    ---------
    Jab : float
        the Ha-Hb coupling constant (Hz).
    Jax : float
        the Ha-Hb coupling constant (Hz).
    Jbx : float
        the Ha-Hb coupling constant (Hz).
    Vab : float
        the difference in the frequencies (Hz) of Ha and Hb in the absence of
        coupling. Positive when vb > va.
    Vcentr : float
        the frequency (Hz) for the center of the AB signal.

    Returns
    -------
    [(float, float)...]
        a list of (frequency, intensity) tuples.
    """
    # First: simulate two quartets for va and vb ("Jab turned off")
    va = Vcentr - Vab / 2
    vb = Vcentr + Vab / 2
    a_quartet = multiplet((va, 1), [(Jax, 3)])
    b_quartet = multiplet((vb, 1), [(Jbx, 3)])
    res = []
    # Then: for each pair of a and b singlets in the quartets, calculate an
    # AB quartet ("Turn Jab on").
    for i in range(4):
        dv = b_quartet[i][0] - a_quartet[i][0]
        abcenter = (b_quartet[i][0] + a_quartet[i][0]) / 2
        sub_abq = AB(Jab, dv, abcenter, normalize=True)
        scale_factor = a_quartet[i][1]
        scaled_sub_abq = [(v, i * scale_factor) for v, i in sub_abq]
        res.extend(scaled_sub_abq)
    return res


def AAXX(Jaa, Jxx, Jax, Jax_prime, Vcentr, normalize=True):
    """
    Simulates one half ('A' part) of an AA'XX' spin system.

    All frequencies are in Hz.

    Parameters
    ---------
    Jaa, Jax, Jax, Jax_prime : float
        Jaa is the JAA' coupling constant;
        Jxx the JXX';
        Jax the JAX; and
        JAX_prime the JAX'.
    Vcentr : float
        the frequency for the center of the signal.
    normalize: bool
        whether the signal intensity should be normalized (to 2).

    Returns
    -------
    [(float, float)...]
        a list of (frequency, intensity) tuples.
    """
    # Define the constants required to calculate frequencies and intensities

    # K, L, M, N are as defined in Pople, Schneider and Bernstein
    K = Jaa + Jxx
    M = Jaa - Jxx
    L = Jax - Jax_prime
    N = Jax + Jax_prime
    p = sqrt((K ** 2 + L ** 2)) / 2
    r = sqrt((M ** 2 + L ** 2)) / 2
    sin2theta_s = (1 - K / (2 * p)) / 2
    sin2theta_a = (1 - M / (2 * r)) / 2
    cos2theta_s = (1 + K / (2 * p)) / 2
    cos2theta_a = (1 + M / (2 * r)) / 2

    # See PSB Table 6-18. Transitions 1-4 are condensed into V1 and V2.
    V1 = Vcentr + N / 2
    V2 = Vcentr - N / 2
    V3 = Vcentr + K / 2 + p
    V4 = Vcentr - K / 2 + p
    V5 = Vcentr + K / 2 - p
    V6 = Vcentr - K / 2 - p
    V7 = Vcentr + M / 2 + r
    V8 = Vcentr - M / 2 + r
    V9 = Vcentr + M / 2 - r
    V10 = Vcentr - M / 2 - r

    I1 = 2
    I2 = I1
    I3 = sin2theta_s
    I4 = cos2theta_s
    I5 = I4
    I6 = I3
    I7 = sin2theta_a
    I8 = cos2theta_a
    I9 = I8
    I10 = I7

    VList = [V1, V2, V3, V4, V5, V6, V7, V8, V9, V10]
    IList = [I1, I2, I3, I4, I5, I6, I7, I8, I9, I10]
    if normalize:
        _normalize(IList, 2)
    return list(zip(VList, IList))


def AABB(Vab, Jaa, Jbb, Jab, Jab_prime, Vcentr, normalize=True, **kwargs):
    """
    A wrapper for a second-order AA'BB' calculation, but using the
    same arguments as WINDNMR.

    Parameters
    ---------
    Vab : float
        the difference in frequency (Hz) between Ha and Hb in the absence of
        coupling. A positive number indicates vb > va.
    Jaa, Jbb, Jab, Jab_prime : float
        Jaa is the JAA' coupling constant;
        Jxx the JXX';
        Jax the JAX; and
        JAX_prime the JAX'.
    Vcentr : float
        the frequency for the center of the signal.
    normalize: bool
        whether the signal intensity should be normalized.

    Returns
    -------
    [(float, float)...]
        a list of (frequency, intensity) tuples.
    """
    from nmrsim.qm import qm_spinsystem
    va = Vcentr - Vab / 2
    vb = Vcentr + Vab / 2
    freqlist = [va, va, vb, vb]
    J = np.zeros((4, 4))
    J[0, 1] = Jaa
    J[0, 2] = Jab
    J[0, 3] = Jab_prime
    J[1, 2] = Jab_prime
    J[1, 3] = Jab
    J[2, 3] = Jbb
    J = J + J.T

    result = qm_spinsystem(freqlist, J, normalize=normalize, sparse=False, **kwargs)
    return result
