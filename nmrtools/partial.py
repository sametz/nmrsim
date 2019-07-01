##############################################################################
# Non-QM solutions for specific second-order patterns
##############################################################################

# TODO: consistent names/API between arguments and internal variables.
# Cobbling these together from various sources resulted in inconsistencies
# between variable names in Reich's VB6 code and other sources. Need to find
# specific references for all formulas and refactor to consistent variable
# names.
# TODO: appropriate use of normalize-- for which of these functions, if any,
# is normalization possibly of interest?
from math import sqrt

import numpy as np

from nmrtools.firstorder import first_order
from nmrtools.math import _normalize, normalize_spectrum


def AB(Jab, Vab, Vcentr, normalize=True):
    """
    Calculates the signal frequencies and intensities for two strongly
    coupled protons (Ha and Hb).

    Parameters
    ---------
    Jab : float
        the coupling constant (Hz) between Ha and Hb
    Vab : float
        the chemical shift difference (Hz) between Ha and Hb in the absence
        of coupling.
    Vcentr : float
        the frequency (Hz) for the center of the AB quartet.
    normalize: bool
        whether the signal intensity should be normalized.

    Returns
    -------
    [(float, float)...]
        a list of four (frequency, intensity) tuples.
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
        the difference in the frequencies (Hz) of Ha and Hb in the absence of
        coupling.
    Vcentr : float
        the frequency (Hz) for the center of the AB2 signal.
    normalize: bool
        whether the signal intensity should be normalized.

    Returns
    -------
    [(float, float)...]
        a list of (frequency, intensity) tuples.
    """
    # Currently, there is a disconnect between the variable names in the GUI
    # and the variable names in this function. The following code provides a
    # temporary interface.

    J, dV, Vab = Jab, Vab, Vcentr

    # for now, old Jupyter code using Pople equations kept hashed out for now
    # Reich vs. Pople variable names are confused, e.g. Vab
    # So, variables being placed by position in the def header--CAUTION
    # From main passed in order of: Jab, Vab, Vcentr, Wa, RightHz, WdthHz
    # Here read in as:              J,   dV,  Vab,    "     "        "
    # dV = va - vb  # Reich: used d = Vb - vA and then mucked with sign of d
    # Vab = (va + vb) / 2  # Reich: ABOff
    dV = - dV
    va = Vab + (dV / 2)
    vb = va - dV
    Jmod = J * (3 / 4)  # This factor used in frequency calculations

    # In Reich's code, the definitions of cp/cm (for C_plus/C_minus) were
    # swapped, and then modifications using sign of d were employed. This
    # code hews closer to Pople definitions
    C_plus = sqrt(dV ** 2 + dV * J + (9 / 4) * (J ** 2)) / 2
    C_minus = sqrt(dV ** 2 - dV * J + (9 / 4) * (J ** 2)) / 2

    cos2theta_plus = (dV / 2 + J / 4) / C_plus  # Reich: cos2x
    cos2theta_minus = (dV / 2 - J / 4) / C_minus  # Reich: cos2y

    # This code differs from Reich's in the calculation of
    # the sin/cos x/y values

    sintheta_plus = sqrt((1 - cos2theta_plus) / 2)  # Reich: sinx
    sintheta_minus = sqrt((1 - cos2theta_minus) / 2)  # Reich: siny
    costheta_plus = sqrt((1 + cos2theta_plus) / 2)  # Reich: cosx
    costheta_minus = sqrt((1 + cos2theta_minus) / 2)  # Reich: cosy

    # Intensity formulas use the sin and cos of (theta_plus - theta_minus)
    # sin_dtheta is Reich's qq; cos_dtheta is Reich's rr

    sin_dtheta = sintheta_plus * costheta_minus - costheta_plus * sintheta_minus
    cos_dtheta = costheta_plus * costheta_minus + sintheta_plus * sintheta_minus

    # Calculate the frequencies and intensities.
    # V1-V4 are "Origin: A" (PSB Table 6-8);
    # V5-V8 are "Origin: B";
    # V9-V12 are "Origin: Comb."

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


def ABX(Jab, Jbx, Jax, Vab, Vcentr, normalize=True):
    """
    Reich-style inputs for an ABX spin system. TODO: complete rewrite
    Jab is the A-B coupling constant (Hz)
    dV is the difference in nuclei frequencies in the absence of coupling (Hz)
    Vab is the frequency for the center of the AB2 signal
    Wa is width of peak at half-height (not implemented yet)
    RightHz is the lower frequency limit for the window (not implemented yet)
    WdthHz is the width of the window in Hz (not implemented yet)
    return: peaklist of (frequency, intensity) tuples
    Calculates signal frequencies and intensities for an ABX spin system.

    Parameters
    ---------
    Jab : float
        the Ha-Hb coupling constant (Hz).
    Jbx : float
        the Ha-Hb coupling constant (Hz).
    Jax : float
        the Ha-Hb coupling constant (Hz).
    Vab : float
        the difference in the frequencies (Hz) of Ha and Hb in the absence of
        coupling.
    Vcentr : float
        the frequency (Hz) for the center of the AB2 signal.
    normalize: bool
        whether the signal intensity should be normalized.

    Returns
    -------
    [(float, float)...]
        a list of (frequency, intensity) tuples.
    """
    """
    In the WINDNMR main toolbar, only the parameters in the function args 
    can be changed (Jab, Jax, Jbx, Vab, Vcentr, ignoring the 
    Wa/Right-Hz/WdthHz parameters that nmrtools isn't adopting). However, 
    in WINDNMR parameters popup, each individual chemical shift for Ha, Hb, 
    and Hx can be entered.

    In the original WINDNMR interface, Vab is the difference in Ha/Hb 
    frequencies in the absence of coupling, and Vcentr is the average of 
    those frequencies.

    There's two ways I can see a user wanting to use the ABX function:
    1. as in WINDNMR: to see the effect of moving signals closer to each 
    other by adjusting Vab/Vcentr
    2. as in 2nd-order: supplying frequencies for all signals and Js. 

    The ABX formula here is a simplification that assumes X is far away in 
    chemical shift. Moving the frequency of x (vx) closer to those of a and b
    (va, vb) has no effect. Whatever the final decision is on API 
    implementation, this difference must be clear to the user.
    """
    # Another function where Reich vs. non-Reich variable names gets confusing
    # See comments in AB2 function
    # So, variables being placed by position in the def header--CAUTION
    # From main passed in order of: Jab, Jax, Jbx, Vab,  Vcentr, ...
    # Here read in as:              Jab, Jbx, Jax, dVab, Vab,    ...

    # CHANGE: with switch to kwargs used in function calls, the following
    # code matches this Reich code to the current view dictionary
    Jbx, Jax = Jax, Jbx
    dVab = Vab
    Vab = Vcentr

    # dVab = va - vb  # Reich: Vab
    # Vab = (va + vb) / 2  # Reich: ABOff

    # Reich's ABX: vx initialized as vb + 100
    vx = Vab - (dVab / 2) + 100

    dJx = Jax - Jbx  # GMS stepping-stone constant for readability

    # Retaining Reich names for next two constants
    cm = dJx / 2
    cp = Jax + Jbx

    # Reich re-defines constants m and l
    # (declaration/garbage-collection efficiency?)
    # GMS: using M and L for the first instance, m and n for second
    # (avoid lower-case l for variables)
    # Reich redefines m a third time for calculating X intensities
    # GMS: uses t

    M = dVab + cm
    L = dVab - cm

    D_plus = sqrt(M ** 2 + Jab ** 2) / 2
    D_minus = sqrt(L ** 2 + Jab ** 2) / 2

    sin2phi_plus = Jab / (2 * D_plus)  # Reich: sin2x
    sin2phi_minus = Jab / (2 * D_minus)  # Reich: sin2y
    cos2phi_plus = M / (2 * D_plus)  # Reich: cos2x
    cos2phi_minus = L / (2 * D_minus)  # Reich: cos2y

    m = (cp + 2 * Jab) / 4
    n = (cp - 2 * Jab) / 4  # Reich: l

    t = cos2phi_plus * cos2phi_minus + sin2phi_plus * sin2phi_minus
    # Calculate the frequencies and intensities.
    # V1-V4 are "Origin: B" (PSB Table 6-15);
    # V5-V8 are "Origin: A";
    # V9-V12 are "Origin: X" and V13-14 are "Origin: Comb. (X)"

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


def ABX3(Jab, Jax, Jbx, Vab, Vcentr, normalize=True):
    """
    Simulation of the AB part of an ABX3 spin system.

    TODO: explain simplification

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
        coupling.
    Vcentr : float
        the frequency (Hz) for the center of the AB2 signal.
    normalize: bool
        whether the signal intensity should be normalized.

    Returns
    -------
    [(float, float)...]
        a list of (frequency, intensity) tuples.
    """
    # Refactoring of Reich's code for simulating the ABX3 system.
    va = Vcentr - Vab / 2
    vb = Vcentr + Vab / 2
    a_quartet = first_order((va, 1), [(Jax, 3)])
    b_quartet = first_order((vb, 1), [(Jbx, 3)])
    res = []
    for i in range(4):
        dv = b_quartet[i][0] - a_quartet[i][0]
        abcenter = (b_quartet[i][0] + a_quartet[i][0]) / 2
        sub_abq = AB(Jab, dv, abcenter, normalize)  # Wa, RightHz, WdthHz not
        # implemented
        scale_factor = a_quartet[i][1]
        scaled_sub_abq = [(v, i * scale_factor) for v, i in sub_abq]
        res.extend(scaled_sub_abq)
    if normalize:
        normalize_spectrum(res, 5)  # TODO: check this factor
    return res


def AAXX(Jaa, Jxx, Jax, Jax_prime, Vcentr, normalize=True):
    """
    Simulates one half ('A' part) of an AA'XX' spin system.

    All frequencies are in Hz.

    Parameters
    ---------
    float Jaa, Jax, Jax, Jax_prime :
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
    # Define the constants required to calculate frequencies and intensities

    # K, L, M, N are as defined in PSB
    K = Jaa + Jxx  # Reich: K
    M = Jaa - Jxx  # Reich: l
    L = Jax - Jax_prime  # Reich: m
    N = Jax + Jax_prime  # Reich: n

    # Retaining Reich names for next two constants
    # Suggested refactoring: don't divide by 2 here; can simplify later formulas

    p = sqrt((K ** 2 + L ** 2)) / 2
    r = sqrt((M ** 2 + L ** 2)) / 2

    sin2theta_s = (1 - K / (2 * p)) / 2
    sin2theta_a = (1 - M / (2 * r)) / 2
    cos2theta_s = (1 + K / (2 * p)) / 2
    cos2theta_a = (1 + M / (2 * r)) / 2

    # Calculate the frequencies and intensities.
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
        _normalize(IList, 4)

    return list(zip(VList, IList))


def AABB(Vab, Jaa, Jbb, Jab, Jab_prime, Vcentr, normalize=True):
    """
    A wrapper for a second-order AA'BB' calculation, but using the
    values taken from the WINDNMR-style AA'BB' bar selected by the Multiplet
    menu.

    Parameters
    ---------

    Vab : float
        the difference in frequency (Hz) between Ha and Hb in the absence of
        coupling.
    float Jaa, Jbb, Jab, Jab_prime :
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
    from nmrtools.qm import spectrum
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

    result = spectrum(freqlist, J, normalize=normalize, sparse=False)
    return result