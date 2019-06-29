import numpy as np

# There are multiple, redundant routines for calculating DNMR lineshapes. All
#  previous approaches are preserved here for now.
# TODO: review, refactor, and select the best-performing functions based on
# speed test results.


def dnmr_2spin(v, va, vb, ka, Wa, Wb, pa):
    """
    A translation of the equation from Sandström's Dynamic NMR Spectroscopy,
    p. 14, for the uncoupled 2-site exchange simulation.

    This function is to be applied along a numpy linspace (evenly spaced x
    coordinates corresponding to frequency in Hz), to create a list of
    intensities (y coordinate).

    Parameters
    ---------

    v : float
        a frequency (x coordinate) at which an amplitude (y coordinate) is to be
        calculated.
    float va, vb :
        frequencies of a and b singlets (slow exchange limit) (`va` > `vb`)
    ka : float
        rate constant for state A--> state B
    float Wa, Wb :
        peak widths at half height (slow exchange limit)
    pa : float
        fraction of population in state A
    Returns
    -------
    float
        the intensity (y coordinate of the lineshape) at frequency `v`.

    References
    ----------
    TODO: add reference
    """

    pi = np.pi
    pb = 1 - pa
    tau = pb / ka
    dv = va - vb
    Dv = (va + vb) / 2 - v
    T2a = 1 / (pi * Wa)
    T2b = 1 / (pi * Wb)

    P = tau * ((1 / (T2a * T2b)) - 4 * (pi ** 2) * (Dv ** 2) +
               (pi ** 2) * (dv ** 2))
    P += ((pa / T2a) + (pb / T2b))

    Q = tau * (2 * pi * Dv - pi * dv * (pa - pb))

    R = 2 * pi * Dv * (1 + tau * ((1 / T2a) + (1 / T2b)))
    R += pi * dv * tau * ((1 / T2b) - (1 / T2a)) + pi * dv * (pa - pb)

    I = (P * (1 + tau * ((pb / T2a) + (pa / T2b))) + Q * R) / (P ** 2 + R ** 2)
    return I


def two_spin(v, va, vb, ka, wa, wb, pa):
    """pyDNMR renamed dnmr_2spin and some arguments. This is temporary glue
    to make sure that pyDNMR will work with the nmrtools library.
    """
    # TODO: make sure this is no longer needed, then delete
    return dnmr_2spin(v, va, vb, ka, wa, wb, pa)


def d2s_func(va, vb, ka, wa, wb, pa):
    """
    Create a function that requires only frequency as an argument, and used to
    calculate intensities across array of frequencies in the DNMR
    spectrum for two uncoupled spin-half nuclei.

    The idea is to calculate expressions
    that are independent of frequency only once, and then use them in a new
    function that depends only on v. This would avoid unnecessarily
    repeating some of the same operations.

    This function-within-a-function should be refactored to
    function-within-class!

    :param va: The frequency of nucleus 'a' at the slow exchange limit. va > vb
    :param vb: The frequency of nucleus 'b' at the slow exchange limit. vb < va
    :param ka: The rate constant for state a--> state b
    :param wa: The width at half heigh of the signal for nucleus a (at the slow
    exchange limit).
    :param wb: The width at half heigh of the signal for nucleus b (at the slow
    exchange limit).
    :param pa: The fraction of the population in state a.
    :param pa: fraction of population in state a
    wa, wb: peak widths at half height (slow exchange), used to calculate T2s

    returns: a function that takes v (x coord or numpy linspace) as an argument
    and returns intensity (y).
    """
    # TODO: this seems like the hard way to create a partial function. Try a
    # functools.partial version of this.
    # TODO: factor pis out; redo comments to explain excision of v-independent
    # terms

    pi = np.pi
    pi_squared = pi ** 2
    T2a = 1 / (pi * wa)
    T2b = 1 / (pi * wb)
    pb = 1 - pa
    tau = pb / ka
    dv = va - vb
    Dv = (va + vb) / 2
    P = tau * (1 / (T2a * T2b) + pi_squared * (dv ** 2)) + (pa / T2a + pb / T2b)
    p = 1 + tau * ((pb / T2a) + (pa / T2b))
    Q = tau * (- pi * dv * (pa - pb))
    R = pi * dv * tau * ((1 / T2b) - (1 / T2a)) + pi * dv * (pa - pb)
    r = 2 * pi * (1 + tau * ((1 / T2a) + (1 / T2b)))

    def maker(v):
        """
        Scheduled for refactoring.
        :param v: frequency
        :return: function that calculates the intensity at v
        """
        # TODO: fix docstring, explain _P _Q etc correlate to P, Q etc in lit.
        # FIXED: previous version of this function used
        # nonlocal Dv, P, Q, R
        # but apparently when function called repeatedly these values would
        # become corrupted (turning into arrays?!)
        # Solution: add underscores to create copies of any variables in
        # outer scope whose values are changed in the inner scope.

        _Dv = Dv - v
        _P = P - tau * 4 * pi_squared * (_Dv ** 2)
        _Q = Q + tau * 2 * pi * _Dv
        _R = R + _Dv * r
        return(_P * p + _Q * _R) / (_P ** 2 + _R ** 2)
    return maker


def reich(v, va, vb, ka, Wa, Wb, pa):
    """
    A traslation of the actual VB code used in WINDNMR. Was used for error
    checking. Scheduled for deletion.
    """
    # TODO: delete when no longer needed
    # print('Reich was entered')
    PI = np.pi
    R21 = PI * Wa
    R22 = PI * Wb
    mshifts1 = va
    mshifts2 = vb
    pop1 = pa
    pop2 = 1 - pop1  # i.e. pb
    tau = pop2 / ka
    deltanu = va - vb
    r1 = 1 + tau * (R21 + R22)
    Rr2 = PI * deltanu * tau * (R22 - R21)
    R3 = PI * deltanu * (pop1 - pop2)
    p1 = tau * R21 * R22
    p3 = tau * PI * PI * deltanu * deltanu
    PI2 = tau * 4 * PI * PI
    pitau = tau * 2 * PI
    popratio1 = p1 + p3 + (pop1 * R21) + (pop2 * R22)
    popratio2 = 1 + tau * ((pop2 * R21) + (pop1 * R22))
    CentFreq = 0.5 * (mshifts1 + mshifts2)
    Delfreq = CentFreq - v
    p2 = PI2 * Delfreq * Delfreq
    P = -p2 + popratio1
    Q = pitau * Delfreq - tau * R3
    R = 2 * PI * Delfreq * r1 + Rr2 + R3
    return (P * popratio2 + Q * R) / ((P * P) + (R * R))


class TwoSinglets:
    """
    Attempt at using a class instead of separate functions to represent two
    uncoupled spin-1/2 nuclei undergoing exchange.
    """
    # TODO: currently not seeing any advantage to this approach

    pi = np.pi
    pi_squared = pi ** 2

    def __init__(self, va=1, vb=0, k=0.01, wa=0.5, wb=0.5, percent_a=50):
        """
        Initialize the system with the required parameters:
        :param va: Frequency of nucleus a
        :param vb: Frequency of nucleus b (must be < va)
        :param k: Rate of nuclear exchange
        :param wa: With at half height for va signal at the slow exchange limit
        :param wb: With at half height for vb signal at the slow exchange limit
        :param percent_a: Fractional population of state 'a'
        """
        # Idea is to complete the frequency-independent calculations when the
        #  class is instantiated, and thus calculations may be faster.
        self.l_limit = vb - 50
        self.r_limit = va + 50

        T2a = 1 / (self.pi * wa)
        T2b = 1 / (self.pi * wb)
        pa = percent_a / 100
        pb = 1 - pa
        self.tau = pb / k
        dv = va - vb
        self.Dv = (va + vb) / 2
        self.P = self.tau * (1 / (T2a * T2b) + self.pi_squared * (dv ** 2)) \
            + (pa / T2a + pb / T2b)
        self.p = 1 + self.tau * ((pb / T2a) + (pa / T2b))
        self.Q = self.tau * (- self.pi * dv * (pa - pb))
        self.R = self.pi * dv * self.tau * ((1 / T2b) - (1 / T2a)) \
            + self.pi * dv * (pa - pb)
        self.r = 2 * self.pi * (1 + self.tau * ((1 / T2a) + (1 / T2b)))

    def intensity(self, v):
        """
        Yield a function for the lineshape for TwoSinglets
        :param v: frequency
        :return: a frequency-dependent function that returns the intensity of
        the spectrum at frequency v
        """
        # TODO: add to docstring
        p = self.p
        Dv = self.Dv
        P = self.P
        Q = self.Q
        R = self.R
        r = self.r
        tau = self.tau
        Dv -= v
        P -= tau * 4 * self.pi_squared * (Dv ** 2)
        Q += tau * 2 * self.pi * Dv
        R += Dv * r
        return (P * p + Q * R) / (P ** 2 + R ** 2)

    def spectrum(self):
        """
        Calculate a DNMR spectrum, using the parameters TwoSinglets was
        instantiated with.
        :return: a tuple of numpy arrays (x = numpy linspace representing
        frequencies, y = numpy array of intensities along those frequencies)pwd
        """
        x = np.linspace(self.l_limit, self.r_limit, 800)
        y = self.intensity(x)

        return x, y


def dnmr_AB(v, v1, v2, J, k, W):
    """
    A translation of the equation from Weil's JCE paper (NOTE: Reich pointed
    out that it has a sign typo!).
    p. 14, for the uncoupled 2-site exchange simulation.

    Parameters
    ---------

    v : float or array-like
        a frequency (x coordinate) or array of frequencies at which an
        amplitude (y coordinate) is to be calculated.
    float v1, v2 :
        frequencies of a and b nuclei (at the slow exchange limit,
        in the absence of coupling; `va` > `vb`)
    J : float
        the coupling constant between the two nuclei.
    k : float
        rate constant for state A--> state B
    W : float
        peak widths at half height (slow exchange limit).

    Returns
    -------
    float
        amplitude at frequency `v`.

    Notes
    -----

    References
    ----------
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
