import numpy as np


class DnmrTwoSinglets:
    """
    A DNMR simulation for two uncoupled nuclei undergoing exchange.
    """

    _pi = np.pi
    _pi_squared = _pi ** 2

    def __init__(self, va=1, vb=0, k=0.01, wa=0.5, wb=0.5, pa=0.5):
        """
        Initialize the system with the required parameters:
        :param va: Frequency of nucleus a
        :param vb: Frequency of nucleus b (must be < va)
        :param k: Rate of nuclear exchange
        :param wa: With at half height for va signal at the slow exchange limit
        :param wb: With at half height for vb signal at the slow exchange limit
        :param pa: Fractional population of state 'a'
        """
        # Idea is to complete the frequency-independent calculations when the
        #  class is instantiated, and thus calculations may be faster.
        self._va = va
        self._vb = vb
        self._k = k
        self._wa = wa
        self._wb = wb
        self._pa = pa
        self.l_limit = vb - 50
        self.r_limit = va + 50

        self._set_T2a()
        self._set_T2b()
        self._set_pb()
        self._set_tau()
        self._set_dv()
        self._set_Dv()
        self._set_P()
        self._set_p()
        self._set_Q()
        self._set_R()
        self._set_r()

    def _set_T2a(self):
        self._T2a = 1 / (self._pi * self._wa)

    def _set_T2b(self):
        self._T2b = 1 / (self._pi * self._wb)

    def _set_pb(self):
        self._pb = 1 - self._pa

    def _set_tau(self):
        self._tau = self._pb / self._k

    def _set_dv(self):
        self._dv = self._va - self._vb

    def _set_Dv(self):
        self._Dv = (self._va + self._vb) / 2

    def _set_P(self):
        self._P = self._tau * (1 / (self._T2a * self._T2b) + self._pi_squared * (self._dv ** 2)) \
                  + (self._pa / self._T2a + self._pb / self._T2b)

    def _set_p(self):
        self._p = 1 + self._tau * ((self._pb / self._T2a) + (self._pa / self._T2b))

    def _set_Q(self):
        self._Q = self._tau * (- self._pi * self._dv * (self._pa - self._pb))

    def _set_R(self):
        self._R = self._pi * self._dv * self._tau * ((1 / self._T2b) - (1 / self._T2a)) \
                  + self._pi * self._dv * (self._pa - self._pb)

    def _set_r(self):
        self._r = 2 * self._pi * (1 + self._tau * ((1 / self._T2a) + (1 / self._T2b)))

    @property
    def va(self):
        return self._va

    @va.setter
    def va(self, value):
        self._va = value
        self._set_vab_dependencies()

    def _set_vab_dependencies(self):
        self._set_dv()
        self._set_Dv()
        self._set_P()
        self._set_Q()
        self._set_R()

    @property
    def vb(self):
        return self._vb

    @vb.setter
    def vb(self, value):
        self._vb = value
        self._set_vab_dependencies()

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, value):
        self._k = value
        self._set_tau()
        self._set_p()
        self._set_P()
        self._set_Q()
        self._set_R()
        self._set_r()

    @property
    def wa(self):
        return self._wa

    @wa.setter
    def wa(self, value):
        self._wa = value
        self._set_T2a()
        self._set_wab_dependencies()

    def _set_wab_dependencies(self):
        self._set_p()
        self._set_P()
        self._set_R()
        self._set_r()

    @property
    def wb(self):
        return self._wb

    @wb.setter
    def wb(self, value):
        self._wb = value
        self._set_T2b()
        self._set_wab_dependencies()

    @property
    def pa(self):
        return self._pa

    @pa.setter
    def pa(self, value):
        self._pa = value
        self._set_pb()
        self._set_tau()
        self._set_p()
        self._set_P()
        self._set_Q()
        self._set_R()
        self._set_r()

    def intensity(self, v):
        """
        Yield a function for the lineshape for TwoSinglets
        :param v: frequency
        :return: a frequency-dependent function that returns the intensity of
        the spectrum at frequency v
        """
        # TODO: add to docstring
        p = self._p
        Dv = self._Dv
        P = self._P
        Q = self._Q
        R = self._R
        r = self._r
        tau = self._tau
        Dv -= v
        P -= tau * 4 * self._pi_squared * (Dv ** 2)
        Q += tau * 2 * self._pi * Dv
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
