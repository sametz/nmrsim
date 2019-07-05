"""This module provides classes to abstract NMR concepts such as spin systems
and spectra.
"""
from nmrtools.firstorder import first_order_spin_system
from nmrtools.math import reduce_peaks
from nmrtools.qm import spectrum

class SpinSystem:
    """Stub implementation of SpinSystem.
    Flesh out API (e.g. getters/setters; dunder methods) later."""
    def __init__(self, v, J, second_order=True):
        self.v = v
        self.J = J
        self.second_order = second_order
        self._peaklist = self.peaklist()
        print(self.v, self.J, self.second_order, self.peaklist)

    def peaklist(self):
        if self.second_order:
            return spectrum(self.v, self.J)
        else:
            return first_order_spin_system(self.v, self.J)


class Spectrum:
    """Stub implementation of Spectrum.
    Flesh out API (e.g. getter/setters; dunder methods) later."""
    def __init__(self, components, mhz=300, limits=None):
        self.components = components[:]
        self.mhz = mhz
        # TODO make the following more elegant. Maybe itertools.chain ?
        peaklists = [c.peaklist() for c in self.components]
        peaklists_flattened = [peak
                               for peaklist in peaklists
                               for peak in peaklist]
        self._peaklist = reduce_peaks(peaklists_flattened)

    def peaklist(self):
        return self._peaklist
