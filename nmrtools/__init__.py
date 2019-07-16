"""
nmrtools
========

The nmrtools package provides tools for simulating nuclear magnetic resonance
(NMR) spectra.

The API is still in flux. Currently, it includes the following modules:

* dnmr: for modeling Dynamic NMR systems
* firstorder: for modeling first-order spectra
* math: core math routines for handling NMR data
* partial: uses non-quantum-mechanical solutions for common second-order
NMR patterns
* plt: convenience plotting routines for NMR results
* qm: quantum-mechanical second-order simulation of spin systems (currently
capped at 11 nuclei)

Currently, only spin-1/2 nuclei are accommodated.
"""

from ._classes import Multiplet, SpinSystem, Spectrum
