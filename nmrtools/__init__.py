"""
nmrtools
========

The nmrtools package provides tools for simulating nuclear magnetic resonance
(NMR) spectra.

The overall API has not been settled on. Currently, there are two modules:

* nmrmath: provides functions for calculating spectral parameters
* nmrplot: provides functions for converting calculation results to lineshapes
  and plotting the results.

TODO: Elaborate.
"""
from . import nmrmath
from . import nmrplot
