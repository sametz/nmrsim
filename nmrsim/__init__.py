"""
nmrsim
======

The nmrsim package provides tools for simulating nuclear magnetic resonance
(NMR) spectra.

The API is still in flux. Currently, it includes the following modules:

* dnmr: for modeling Dynamic NMR systems
* firstorder: for modeling first-order spectra
* math: core math routines for handling NMR data
* partial: uses non-quantum-mechanical solutions for common second-order NMR
  patterns
* plt: convenience plotting routines for NMR results
* qm: quantum-mechanical second-order simulation of spin systems (currently
  capped at 11 nuclei)

Currently, only spin-1/2 nuclei are accommodated.

The top-level nmrsim namespace provides the following classes:

* Multiplet: a representation of a first-order multiplet (e.g. quartet;
  doublet of triplets).
* SpinSystem: a representation of a set of coupled nuclei (modeled as either
  first-order or second-order).
* Spectrum: a collection of components such as Multiplets or SpinSystems that
  contribute to a total NMR spectrum simulation.

Definitions of Terms Used
-------------------------
In naming classes, functions, methods, data types etc. certain phrases, taken from NMR nomenclature, have the following
interpretations:

* **multiplet** (e.g. the `nmrsim.Multiplet` class): a first-order simulation
  for one signal (i.e. one or more chemical shift-equivalent nuclei).
  Examples: doublet, triplet, doublet of triplets, but **not** an AB quartet
  (which is a second-order pattern for two nuclei).
* **spin system** (e.g. the `SpinSystem` class): a simulation of a set of
  coupled nuclei.
* **spectrum** (e.g. the `Spectrum` class): a complete collection of first-
  and/or second-order components for simulating a total NMR spectrum.
  'Spectrum' can also refer in general to the simulation results for the
  system, e.g a peaklist or lineshape (see below).
* **peak**: a pair of frequency (Hz), intensity values corresponding to a
  resonance in an NMR spectrum. For example, a 1H triplet centered at 100 Hz
  with J = 10 Hz would have the following  peaks:
  (110, 0.25), (100, 0.5), (90, 0.25).
* **peaklist**: a list of peaks
  (e.g. [(110, 0.25), (100, 0.5), (90, 0.25)] for the above triplet).
* **lineshape**: a pair of [x coordinates...], [y coordinates] arrays for
  plotting the lineshape of a spectrum.

The following idioms are used for arguments:
* **v** for a frequency or list of frequencies (similar to the Greek lowercase
"nu" character).
* **I** for a signal intensity  (despite being a PEP8 naming violation)
* **J** for coupling constant data (exact format depends on the
implementation).

"""

from ._classes import Multiplet, SpinSystem, Spectrum  # noqa: F401

__version__ = '0.5.2'
