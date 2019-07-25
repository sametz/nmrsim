"""
nmrsim
========

The nmrsim package provides tools for simulating nuclear magnetic resonance
(NMR) spectra.

The API is still in flux. Currently, it includes the following modules:

* dnmr: for modeling Dynamic NMR systems
* firstorder: for modeling first-order spectra
* math: core math routines for handling NMR data
* partial: uses non-quantum-mechanical solutions for common second-order NMR patterns
* plt: convenience plotting routines for NMR results
* qm: quantum-mechanical second-order simulation of spin systems (currently capped at 11 nuclei)

Currently, only spin-1/2 nuclei are accommodated.

The top-level nmrsim namespace provides the following classes:

* Multiplet: a representation of a first-order multiplet (e.g. quartet; doublet of triplets).
* SpinSystem: a representation of a set of coupled nuclei (modeled as either first-order or second-order).
* Spectrum: a collection of components such as Multiplets or SpinSystems that contribute to a total NMR spectrum
  simulation.

"""
# TODO: consider including a glossary here? e.g. peaklist, spectrum

from ._classes import Multiplet, SpinSystem, Spectrum
