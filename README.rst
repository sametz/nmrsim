.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/sametz/nmrsim/master?filepath=jupyter

Click the "**Launch Binder**" link above to see how **nmrsim** can be used in Jupyter notebooks.

`Documentation on Read the Docs <https://nmrsim.readthedocs.io/>`_

nmrsim (version 0.4.0 beta)
============================

**nmrsim** is a Python library for the simulation of solution-state nuclear magnetic resonance (NMR) spectra.

The project is inspired by the program `WINDNMR <https://www.chem.wisc.edu/areas/reich/plt/windnmr.htm>`_ by Hans
Reich. The goal for Version 1.0 of **nmrsim** is to provide Python tools for the same types of simulations that
WINDNMR did--first- and second-order simulation of spin-1/2 spin systems, plus simulation of some dynamic NMR (DNMR)
lineshapes. A longer-term goal is to expand the toolset (e.g. to allow higher-spin nuclei, or new DNMR models).

Features
--------
* Class-based abstractions for NMR features (:code:`nmrsim.Multiplet`; :code:`nmrsim.SpinSystem`;
  :code:`nmrsim.Spectrum`).
* Convenience functions for quickly plotting simulation results (:code:`nmrsim.plt`).
* A lower-level API for more "hands-on" calculation of NMR spectra (e.g. functions for creating and solving spin
  Hamiltonians in :code:`nmrsim.qm`).

Examples
--------

See the Jupyter folder for examples of typical use cases.

Installation
------------

:code:`pip install nmrsim`

See the `Developer Page`_
for details on installing a developer version into a virtual environment.

.. _Developer Page: DEVELOPERS.rst

Contribute
----------

See the `Contributing to nmrsim`_ page for details
on how to contribute to the project.
You don't have to have any particular expertise!
If you've used (or tried to use) the library and have suggestions,
you can make a valuable contribution.

.. _Contributing to nmrsim: CONTRIBUTING.rst