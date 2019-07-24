.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/sametz/nmrtools/cleanup?filepath=jupyter

Click the "**Launch Binder**" link above to see how **nmrtools** can be used in Jupyter notebooks.

nmrtools (version 0.2 alpha)
============================

**nmrtools** is a Python library for the simulation of solution-state nuclear magnetic resonance (NMR) spectra.

The project is inspired by the program `WINDNMR <https://www.chem.wisc.edu/areas/reich/plt/windnmr.htm>`_ by Hans
Reich. The goal for Version 1.0 of **nmrtools** is to provide Python tools for the same types of simulations that
WINDNMR did--first- and second-order simulation of spin-1/2 spin systems, plus simulation of some dynamic NMR (DNMR)
lineshapes. A longer-term goal is to expand the toolset (e.g. to allow higher-spin nuclei, or new DNMR models).

Features
--------
* Class-based abstractions for NMR features (:code:`nmrtools.Multiplet`; :code:`nmrtools.SpinSystem`;
  :code:`nmrtools.Spectrum`).
* Convenience functions for quickly plotting simulation results (:code:`nmrtools.plt`).
* A lower-level API for more "hands-on" calculation of NMR spectra (e.g. functions for creating and solving spin
  Hamiltonians in :code:`nmrtools.qm`).

Installation
------------
Currently, the project is an alpha version, meaning that the files must be downloaded to a user's code directory and
used there.

When the project becomes beta, there will be instructions to :code:`pip install` the code from the downloaded code. A
"release candidate" (-rc) version will mean that the package can be installed directly from the Python Package Index
(PyPI) via :code:`pip install`.

Contribute
----------
While the project is alpha, the best way to contribute is to send feedback to the author, either by opening an Issue
or by email (sametz at udel dot edu). This can be anything from a technical suggestion (NMR physics and code) to
documentation and user-friendliness (e.g. unclear descriptions, typos, awkward namespace).

By the time the project goes beta, there will be developer instructions and a Code of Conduct.
