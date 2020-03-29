##########
Change Log
##########

All notable changes to this project will be documented in this file.

The format is inspired by
`Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_
and tries to adhere to `Semantic Versioning <http://semver.org>`_
as well as `PEP 440 <https://www.python.org/dev/peps/pep-0440/>`_.

Working towards a Version 1.0.0 release,
the author interprets the terms below as follows:

* **pre-alpha status**:
  Working code,
  but every aspect of the project is in flux,
  including API and documentation.

* **alpha status**:
  The "broad strokes" of an API are in place.
  Code testing and optimization, API adjustments,
  addition of new features and tests,
  and creating packaging requirements will be ongoing.
  The package can be installed via setup.py.

* **beta status**:
  All anticipated Version 1.0.0 features are implemented and documented.
  The package can be
  'pip install'ed via TestPyPI and possibly PyPI.

* **release candidate status**:
  The package passes tests on multiple platforms and python/dependency versions.
  The package can be pip installed from PyPI.

* **Version 1.0.0 release**:
  API is stable.
  The package is available on PyPI (and perhaps conda).


0.4.0 - 2020-xx-xx (beta release)
---------------------------------
Changed
^^^^^^^
* nmrsim.qm now uses importlib.resources.path to find the nmrsim/bin folder,
  instead of relying on the use of __file__.
  For users, this means that if your application uses nmrsim as a dependency,
  and you want to freeze the application (e.g. with PyInstaller or PyOxidizer),
  the frozen app should now find the bin folder and contents.

Fixed
^^^^^
* Documentation errors (Issues #2, #3, #4)

0.3.0 - 2019-11-08 (beta release)
---------------------------------
Added
^^^^^
* nmrsim.Multiplet:
    * :code:`w` attribute added (peak width at half height).
* nmrsim.SpinSystem:
    * :code:`w` attribute added (peak width at half height).
* nmrsim.Spectrum:
    * In-place addition modifies the Spectrum object in-place.
    * :code:`vmin` and :code:`vmax` attributes added, to set spectral width.
    * :code:`default_limits` method added, to reset spectral width to default.
    * :code:`lineshape()` method added, to return lineshape data for the
      spectrum.

Changed
^^^^^^^
* nmrsim.firstorder.multiplet now returns a *sorted* peaklist.

Fixed
^^^^^
* nmrsim.Spectum:
    * Addition of Spectrum objects is commutative, and returns a new Spectrum
      object.


0.2.1 - 2019-07-26 (alpha release)
----------------------------------
* Binaries for accelerating qm calculations packaged in distribution.


0.2.0 - 2019-07-25 (alpha release)
----------------------------------
Radical revisions (including a renaming of the project from "nmrtools" to "nmrsim") and migration to the
sametz/nmrsim repo on GitHub). Includes:

- New API
- Faster second-order calculations using sparse matrices and cached partial solutions
- High-level class-based API for convenient simulation and plotting of NMR spectra and components
- Jupyter notebooks demonstrating key API features and mathematical underpinnings


0.1.0 - 2018-08-07 (pre-alpha release)
--------------------------------------

Initial Commit
