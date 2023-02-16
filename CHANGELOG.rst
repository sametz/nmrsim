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

* **beta status**:
  All anticipated Version 1.0.0 features are implemented and documented.
  However, breaking changes to the API are still possible.
  The package can be pip-installed.

* **release candidate status**:
  No further changes to the API are expected before the next release.

* **Version 1.0.0 release**:
  API is stable.
  The package is available on PyPI (and perhaps conda).

0.6.0 - 2023-02-16 (beta release)
---------------------------------
Changed
^^^^^^^
* Python 3.6 reached its End of Life, and is no longer supported.
* Requirement for the :code:`sparse` library is now<=v0.10.0 *or* >=0.13.0.
  The library maintainers introduced bugs in :code:`sparse 0.11` that broke :code:`nmrsim`,
  but they were fixed in :code:`0.13`.

Added
^^^^^
* nmrsim.plt: the plot functions have the optional boolean argument :code:`hidden`.
  The default :code:`hidden=False` retains previous behavior,
  where a plot is displayed when possible.
  When :code:`hidden=True`, the :code:`.show()` method is not executed,
  and the plot is not displayed.
  The main use for this is to allow automated tests to run in CI without hanging.

0.5.2 - 2021-07-21 (beta release)
---------------------------------
Changed
^^^^^^^
* Requirement for the `sparse` library capped at v0.10.0.
  Version 0.11.0+ results in numba errors.
  See `sparse issue #499 <https://github.com/pydata/sparse/issues/499>`_.

0.5.1 - 2020-05-10 (beta release)
---------------------------------
Fixed
^^^^^
* unneccessary print statements removed rom qm.py (Issue #5)

0.5.0 - 2020-03-29 (beta release)
---------------------------------
Changed
^^^^^^^
* add_lorentzians moved from nmr.plt to nmr.math.
  Besides being more appropriate here,
  it fixes a problem where importing from nmrsim.plt also requires importing
  from matplotlib/Tkinter.
  Some environments
  (some Unix systems; BeeWare's Briefcase packaging tool)
  do not have Tkinter available.
* nmrsim.plt will substitute the matplotlib "Agg" backend for "TkAgg"
  if tkinter is not found on the users system, and print a warning message
  that plots will not be visible.

0.4.0 - 2020-02-22 (beta release)
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
