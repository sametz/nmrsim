##########
Change Log
##########

All notable changes to this project will be documented in this file.

The format is inspired by `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_ and tries to adhere to `Semantic Versioning <http://semver.org>`_ as well as `PEP 440 <https://www.python.org/dev/peps/pep-0440/>`_.

Working towards a Version 1.0.0 release, the author interprets the terms below as follows:

* **pre-alpha status**: Working code, but every aspect of the project is in flux, including API and documentation.

* **alpha status**: The "broad strokes" of an API are in place. Code testing and optimization, API adjustments,
  addition of new features and tests, and creating packaging requirements will be ongoing. The package can be
  installed via setup.py.

* **beta status**: All anticipated Version 1.0.0 features are implemented and documented. The package can be
  'pip install'ed via TestPyPI or possibly PyPI.

* **release candidate status**: The package passes tests on multiple platforms. The package can be deployed on PyPI
  and installed with pip.

* **Version 1.0.0 release**: API is stable. The package is available on PyPI (and perhaps conda).


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
