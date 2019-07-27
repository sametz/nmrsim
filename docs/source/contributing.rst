Contributing to **nmrsim**
--------------------------

While the project is alpha, the best way to contribute is to send feedback to
the author, either by opening an Issue or by email (sametz at udel dot edu).
This can be anything from a technical suggestion (NMR physics and code) to
documentation and user-friendliness (e.g. unclear descriptions, typos, awkward
namespace).

If you are familiar with virtual environments (e.g. with venv, or a conda
environment if you use a conda/Anaconda install), you can fork/clone/download
the repo; create your virtual environment; and then, from the base nmrsim
directory, run this from the command line:

.. code-block::

   pip install -e ".[dev]"

This will install nmrsim into your environment as well as developer tools. To
check that your installation is working, you can then run the tests from the
command line:

.. code-block::

   pytest tests

There will be some "graph spam" since some of the tests generate matplotlib
plots of spectra; just close the windows as they appear to move through the
tests.

If you are contributing to documentation, you can generate the html
documentation by navigating to the "docs" directory and running:

.. code-block::

   make html

By the time the project goes beta, there will be more detailed developer
instructions and a Code of Conduct.

Disclaimer: the author is neither an NMR spectroscopist, nor a software
engineer. I'm figuring this out as I go along. I welcome any constructive
criticism on any aspect of the project.