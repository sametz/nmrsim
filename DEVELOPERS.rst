Developers Guide
================

`Code of Conduct <https://github.com/sametz/nmrsim/blob/master/CODE_OF_CONDUCT.md>`_

This is a basic guide to setting up a development environment on your computer,
to get you up and running with the nmrsim code base.
It also provides brief guidelines for how to make a pull request.
If you would like more information or guidance,
or if these instructions are not working for you,
you can create an Issue on GitHub
and this guide can be improved.

.. contents::

Creating a Development Environment
----------------------------------

Set up git and GitHub
^^^^^^^^^^^^^^^^^^^^^
If you are completely new to Github,
you may have to set up an account and install git.
See the
`GitHub Help on Git Setup
<https://help.github.com/en/github/getting-started-with-github/set-up-git>`_
for more information.

Forking the repository
^^^^^^^^^^^^^^^^^^^^^^
Navigate to the nmrsim GitHub page,
and click the "Fork" icon in the upper right.
This will create your own copy of the repository
that you can feel free to alter.

See the related
`GitHub Help on Forking a Repo
<https://help.github.com/en/github/getting-started-with-github/fork-a-repo>`_
for more info.

Cloning the repository
^^^^^^^^^^^^^^^^^^^^^^
Using a terminal (e.g. Command Prompt on Windows; // on Mac),
navigate to the directory where you would like to create the nmrsim project folder,
then enter:

.. code-block:: bash

   git clone https://github.com/your_GitHub_name_here/nmrsim.git
   cd nmrsim

The URL can be obtained from your fork's GitHub page
by clicking the "Clone or Download" button and copying the URL.

See the `GitHub Help on Forking a Repo`_ for more information.

Creating the virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For any Python project,
you don't want to install into your operating system's Python.
Instead, you should create a custom Python environment
and install nmrsim plus dependencies there.

This can be tricky,
and especially for Windows users may pose a significant barrier to entry.
The instructions below have been tested for Mac and for Windows 10 machines
that had an Anaconda install (and thus Python 3.6+).
Linux users should be able to follow the Mac instructions,
but can give feedback if they have trouble developing on a Linux machine.
The examples use Python 3.7, but nmrsim should be compatible for 3.6+
(i.e. since the introduction of f-string syntax).

Using venv
^^^^^^^^^^
If your system already has a python version of 3.6 or higher,
you can create a virtual environment from the command line with:

.. code-block:: bash

   python -m venv env

This creates an 'env' folder with the python environment.
Note that on your system you may need to use 'python3', 'python3.7' etc.
instead of 'python' if you have more than one version of python installed
(e.g. 'python3' if 'python' refers to version 2.7).

On Mac, you can activate the environment with:

.. code-block:: bash

   source env/bin/activate (Mac)
   env\Scripts\activate.bat (Windows)

NOTE: If you have an Anaconda install, and try to run the tests, it may fail.
If you see two indicators for the environment in parentheses in the terminal,
e.g.:

.. code-block:: bash

   (env) (base) ...$

enter :code:`conda deactivate`
to make sure the conda environment isn't superseding the venv environment.
You should see the (base) indicator disappear.

If your system does *not* have a Python version 3.6+ already installed,
or if you want to have more than one version of Python on your system,
look into the pyenv (Mac/Linux) or pyenv-win (Windows) libraries.

source venv/bin/activate (linux/macOS), or
venv\Scripts\activate.bat (Windows CLI)
pip install -e ".[dev]"
then test
deactivate  # when done

Using conda
^^^^^^^^^^^
It's common for scientists to use an Anaconda or miniconda installation
to manage their Python (or other software) dependencies.
However, you will be installing packages in development mode with `pip`,
and sometimes there are conflicts with pip- and conda-installed packages.
This may work on your system,
but if there are problems with package conflicts
it may be best to use the venv option.

For example, the instructions worked on a 2019 Macbook Pro,
but failed on a Windows 10 machine
(despite the conda 3.7 environment being activated,
the system Python 3.5 installation was still used to run pytest,
causing any code with an f-string to fail).

To create a new Python environment named "nmrsim" and activate it,
use the command line:

.. code-block:: bash

   conda create --name nmrsim python=3.7
   conda activate nmrsim (Mac)
   activate nmrsim (Windows)

If you later want to exit this environment, you can activate another environment,
or enter:

.. code-block:: bash

   conda deactivate (Mac)
   deactivate (Windows)

Installing nmrsim in developer mode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you were to just install nmrsim directly from PyPI (by "pip install"),
or if you just ran `setup.py`,
the current nmrsim would be installed into your python environment *immutably*.
Any changes you made to the code would not be noticed by you or the tests.
Instead, you will install the package in "developer mode".
This will install nmrsim, plus dependencies.
It will also install the developer dependencies,
which are not required by casual nmrsim users,
but are required for developers to run tests, check formatting and so on.
From the command line, in the top nmrsim directory that contains setup.py, enter:

.. code-block:: bash

   pip install -e ".[dev]"

To check your installation, run the tests using pytest:

.. code-block:: bash

   pytest

There will be several pop-up plots that are visual tests for correct behavior;
close these windows as they pop up to proceed through the tests.

As an extra test of a correct installation,
you can deliberately break some of the code you're working on
and re-run the tests to see the tests fail
(assuming the code was covered by the tests).

Submitting a Pull Request
-------------------------

branch
^^^^^^

checklist
^^^^^^^^^

* flake8
* run tests
* build docs
* submit

Code Style and Conventions
--------------------------

PEP8
^^^^

docstrings
^^^^^^^^^^

documentation
^^^^^^^^^^^^^
semantic line wraps





