.. _developers-guide:

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

You can activate the environment with:

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
Note that Windows 10  users can now get Python 3.7+ via the Microsoft store.

If you wish to deactivate the venv at any point,
enter :code:`deactivate` from the command line.

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

If you later want to exit this environment,
you can activate another environment,
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
From the command line, in the top nmrsim directory that contains setup.py,
enter:

.. code-block:: bash

   pip install -e ".[dev]"

To check your installation, run the tests using pytest,
then navigate to the `docs` directory and build the documentation:

.. code-block:: bash

   pytest
   cd docs
   make html

There will be several pop-up plots that are visual tests for correct behavior;
close these windows as they pop up to proceed through the tests.

As an extra test of a correct installation,
you can deliberately break some of the code you're working on
and re-run the tests to see the tests fail
(assuming the code was covered by the tests).

Making a contribution
---------------------

Create a git branch with a descriptive name for your contribution, e.g.

.. code-block:: bash

   git checkout -b add_dnmr_tutorial

Make your changes, and then:

.. code-block:: bash

   pytest
   flake8

When these tests both pass, navigate to the docs directory,
and build the html documentation:

.. code-block:: bash

   make html

Open the docs/build/html/index.html page in your browser.
If you made changes to the documentation, including public docstrings,
navigate to where the change should appear and check that it looks OK.
After you're done with the documentation, run:

.. code-block:: bash

   make clean

to delete the contents of the build directory prior to publishing your work.

Commit and push to your fork of nmrsim:

.. code-block:: bash

   git status  # check that your work is staged to commit
   git commit -m "Brief description of the change you made"
   git push

Submit a pull request
---------------------

`See the GitHub Help on creating a pull request from a fork
<https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork>`_.

Pull requests should be made to nmrsim's 'develop' branch,
and not directly to 'master'.

From your GitHub page for your fork,
select the name of your working branch from the 'branch' drop-down menu
(e.g. "add_dnmr_tutorial" using the above example).
Click 'New pull request'.

You should check that 'base repository' is 'sametz/nmrsim',
'base' is 'develop',
'head repository' is 'yourGitHubname/nmrsim',
and 'compare' is your branch name (e.g. 'add_dnmr_tutorial').
Check that you have a commit message
(a longer message in the "Leave a comment" text field is optional)
and click "Create pull request" when ready.

The package maintainer will respond via GitHub notification.
If there is no response after a week, feel free to email them
(sametz at udel dot edu) with 'nmrsim' somewhere in the subject line... they
may be busy, on vacation or just distracted :) but will eventually respond.

Code Style and Conventions
--------------------------

If your code is passing the flake8 test,
and if the html documentation looks OK, then it should be acceptable. Here are
some of the guidelines:

PEP 8
^^^^^

`PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_
(the Python style guide) is followed, with the following exceptions:

* The max line length is 119, the width of a GitHub preview.
  This can be exceeded with good reason. The PEP 8 guideline of 79 characters
  is a good goal, but readability (e.g. splitting up long URLs) shouldn't be
  sacrificed.
* Some naming conventions are violated for consistency with NMR terms
  and with pre-existing code.
  For example, many variables are upper-case single letters,
  including H for Hamiltonian, J for coupling constant(s),
  and (scandalously) I (upper-case 'i') for signal intensity.

The project's .flake8 file makes accomodations for these and other exceptions.

import sorting
^^^^^^^^^^^^^^
imports should be sorted into three categories,
with a blank line separating the categories:

* standard library
* third-party libraries
* nmrsim modules

Within each, they should be sorted alphabetically (ignoring "from").

type annotations
^^^^^^^^^^^^^^^^

We currently don't use type annotations,
because this is difficult to implement with numpy and related packages.


documentation
^^^^^^^^^^^^^

The project follows `PEP 257's guidelines
<https://www.python.org/dev/peps/pep-0257/>`_ for docstrings,
and adopts `Numpy-style docstrings
<https://numpydoc.readthedocs.io/en/latest/format.html>`_.

Docstrings are only required for public classes and functions
(i.e. not for those whose name begins in a single underscore,
e.g. _normalize).
However, you may document private classes and functions if you wish--
it can make the code's purpose clearer to others,
and it's possible that private code may at some point be 'promoted' to the
public API.

Currently, "test docstrings" are not used. If you think they should, feel free
to make a case for them.

The nmrsim project uses Sphinx for documentation,
and restructuredtext (.rst) for content.
`Semantic line breaks <https://sembr.org/>`_ are encouraged--
they make editing and formatting easier.






