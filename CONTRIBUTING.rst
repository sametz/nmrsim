Contributing to nmrsim
======================

`Code of Conduct <https://github.com/sametz/nmrsim/blob/master/CODE_OF_CONDUCT.md>`_

You don't need to be an experienced programmer, or NMR spectroscopist,
to contribute to nmrsim--
the creator of the project is neither!
Here are some ways that you can contribute to the project:

Use the library, and give feedback
----------------------------------

Prior to the release of a Version 1,
the easiest way to contribute is to use the library
and give your feedback.
If you are a GitHub user, you can open an issue;
if not, you can email the creator: sametz at udel dot edu.

Feedback includes bug reports, feature suggestions and such,
but in particular feedback on *how* you use nmrsim
(or what *keeps you* from using nmrsim) is valuable.
Is the API awkward, non-intuitive, or un-Pythonic?
Is the documentation for a feature unclear (or even missing)?

It is a best practice
(see: `Semantic Versioning <https://semver.org/>`_)
to not break backwards compatability within a major version,
because it also breaks user trust.
A Version 1 release is a promise
that no updates will break your programs that use Version 1,
but a future Version 2 may break things.
User feedback prior to a Version 1 release can help create a user-friendly library API
before it is "set in stone".

Lend Expertise
--------------

If you have experience in NMR spectroscopy,
but not necessarily programming,
you may be able to describe how the library can be improved.
Could the calculations be faster?
Are there other models (such as other DNMR systems) that can be added?
Much of the work that has gone into nmrsim involved creating
and then speed optimizing,
the quantum-mechanical calculations for second-order systems
(:code:`nmrsim.qm`).
There is probably room for improvement here.

If you have experience with Python projects,
but not necessarily with NMR spectroscopy,
you may help create a "professional" library.
The creator of the project is learning Python best practices "on the fly".
This includes:

* testing (pytest),
* documentation (Sphinx),
* python packaging,
* automation (e.g. tox or nox),
* continuous integration (CI)

Also, Pythonistas can point out any "broken windows" the project may have
(bad docstrings, anti-patterns, etc.)

Become a Developer
------------------

You can also contribute by forking the project
and making a pull request.
Contributions to the :code:`nmrsim` core code are welcome,
but you can also contribute to:

* tests (we aim for 100% code coverage)
* documentation
* tutorials
* jupyter notebooks

To get started, see the
`Developer Page`_.

.. _Developer Page: DEVELOPERS.rst
