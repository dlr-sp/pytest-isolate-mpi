Contributing
============

Thank you for partaking in the development and improvement of
``pytest-mpi-isolate``. Nevertheless, there are some rules we would like
you to follow.  These rules shall keep the source code in a readable,
understandable, well tested and documented and therefore maintainable
state. 

The rules are not defined to be broken or circumvented. That said, keep
in mind the objective you try to accomplish. A breach of the rules might
be necessary on rare occasions. When this occurs, it shall be
well-founded and documented. A personal aversion against the rules is
not a reason to deviate.

The rules will be detailed below.

.. contents:: table of contents
   :local:


Code Guidelines
---------------

Programming Language
~~~~~~~~~~~~~~~~~~~~

The programming language of the project is python. The oldest still
supported version of python is defined in the file ``pyproject.toml``.


Programming Style
~~~~~~~~~~~~~~~~~

All source code shall be formatted according to the `PEP 8`_ rules.

.. _PEP 8: https://python.org/dev/peps/pep-0008

Additionally, the following rules or deviations from the PEP 8 style
apply: 


Maximum line length
    In code files, limit the maximum line length to 120 characters,
    except for comment lines which are limited at 72 characters. For
    reStructuredText files the limit is 72 characters.

Imports
    Only one symbol shall be imported per line.  

Type hints
    New or updated APIs (functions, methods, attributes, etc.) shall be
    annotated with type hints. 


Testing
~~~~~~~

``pytest-isolate-mpi`` has extensive automated test suite for quality assurance. To
facilitate easy, small and readable tests,  the
`Pytest`_ framework. 

.. _Pytest: https://docs.pytest.org

All new and changed code is expected to be covered by the test suite, at
the minimum, by at least one of the system tests and better by one or
more unit tests. Do not only test for the `happy path`_. Also add tests
for potential exceptions and error conditions, e.g. IO errors or invalid
input.

.. _happy path: https://en.wikipedia.org/wiki/happy_path
 

Documentation
~~~~~~~~~~~~~

From an user's perspective, ``pytest-isolate-mpi`` shall be fully documented. Any
new feature or changed behavior must be reflected in the documentation.

``pytest-isolate-mpi`` uses sphinx to generate html pages from `reStructuredText`_
files in ``docs`` directory and `Python docstrings`_.

.. _reStructuredText: http://www.sphinx-doc.org/en/stable/rest.html
.. _Python docstrings: https://www.python.org/dev/peps/pep-0257/

Any user-relevant change is also to be recorded in changelog located in
``CHANGES.rst``.

The in-code documentation encompasses the bulk of the developer
documentation. It should explain how a given module, class, methods, or
function can be used. Therefore, document the API provided by a module/
package and a *short* description of its functionality.

The Python docstrings are formatted in the pleasantly compact `Google
style`_. Types shall be documented with `Python type hints`_ and
no longer in the docstrings.

.. _Google style: https://sphinxcontrib-napoleon.readthedocs.io/en/
   latest/sphinxcontrib.napoleon.html
.. _Python type hints: https://docs.python.org/3/library/typing.html


Changing Existing Code — Dealing With Legacy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `boy scout rule`_ applies.  Whenever you change an existing piece of
source code, please also pay attention, that all the rules described in
this document are followed. If you find any deviation, it is now up to
you to fix it. 

.. _boy scout rule: https://clean-code-developer.com/grades/
    igrade-1-red/#boy_scout_rule


If there are no tests, unit tests or other, please add them. If there is
no documentation or there are parts missing, please add it. If there is
a deviation from the style guide, please correct it...

We are aware that this will cause some overhead for everyone involved in
the code development at the beginning, but it is considered to be of
great value in the long run as we get the whole code documented and
under test coverage bit by bit, making life easier for everybody.

Please leave the code always in a better state than you found it.


Development Workflow
--------------------

The following sections give a short overview how the standard development
workflow should be conducted.

GitHub Project
~~~~~~~~~~~~~~

All project activity is centered around the ``pytest-isolate-mpi``
`GitHub project`_.

.. _GitHub project: https://github.com/dlr-sp/pytest-isolate-mpi

The main branch shall always be held in release quality.  Thus, it
would be inadvisable to allow to push any piece of code directly into
the main branch. Such action is also suppressed technically. The
workflow *feature branch and pull request* is to be used. To get your
development accepted into the main branch, please create a pull
request as described below.


Adding Features and Fixing Bugs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Always reference to an open an issue when opening a branch. This will
help with the provenience of modifications and additions to the source
code in the long run. If there is no issue yet, create one.  

Create a separate branch for every bug fix or feature you would like to
implement. Please make sure to use the naming convention and a telling
name for the branches you create, so that they may be found and sorted
out easily:

* *for bugfixes:* ``fix/descriptive_name``
* *for features:* ``feature/descriptive_name``
* *for maintenance:* ``maintenance/descriptive_name``


Commiting Changes
~~~~~~~~~~~~~~~~~

Please make small commits, as a general rule: one commit equals one
change. Commit and push working code only. 

Commits shall have meaningful commit messages which should mainly explain
the *reasoning* leading the change in the commit. Commit messages have
at least a title line limited to 50 characters. It may be optionally
followed by blank line and additional paragraphs of explanation or
context. For these, a line limit length of 72 characters applies. Commit
titles use `title case`_ in the AP style.

.. _title case: https://titlecaseconverter.com/


Pull Request and Code Review
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before a change to the code is accepted into the main branch, a review
will take place. These reviews are supposed to be reviews by peer. 

Once the work in topic branch is ready to be included in main, create
a Pull Request on GitHub and assign another developer as Reviewer. The
reviewer will evaluate the proposed changes to certain criteria:

* The code shall run without errors and the tests shall pass.
* The code shall be correct, it shall do, what it is supposed to.
* There shall be a certain technical quality w.r.t. logic, naming 
  convention, ...
* The code shall be reusable, there shall be no duplications.
* The code shall handle exceptions and errors.
* The code shall be documented.
* The code shall contain all relevant tests.
* The code shall follow the style guide.

The developer of the branch is expected to update the merge request with
new commits until the review criteria are met. 

The overall review goal is to ensure merges improve the overall quality
of ``pytest-isolate-mpi``. Sometimes it is now possible to make changes
which meet all review criteria fully without enlarging the scope of the
change to the point of impracticability. In such cases, the Merge
Request should be accepted as soon as it improves ``pytest-isolate-mpi``
in some traceable manner.

Mind our manners! If there is criticism, stay fair and use constructive
criticism.


Howtos
------

How to Prepare a New Development Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to start developing on ``pytest-isolate-mpi``, one has to clone
repository, setup ``pytest-isolate-mpi`` with its dependencies, and optionally
install the Sertlib tools for the built-in examples:

1.  Obtain the source code::

        git clone git@github.com:dlr-sp/pytest-isolate-mpi.git
        cd pytest-isolate-mpi

2.  Install ``pytest-isolate-mpi`` in a new Python |venv|_::

        python3 -m venv venv
        source venv/bin/activate
        pip install -e ".[dev]"

    
    Using ``venv`` is not strictly necessary, but recommended to isolate
    MDO Diver's dependencies from the ones of other projects. In the
    ``pip`` step, the development dependencies to run the tests and
    generate the HTML documentation are installed as well. The
    constraints file tells ``pip`` where to find the ``slb`` requirement
    of ``pytest-isolate-mpi``.  By passing the flag ``-e`` the package
    is installed in editable mode. Changes to the source code don't have
    to be installed to become effective when running
    ``venv/bin/mdo-driver``.

.. |venv| replace:: ``venv``
.. _venv: https://docs.python.org/3/library/venv.html

The other how-tos in this section assume you have completed the full MDO
Diver installation and you using a shell with the MDO Diver ``venv``
activated.


How to Run Pre-Commit Checks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

While all tests are executed in via CI pipeline, it is advisable to run
these checks before committing changes to avoid unnecessary follow-up
commits polluting the commit history with noise.

To run these checks, run the following commands in the root folder of
the repository:

1.  Running the test suite::

        make tests

2.  Validating source code style::

        make lint

3.  Validating the source distribution files (``MANIFEST.in``)::

        check-manifest


How to Generate the Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``pytest-isolate-mpi`` uses Sphinx for its documentation.

To generate help documents in HTML format, run::

  make docs

Sphinx stores the HTML documentation files in ``docs/_build/html``.

