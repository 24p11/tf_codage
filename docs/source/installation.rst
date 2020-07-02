############
Installation
############

This package is still work in progress and we don't release new versions in the package repositories.


Sandboxing
==========

We recommend installing the library in a sandboxed environment, such as `conda` or `venv`.

To create a new environment with `conda`::

    $ conda env create -n tf_codage_dev
    $ conda activate tf_codage_dev
    $ python -m ipykernel install --user --name tf_codage_dev

The last step is required if you want to use ``tf_codage`` from Jupyter notebook.
It installs a new kernel called ``tf_codage_dev`` that directly connects to the
environment.

Installing
==========

You can install this library using ``pip``::

    $ git https://github.com/24p11/automate_pmsi.git
    $ cd automate_pmsi
    $ pip install -e .

It will also install the requirements of the library.

Requirements
============

The requirements are automatically installed by ``pip``. The most important ones are:

* Python >= 3.7.0
* tensorflow >= 2.0
* scikit-learn >= 0.21.3
* click
* and others

Testing
=======

The tests are located in the ``tests`` sub-directory. To run them, you can use ``pytest``::

    $ pip install pytest
    $ pytest tests
