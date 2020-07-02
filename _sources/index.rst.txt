.. tf_codage documentation master file, created by
   sphinx-quickstart on Thu Jul  2 14:07:45 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

``tf_codage`` - deep learning for medical records!
==================================================

``tf_codage`` is a Python library with a set of tools and model
for treatment of medical records. In particular, it implements functions
for:

* encoding ICD-10 codes from hospital discharge records,
* predicting CCAM codes from surgery records,
* predicting severity level of disease according to french GHM scale.
  
Most models are implemented with TensorFlow, but it also contains 
more general functions that simplify data preprocessing and handling
Python/Jupyter
files.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation.rst
   api.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
