#################
``tf_codage`` API
#################

You will find here the documentation of all modules within the ``tf_codage`` package. This documentation is automatically generated from the docstring in the code and can be also found using the `help` command or the quick-help Jupyter magic ``command?``.

:mod:`tf_codage.models` - base deep learning models
===================================================

.. automodule:: tf_codage.models

:mod:`tf_codage.models.common` -- long-text transformers models
---------------------------------------------------------------

.. automodule:: tf_codage.models.common

.. currentmodule:: tf_codage

.. autosummary::
   
   models.TFCamembertForSentenceEmbedding
   models.FullTextBert
   models.FullTextConfig
   models.BertForMultilabelClassification
   models.CamembertForMultilabelClassification
   models.MeanMaskedPooling
   models.MaxMaskedPooling
   models.PoolingClassificationHead

:mod:`tf_codage.models.transformer` -- vanilla transformer model
----------------------------------------------------------------

.. automodule:: tf_codage.models.transformer

.. currentmodule:: tf_codage.models.transformer

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Encoder
   Decoder
   Transformer



:mod:`tf_codage.utils` - common utilities
=========================================

.. automodule:: tf_codage.utils

.. currentmodule:: tf_codage

.. autosummary::
   :toctree: generated

   utils.TeeStream
   utils.print_console
   utils.download_hdfs
   utils.notebook_copy_stdout
   utils.save_model
   utils.grep_keras_results_from_notebook
   utils.batch_generator
   utils.split_file
