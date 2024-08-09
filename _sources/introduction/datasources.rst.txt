.. role:: python(code)
     :language: python
     :class: highlight

.. :tocdepth: 2

.. _datasources:

Data Sources
=========================

..  

.. ---------------------------

Data Sources in ``audiotree.datasources`` are `Grain <https://github.com/google/grain>`_ `data sources <https://github.com/google/grain/blob/main/docs/data_sources.md>`_
that are specially designed for audio. Grain is a new library for dataset operations in JAX with no TensorFlow dependency.

For now, there are only two types of Data Sources, but they fit many needs.
You can take a look at DAC-JAX's `input_pipeline.py <https://github.com/DBraun/DAC-JAX/blob/main/scripts/input_pipeline.py>`_ to see how they're used.
