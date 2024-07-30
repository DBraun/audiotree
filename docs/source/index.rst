AudioTree documentation
=======================

**AudioTree** is a `JAX <https://jax.readthedocs.io/en/latest/>`_ library for audio data loading and augmentations.
The source code is `here <https://github.com/DBraun/audiotree>`_, and it can be installed with pip:

.. code-block:: bash

   pip install audiotree

The namesake class :class:`~audiotree.core.AudioTree` is a `flax.struct.dataclass <https://flax.readthedocs.io/en/latest/api_reference/flax.struct.html#flax.struct.dataclass>`_
with properties for audio data, sample rate, on-demand data such as loudness, and optional metadata such as MIDI pitch, velocity, and duration.
Although ``AudioTree`` is a specific class, we loosely refer to any combination of dictionaries and lists of ``AudioTrees`` as also an ``AudioTree``.
For example, in the code below, we consider ``batch`` to be an ``AudioTree``.

.. code-block:: python

    from jax import numpy as jnp
    from audiotree import AudioTree
    audio_tree = AudioTree(jnp.zeros((16, 2, 441000)), 44100)  # dummy placeholder shaped (B, C, T)
    batch = {"src": [audio_tree, audio_tree], "target": audio_tree}

The batch above can be used with `jax.tree.map <https://jax.readthedocs.io/en/latest/_autosummary/jax.tree.map.html#jax.tree.map>`_ to create a new batch. That's essentially what the Transform classes in :data:`~audiotree.transforms` do.
They perform GPU-based jit-compatible augmentations on arbitrarily shaped AudioTrees.
They are also highly configurable from the command-line and YAML.

Whether you're creating a data loader for training, validation, testing, or prompt generation, ``AudioTree`` can help.

Content
--------------------------
.. toctree::
   :maxdepth: 1
   :caption: Introduction

   introduction/datasources
   introduction/transforms

.. toctree::
   :maxdepth: 1
   :caption: AudioTree API

   audiotree_api/core
   audiotree_api/datasources
   audiotree_api/transforms


Acknowledgments
---------------

AudioTree is inspired by `AudioTools <https://github.com/descriptinc/audiotools/>`_. Thank you!
