AudioTree documentation
=======================

**AudioTree** is a `JAX <https://jax.readthedocs.io/en/latest/>`_ library for audio data loading and augmentations.
The source code is `here <https://github.com/DBraun/audiotree>`_.

There are three requirements:

.. code-block:: bash

   pip install "git+https://github.com/DBraun/argbind.git@improve.subclasses"
   pip install "git+https://github.com/DBraun/dm_aux.git@DBraun-patch-2"
   pip install "git+https://github.com/DBraun/jaxloudnorm.git@feature/speed-optimize"

Then AudioTree can be installed with pip:

.. code-block:: bash

   pip install audiotree

The namesake class :class:`~audiotree.core.AudioTree` is a `flax.struct.dataclass`_ with properties for audio data,
sample rate, on-demand data such as loudness, and optional metadata such as MIDI pitch, velocity, and duration.

Although ``AudioTree`` is a specific class, we loosely refer to any combination of dictionaries and lists of
``AudioTrees`` as also an ``AudioTree`` (check out the `Pytree`_ JAX docs).
For example, in the code below, we consider ``batch`` to be an ``AudioTree``.

.. code-block:: python

    from jax import numpy as jnp
    from audiotree import AudioTree
    sample_rate = 44100
    data = jnp.zeros((16, 2, 441000)) # dummy placeholder shaped (B, C, T)
    audio_tree = AudioTree(data, sample_rate)
    batch = {"src": [audio_tree, audio_tree], "target": audio_tree}

The batch above can be used with `jax.tree.map`_ to create a new batch. That's essentially what the Transform classes in
:mod:`~audiotree.transforms` do.
They perform GPU-based `jax.jit`_-compatible augmentations on arbitrarily shaped AudioTrees.
When used with `ArgBind`_, they are also highly configurable from the command-line and YAML.

Whether you're creating a data loader for training, validation, testing, or prompt generation, AudioTree can help.

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

AudioTree is inspired by `AudioTools`_. Thank you!

Citation
---------------

.. code-block::

   @software{Braun_AudioTree_2024,
      author = {Braun, David},
      month = aug,
      title = {{AudioTree}},
      url = {https://github.com/DBraun/audiotree},
      version = {0.1.2},
      year = {2024}
   }

.. _flax.struct.dataclass: https://flax.readthedocs.io/en/latest/api_reference/flax.struct.html#flax.struct.dataclass
.. _Pytree: https://jax.readthedocs.io/en/latest/pytrees.html
.. _jax.tree.map: https://jax.readthedocs.io/en/latest/_autosummary/jax.tree.map.html#jax.tree.map
.. _jax.jit: https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html
.. _ArgBind: https://github.com/pseeth/argbind/
.. _AudioTools: https://github.com/descriptinc/audiotools/