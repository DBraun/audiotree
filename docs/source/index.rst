AudioTree documentation
=======================

**AudioTree** represents audio as an ML-framework-agnostic `pytree <https://jax.readthedocs.io/en/latest/pytrees.html>`_, with `JAX <https://jax.readthedocs.io/en/latest/>`_ data loading and augmentations.
The source code is `here <https://github.com/DBraun/audiotree>`_.

AudioTree can be installed with pip:

.. code-block:: bash

   pip install audiotree

The namesake class :class:`~audiotree.core.AudioTree` is a container for audio-related information with a batch-axis convention.
Specifically, it's a `flax.struct.dataclass`_ with properties for the time-domain waveform,
sample rate, on-demand data such as loudness, and optional data such as filepaths, MIDI pitch, velocity.
An AudioTree can also store arrays for codebooks or latent embeddings.

AudioTree integrates with `Grain`_ to provide complete data pipelines. Load audio from directories,
apply balanced sampling across groups, and chain augmentations:

.. testsetup::

    # Hidden setup: stand in for the "/data/speech" and "/data/music" directories
    # below with two temp dirs of small synthetic WAVs, so this example runs.
    import os
    import tempfile
    import numpy as np
    import soundfile

    _g = np.random.default_rng(0)
    _speech_dir, _music_dir = tempfile.mkdtemp(), tempfile.mkdtemp()
    for _d in (_speech_dir, _music_dir):
        for _i in range(3):
            soundfile.write(
                os.path.join(_d, f"{_i}.wav"),
                (0.1 * _g.standard_normal((44_100, 2))).astype(np.float32),
                44_100,
            )

.. testcode::

    from audiotree import AudioTree
    from audiotree.sources import create_balanced_audio_dataset
    from audiotree.transforms import volume_norm, stereo

    # Create dataset with balanced sampling across groups
    ds = create_balanced_audio_dataset(
        sources={"speech": [_speech_dir], "music": [_music_dir]},
        weights={"speech": 0.7, "music": 0.3},
        sample_rate=44100,
        duration=10.0,
    )

    # Chain transforms using Grain's API
    ds = ds.seed(42)
    ds = ds.map(stereo())
    ds = ds.random_map(volume_norm(min_db=-20, max_db=-15))

    # Convert to iterable and batch
    iter_ds = ds.to_iter_dataset().batch(32, batch_fn=AudioTree.batch)

    # Access batched AudioTrees
    batch: AudioTree = next(iter(iter_ds))
    print(batch.waveform.shape)    # (32, channels, 441000)
    print(batch.source[:3])        # ["speech", "music", ...]
    # print(batch.filepath)        # ["path/to/file_abc.wav", "path/to/file_xyz.wav", ...]

.. testoutput::

    (32, 2, 441000)
    ['speech', 'speech', 'speech']

Transforms work on any `Pytree`_ of AudioTrees, including dictionaries and lists.
This means transforms can receive and send patterns like ``{"dry": audio_tree, "wet": audio_tree}`` where you selectively augment
specific keys using the ``scope`` parameter (see :ref:`dict_batches`).

When used with `ArgBind`_, transforms are configurable from the command-line and YAML (see :ref:`argbind_guide`):

.. code-block:: bash

    python train.py --volume_norm.min_db=-25 --volume_norm.max_db=-15

Content
--------------------------
.. toctree::
   :maxdepth: 1
   :caption: Introduction

   introduction/introduction
   introduction/sources
   introduction/balanced_datasets
   introduction/windowed_datasets
   introduction/transforms
   introduction/transform_chaining
   introduction/dict_batches
   introduction/argbind_guide
   introduction/multiprocessing
   introduction/writer

.. toctree::
   :maxdepth: 1
   :caption: AudioTree API

   audiotree_api/core
   audiotree_api/sources
   audiotree_api/transforms
   audiotree_api/writer
   audiotree_api/tree

.. toctree::
   :maxdepth: 1
   :caption: Project

   changelog

Acknowledgments
---------------

AudioTree is inspired by `AudioTools`_. Thank you!

Citation
---------------

.. code-block::

   @software{Braun_AudioTree_2025,
      author = {Braun, David},
      month = mar,
      title = {{AudioTree}},
      url = {https://github.com/DBraun/audiotree},
      version = {0.2.0},
      year = {2025}
   }

.. _flax.struct.dataclass: https://flax.readthedocs.io/en/latest/api_reference/flax.struct.html#flax.struct.dataclass
.. _Grain: https://github.com/google/grain
.. _Pytree: https://jax.readthedocs.io/en/latest/pytrees.html
.. _jax.tree.map: https://jax.readthedocs.io/en/latest/_autosummary/jax.tree.map.html#jax.tree.map
.. _jax.jit: https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html
.. _ArgBind: https://github.com/DBraun/argbind/
.. _AudioTools: https://github.com/descriptinc/audiotools/
