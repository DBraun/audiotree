.. role:: python(code)
     :language: python
     :class: highlight

.. :tocdepth: 2

.. _transforms:

Transforms
======================

..  

.. ---------------------------

Transforms in ``audiotree.transforms`` are `Grain <https://github.com/google/grain>`_ 
`transformations <https://github.com/google/grain/blob/754636534bb16b5b2dd74970043d03e24ea44d3f/docs/transformations.md>`_ that operate on batches.
Examples include:

   * GPU-based `volume normalization <https://github.com/boris-kuz/jaxloudnorm/pull/1>`_ to a LUFS value in a configurable uniformly sampled range
   * Encoding to `DAC <https://github.com/DBraun/DAC-JAX>`_ audio tokens
   * Swapping stereo channels
   * Randomly shifting or corrupting the phase(s) of a waveform
   * and more...

AudioTree is compatible with `ArgBind <https://github.com/pseeth/argbind/>`_ but does not require it.
For the examples directly below, some other setup is required, so consider this to be an overview.
Before transformations, your data source might provide a single :class:`~audiotree.core.AudioTree` or a "tree" of :class:`~audiotree.core.AudioTree`:

.. code-block:: python

    from jax import numpy as jnp
    from audiotree import AudioTree
    sample_rate = 44100
    data = jnp.zeros((16, 2, 441000))  # dummy placeholder shaped (B, C, T)
    audio_tree = AudioTree(data, sample_rate)
    batch = {"src": [audio_tree, audio_tree], "target": audio_tree}

Then from YAML you can write the following to get a 90% chance of a random volume change between -12 and 3 decibels on just the ``"src"`` :class:`~audiotree.core.AudioTree`:

.. code-block:: yaml

    VolumeChange.prob:
        0.9
    VolumeChange.config:
        min_db: -12
        max_db: 3
    VolumeChange.scope:
        src:
            scope: True

By setting ``split_seed`` to False, you can apply the same augmentations to both the ``src`` and ``target``.

.. code-block:: yaml

    VolumeChange.split_seed: 0

This would make the most sense if the waveforms in ``src`` and ``target`` have the same dimensions.
For some transformations, having differently sized tensors would cause the augmentations to be different despite sharing the same ``jax.random.PRNGKey``.

You can specify an output key so that the result of the transformation is stored in a new sibling key:

.. code-block:: yaml

    VolumeChange.output_key: "src_modified"
    VolumeChange.scope:
        src:
            scope: True

The above will produce a batch *shaped* like this:

.. code-block:: python

    {
        "src": [audio_tree, audio_tree],
        "src_modified": [audio_tree, audio_tree],
        "target": audio_tree,
    }

Depending on the scope, we can end up with *multiple* new output leaves. Let's start with this batch:

.. code-block:: python

    batch = {
        "src":
        {
            "GT": audio_tree
        },
        "target":
        {
            "GT": audio_tree
        }
    }

Then with a scope of ``None`` (default) and this YAML:

.. code-block:: yaml

    VolumeChange.output_key: "modified"

We can produce this shape:

.. code-block:: python

    {
        "src":
        {
            "GT": audio_tree,
            "modified": audio_tree
        },
        "target":
        {
            "GT": audio_tree,
            "modified": audio_tree
        }
    }

You can also make more powerful (but complex) configs and scopes:

.. code-block:: yaml

    VolumeChange.config:
        max_db: 3
        src:
            min_db: -12
        target:
            min_db: -2

Note that the ``max_db`` is inherited by both ``src`` and ``target``.
This ability to inherit comes at the cost of potential name clashes between the keys of the config (e.g., ``"min_db"``, ``"max_db"``) and the keys in the AudioTree (``"src"``, ``"target"``, etc.).
The user is expected to use a data source to create AudioTrees that avoid these clashes.

Examples
--------

For now, the `tests/transforms/test_core.py <https://github.com/DBraun/audiotree/blob/main/tests/transforms/test_core.py>`_ is somewhat useful for thinking through the expected outputs.
AudioTree is also used in `DAC-JAX <https://github.com/DBraun/DAC-JAX>`_, which `shows <https://github.com/DBraun/DAC-JAX/blob/main/scripts/input_pipeline.py>`_ how to use `ArgBind <https://github.com/pseeth/argbind/>`_ and data sources.