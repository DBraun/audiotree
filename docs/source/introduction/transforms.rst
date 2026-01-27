.. role:: python(code)
     :language: python
     :class: highlight

.. :tocdepth: 2

.. _transforms:

Transforms
======================

..

.. ---------------------------

Transforms in ``audiotree.transforms`` are `Grain`_
`transformations <https://github.com/google/grain/blob/main/docs/data_loader/transformations.md>`_ that operate on AudioTrees.
Examples include:

   * GPU-based `volume normalization <https://github.com/DBraun/jaxloudnorm/>`_ to a LUFS value in a configurable uniformly sampled range
   * Encoding to `DAC-JAX`_ audio tokens
   * Swapping stereo channels
   * Randomly shifting or corrupting the phase(s) of a waveform
   * and more...

**Quick Start: Chaining with Datasets**

.. code-block:: python

    from audiotree.sources import create_audio_dataset
    from audiotree.transforms import volume_norm, trim

    # Create dataset
    ds = create_audio_dataset(
        sources="/data/audio",
        sample_rate=44100,
        duration=5.0,
    )

    # Chain transforms
    ds = ds.random_map(
        volume_norm(min_db=-20, max_db=-15),
        seed=42,
    )
    ds = ds.map(trim(length=3.0))

    # Access augmented audio
    audio_tree = ds[0]

For complete examples of chaining transforms, building augmentation pipelines, and performance
optimization, see :ref:`transform_chaining`.

**Quick Start: With ArgBind**

.. code-block:: python

    import argbind
    from audiotree import transforms

    # Bind transform function to argbind
    VolumeNorm = argbind.bind(transforms.volume_norm)

    # Config via YAML
    args = argbind.parse_args()
    with argbind.scope(args):
        transform = VolumeNorm()  # Uses params from YAML
        ds = ds.random_map(transform, seed=42)

For a complete guide on using ArgBind with transforms, including scoped configurations, YAML syntax,
and complete examples, see :ref:`argbind_guide`.

Basic Usage
-----------

**Direct Python Usage**

Transforms are functions that return transform instances:

.. code-block:: python

    from audiotree.transforms import volume_norm, volume_change, trim
    from audiotree import AudioTree
    import jax

    # Create audio
    audio_tree = AudioTree(...)
    audio_tree = audio_tree.replace_loudness()

    # Apply transforms
    rng = jax.random.key(42)

    # Random transforms need an RNG
    transform1 = volume_norm(min_db=-20, max_db=-15)
    audio_tree = transform1.random_map(audio_tree, rng)

    # Deterministic transforms
    transform2 = trim(length=3.0)
    audio_tree = transform2.map(audio_tree)

**With Dict Batches**

Transforms work with dictionaries of AudioTrees:

.. code-block:: python

    batch = {"src": audio_tree1, "target": audio_tree2}

    # Transform only 'src' using scope
    transform = volume_change(
        min_db=-12,
        max_db=3,
        prob=0.9,
        scope={'src': {'scope': True}},
    )
    batch = transform.random_map(batch, rng)

For complete information on Dict[str, AudioTree] batches, see :ref:`dict_batches`.

**With ArgBind and YAML**

.. code-block:: yaml

    volume_change.min_db: -12
    volume_change.max_db: 3
    volume_change.prob: 0.9
    volume_change.scope:
      src:
        scope: true

Transform Parameters
--------------------

All transforms support these parameters:

**For random transforms:**

- ``prob``: Probability of applying (0.0 to 1.0, default 1.0)
- ``split_seed``: Use different RNG per leaf (default True)
- ``scope``: Which PyTree leaves to transform (default None = all)
- ``output_key``: Where to store output (default None = in-place)

**For map transforms:**

- ``scope``: Which PyTree leaves to transform (default None = all)
- ``output_key``: Where to store output (default None = in-place)

**Split Seed Example**

By setting ``split_seed=False``, you can apply the same random augmentation to all items:

.. code-block:: python

    # Different augmentation per item (default)
    transform = volume_change(min_db=-6, max_db=6, split_seed=True)

    # Same augmentation for all items
    transform = volume_change(min_db=-6, max_db=6, split_seed=False)

.. code-block:: yaml

    volume_change.split_seed: false


Output Key
----------

You can specify an output key so that the result is stored in a new key instead of replacing the original:

.. code-block:: python

    batch = {"src": audio_tree, "target": audio_tree}

    transform = volume_change(
        min_db=-12,
        max_db=3,
        scope={'src': {'scope': True}},
        output_key='modified',
    )
    batch = transform.random_map(batch, rng)

    # Result has original plus new key
    # {"src": original, "target": original, "modified": transformed}

.. code-block:: yaml

    volume_change.output_key: "modified"
    volume_change.scope:
      src:
        scope: true

Scope
-----

Use ``scope`` to selectively transform specific keys in a dictionary batch:

.. code-block:: python

    batch = {"dry": dry_audio, "wet": wet_audio, "reference": ref_audio}

    # Transform only 'dry' and 'wet', not 'reference'
    transform = volume_norm(
        min_db=-20,
        max_db=-15,
        scope={
            'dry': {'scope': True},
            'wet': {'scope': True},
        },
    )
    batch = transform.random_map(batch, rng)

**Nested dictionaries** also work:

.. code-block:: python

    batch = {
        "input": {"dry": dry_audio, "wet": wet_audio},
        "target": target_audio,
    }

    # Transform only input.dry
    transform = volume_norm(
        min_db=-20,
        max_db=-15,
        scope={'input': {'dry': {'scope': True}}},
    )
    batch = transform.random_map(batch, rng)

For complete examples, see :ref:`dict_batches`.

Available Transforms
--------------------

**Random Transforms** (use with ``.random_map()``):

- ``volume_norm(min_db, max_db)`` - Normalize to random loudness
- ``volume_change(min_db, max_db)`` - Random gain adjustment
- ``invert_phase()`` - Invert audio phase
- ``swap_stereo()`` - Swap stereo channels
- ``corrupt_phase(amount, ...)`` - Corrupt phase spectrum
- ``shift_phase(amount)`` - Shift phase spectrum
- ``roll(min_seconds, max_seconds, mode)`` - Circular shift audio

**Map Transforms** (use with ``.map()``):

- ``trim(length, mode)`` - Trim or pad to fixed length
- ``mono()`` - Convert to mono
- ``stereo()`` - Convert to stereo
- ``rescale_audio()`` - Rescale to [-1, 1] range
- ``identity()`` - No-op transform

**Special Transforms**:

- ``choose(*transforms, c, weights, prob)`` - Randomly select from multiple transforms
- ``encode_with_codec(encoder_fn, num_codebooks)`` - Encode with neural codec
- ``encode_latents(encoder_fn)`` - Encode to latent space

Creating Custom Transforms
---------------------------

Use decorators to create custom transforms:

.. code-block:: python

    from audiotree.transforms.decorators import random_transform, map_transform
    import jax

    @random_transform
    def my_augmentation(audio_tree, rng, strength=1.0):
        # Your augmentation logic here
        noise = jax.random.normal(rng, audio_tree.audio_data.shape) * strength
        audio_data = audio_tree.audio_data + noise
        return audio_tree.replace(audio_data=audio_data)

    # Use it
    transform = my_augmentation(strength=0.1, prob=0.8)
    ds = ds.random_map(transform, seed=42)

    # Or with argbind
    import argbind
    MyAugmentation = argbind.bind(my_augmentation)

    # config.yml:
    # my_augmentation.strength: 0.1
    # my_augmentation.prob: 0.8

See Also
--------

For complete guides and examples:

- :ref:`transform_chaining` - Chaining transforms with datasets
- :ref:`dict_batches` - Using Dict[str, AudioTree] batches
- :ref:`argbind_guide` - Configuring transforms with ArgBind
- :ref:`multiprocessing` - Parallel data loading
- `argbind_augmentations examples <../../examples/argbind_augmentations/>`_ - Complete working examples
- `tests/transforms/test_core.py <https://github.com/DBraun/audiotree/blob/main/tests/transforms/test_core.py>`_ - Test examples
- `DAC-JAX`_ - Real-world usage in production

.. _ArgBind: https://github.com/pseeth/argbind/
.. _DAC-JAX: https://github.com/DBraun/DAC-JAX
.. _Grain: https://github.com/google/grain