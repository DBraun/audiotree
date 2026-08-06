.. role:: python(code)
     :language: python
     :class: highlight

.. :tocdepth: 2

.. _argbind_guide:

Using ArgBind with Transforms
==============================

`ArgBind`_ allows you to configure AudioTree transforms via YAML files or command-line arguments, making experiments reproducible and easy to modify without changing code.

Why ArgBind?
------------

ArgBind enables:

- **YAML Configuration**: Define all parameters in config files
- **Command-Line Overrides**: Quickly test parameter changes
- **Scoped Configs**: Different settings for train vs validation
- **Reproducibility**: Save and reload exact configurations
- **No Boilerplate**: Auto-generated CLI from function signatures

Basic Example
-------------

The core workflow: wrap each transform with ``argbind.bind`` so its parameters
are read from a YAML file (or the command line) at run time instead of being
hard-coded. Below, ``volume_norm`` and ``trim`` are bound, then configured
entirely from ``config.yml``:

.. code-block:: python

    import argbind
    from audiotree import AudioTree, transforms
    import numpy as np

    # Bind transform functions to argbind
    volume_norm = argbind.bind(transforms.volume_norm)
    trim = argbind.bind(transforms.trim)

    def main():
        # Create audio
        audio_tree = AudioTree(np.random.randn(4, 1, 44100*4), sample_rate=44100)
        audio_tree = audio_tree.replace_lufs()

        # Apply transforms (config set via argbind)
        rng = np.random.default_rng(42)
        audio_tree = volume_norm().random_map(audio_tree, rng)
        audio_tree = trim().map(audio_tree)

        print("Final shape:", audio_tree.waveform.shape)

    if __name__ == "__main__":
        args = argbind.parse_args()
        with argbind.scope(args):
            main()

And this config file (``config.yml``):

.. code-block:: yaml

    volume_norm.min_db: -20
    volume_norm.max_db: -15

    trim.length: 1.0

Run with:

.. code-block:: bash

    python script.py --args.load=config.yml

Scoped Configurations
---------------------

Scopes let one transform hold several configurations — e.g. aggressive
augmentation for ``train`` but a deterministic setting for ``val`` — chosen at
call time by the ``with argbind.scope(args, ...)`` block that wraps it. Prefix a
YAML key with ``train/`` or ``val/`` to target a scope:

**config.yml:**

.. code-block:: yaml

    # Training: aggressive augmentation
    train/volume_norm.min_db: -25
    train/volume_norm.max_db: -15

    # Validation: deterministic (same min/max)
    val/volume_norm.min_db: -20
    val/volume_norm.max_db: -20

    # Shared config (no scope prefix)
    trim.length: 1.0

**Python:**

.. skip-snippet-exec: fragment; argbind parses the command line at run time.

.. code-block:: python

    volume_norm = argbind.bind(transforms.volume_norm, "train", "val")

    args = argbind.parse_args()

    # Training pipeline
    with argbind.scope(args, "train"):
        transform = volume_norm()  # Uses train/volume_norm config
        train_audio = transform.random_map(audio_tree, rng)

    # Validation pipeline
    with argbind.scope(args, "val"):
        transform = volume_norm()  # Uses val/volume_norm config
        val_audio = transform.random_map(audio_tree, rng)

Binding Multiple Transforms
----------------------------

``bind_module`` binds an entire transforms module in one call, instead of one
``argbind.bind`` per transform. This is the pattern for a name-driven,
per-scope augmentation pipeline: which ``transforms`` a run applies — and each
one's parameters — come from the (scoped) config, so ``train`` and ``val`` can
augment differently with no code change.

.. code-block:: python

    import numpy as np
    import argbind
    import grain
    from audiotree import AudioTree, transforms as transforms_lib

    # bind_module binds every function/class in the module. filter_fn is
    # optional — here it drops the two non-augmentation helpers (the ``Batch``
    # and ``choose`` classes); omit it and they bind too, but they are simply
    # never named in a transforms list.
    def filter_fn(fn):
        return callable(fn) and not isinstance(fn, type)

    # Bind all transforms in the module
    transforms_lib = argbind.bind_module(
        transforms_lib, "train", "val", filter_fn=filter_fn
    )

    # Now all transforms are available with scoped configs
    @argbind.bind("train", "val")
    def augment_batch(rng, batch, transforms: list[str] = None):
        for transform_name in transforms or []:
            transform = getattr(transforms_lib, transform_name)()
            if isinstance(transform, grain.transforms.RandomMap):
                # A NumPy Generator is stateful, so reusing ``rng`` gives each
                # transform fresh randomness — no key splitting needed.
                batch = transform.random_map(batch, rng)
            elif isinstance(transform, grain.transforms.Map):
                batch = transform.map(batch)
        return batch


    if __name__ == "__main__":
        args = argbind.parse_args()

        rng = np.random.default_rng(0)
        batch = AudioTree(np.random.randn(8, 1, 16_000), 16_000).replace_lufs()

        # Training applies train/augment_batch.transforms with train-scoped params.
        with argbind.scope(args, "train"):
            train_batch = augment_batch(rng, batch)

        # Validation applies val/augment_batch.transforms — here just identity.
        with argbind.scope(args, "val"):
            val_batch = augment_batch(rng, batch)

**config.yml:**

.. code-block:: yaml

    # Training: normalize to a random loudness, then trim.
    train/volume_norm.min_db: -25
    train/volume_norm.max_db: -15
    trim.length: 1.0

    # Which transforms each scope applies (by function name).
    train/augment_batch.transforms:
      - volume_norm
      - trim

    # Validation: no augmentation.
    val/augment_batch.transforms:
      - identity

Configuring a Dataset Pipeline
------------------------------

The same binding works when transforms drive a Grain dataset pipeline rather than a
single tree: bind the module, then call the bound transforms inside
``ds.random_map()`` / ``ds.map()``.

**pipeline.py:**

.. code-block:: python

    import argbind
    from audiotree.sources import create_balanced_audio_dataset
    from audiotree import transforms as transforms_lib

    # Bind the whole module at once, instead of one argbind.bind() per function.
    transforms = argbind.bind_module(transforms_lib)

    def create_pipeline():
        ds = create_balanced_audio_dataset(
            sources={"speech": ["/data/speech"], "music": ["/data/music"]},
            shuffle=True,
            repeat=True,
            sample_rate=44100,
            duration=5.0,
        )
        ds = ds.seed(42)  # seed once; each random_map derives its own seed

        # Parameters come from the config (see config.yml below).
        ds = ds.random_map(transforms.volume_norm())
        ds = ds.random_map(transforms.volume_change())
        ds = ds.random_map(transforms.invert_phase())
        ds = ds.map(transforms.trim())
        return ds

    if __name__ == "__main__":
        args = argbind.parse_args()
        with argbind.scope(args):
            ds = create_pipeline()
            for item in ds.to_iter_dataset():
                print(item.source, item.lufs)
                break

**config.yml:**

.. code-block:: yaml

    volume_norm.min_db: -25
    volume_norm.max_db: -15

    volume_change.min_db: -6
    volume_change.max_db: 6
    volume_change.prob: 0.9

    invert_phase.prob: 0.5

    trim.length: 3.0

**Run:**

.. code-block:: bash

    python pipeline.py --args.load=config.yml

For different train/validation augmentation, bind with scopes
(``argbind.bind_module(transforms_lib, "train", "val")``) and build each pipeline
under its ``with argbind.scope(args, "train")`` / ``"val"`` block — see
`Scoped Configurations`_ above.

Saving and Loading Configs
---------------------------

``--args.save`` writes the fully-resolved configuration of a run — every
parameter, including the ones left at their defaults — so the experiment can be
reproduced or replayed exactly. Save it:

.. code-block:: bash

    python script.py --args.load=config.yml --args.save=run_001.yml

This saves all arguments for exact reproducibility.

Load and override:

.. code-block:: bash

    python script.py --args.load=run_001.yml

Debug Mode
----------

When a parameter doesn't seem to take effect, ``--args.debug=1`` prints every
bound function with the exact argument values ArgBind resolved for it — the
quickest way to confirm the config actually reached the code:

.. code-block:: bash

    python script.py --args.load=config.yml --args.debug=1

Output:

.. code-block:: text

    volume_norm(
      min_db : float = -20
      max_db : float = -15
      split_seed : bool = True
      prob : float = 1.0
    )
    trim(
      length : float = 1.0
    )

Available Transform Parameters
-------------------------------

Every transform's parameters are set as flat ``transform_name.param`` keys in
YAML. A quick reference of the most common transforms and their knobs:

**volume_norm** - Normalize loudness to random LUFS value:

.. code-block:: yaml

    volume_norm.min_db: -20  # Minimum LUFS
    volume_norm.max_db: -15  # Maximum LUFS

**volume_change** - Random gain adjustment:

.. code-block:: yaml

    volume_change.min_db: -12  # Minimum gain in dB
    volume_change.max_db: 3    # Maximum gain in dB

**trim** - Trim to fixed length:

.. code-block:: yaml

    trim.length: 2.0  # Duration in seconds
    trim.mode: "wrap"  # Padding mode: "wrap" or "constant"

**invert_phase** - Randomly invert phase (no parameters):

.. code-block:: yaml

    invert_phase.prob: 0.5  # 50% chance of applying

**swap_stereo** - Randomly swap stereo channels (no parameters):

.. code-block:: yaml

    swap_stereo.prob: 0.5  # 50% chance of applying

**roll** - Circular shift audio:

.. code-block:: yaml

    roll.min_seconds: -1.0  # Minimum shift (negative = left)
    roll.max_seconds: 1.0   # Maximum shift (positive = right)
    roll.mode: "wrap"       # "wrap" or "constant"

Common Transform Parameters
----------------------------

All transforms support:

- ``prob``: Probability of applying (0.0 to 1.0)
- ``split_seed``: Whether to use different RNG for each item in batch
- ``scope``: Which parts of AudioTree to transform
- ``output_key``: Where to store transformed output

Example with all parameters:

.. code-block:: yaml

    volume_change.min_db: -12
    volume_change.max_db: 3
    volume_change.prob: 0.9  # 90% chance
    volume_change.split_seed: true
    volume_change.scope:
      src:  # Only transform "src" key
        scope: true

Complete Example
----------------

See the `argbind_augmentations examples`_ for complete, runnable code:

- **main.py**: Basic usage with individual transform binding
- **main2.py**: Advanced usage with ``bind_module`` and scopes
- **config.yml** and **config2.yml**: Example configurations

Run the examples:

.. code-block:: bash

    cd examples/argbind_augmentations
    python main.py --args.load=config.yml
    python main2.py --args.load=config2.yml

JAX JIT Compatibility
----------------------

Because ``argbind.scope`` only rewrites a plain dict, it composes with
``jax.jit``: enter the scope *inside* the jitted step so the resolved parameters
are baked into the trace alongside the model.

.. skip-snippet-exec: fragment; sketches a training step the guide never defines.

.. code-block:: python

    @jax.jit
    def train_step(audio_tree):
        with argbind.scope(args, "train"):
            return augment(audio_tree)

    # JIT compiled with train scope
    augmented = train_step(audio_tree)

Best Practices
--------------

1. **Always use YAML files** for configuration (not command-line JSON)
2. **Use scopes** for train/val differences instead of separate configs
3. **Save configs** after each run for reproducibility
4. **Use debug mode** when developing to verify parameters
5. **Filter transforms** with ``bind_module`` to avoid binding non-transform classes
6. **Document your configs** with YAML comments

Common Pitfalls
---------------

**Issue**: Config doesn't update when I change the YAML file

**Solution**: Ensure you're using ``--args.load=config.yml`` and not accidentally loading a cached/saved config

**Issue**: Command-line JSON syntax doesn't work

**Solution**: ArgBind doesn't support JSON for dict parameters. Use YAML files instead.

**Issue**: Transform applies to all keys, not just the one I want

**Solution**: Use the ``scope`` parameter to target specific keys

See Also
--------

- :ref:`transform_chaining` - Chaining transforms with datasets
- `ArgBind Documentation`_ - Complete ArgBind guide
- `ArgBind Examples`_ - Official ArgBind examples
- `argbind_augmentations examples`_ - AudioTree-specific examples

.. _ArgBind: https://github.com/DBraun/argbind/
.. _ArgBind Documentation: https://github.com/DBraun/argbind/blob/main/README.md
.. _ArgBind Examples: https://github.com/DBraun/argbind/tree/main/examples
.. _argbind_augmentations examples: https://github.com/DBraun/audiotree/tree/main/examples/argbind_augmentations
