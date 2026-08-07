.. role:: python(code)
     :language: python
     :class: highlight

.. :tocdepth: 2

.. _transforms:

Transforms in the Training Loop
===============================

Every augmentation ships with two backends that share the same names and
parameters:

- **NumPy** (``audiotree.transforms``) — for CPU Grain data pipelines. You chain
  these onto a dataset with ``.map()`` / ``.random_map()``; that is the previous
  chapter, :ref:`transform_chaining`. They take a ``np.random.Generator``.
- **JAX** (``audiotree.transforms.jax``) — the same transforms built from JAX ops,
  so they run on the accelerator and compose inside ``jax.jit``. They take a
  ``jax.random.key``.

Reach for the JAX backend when you want to augment a batch **inside the training
step**, on-device, instead of in CPU data-loader workers. A transform's
``random_map`` is a plain function of ``(tree, key)``, so it traces cleanly under
``@jax.jit`` right next to your model:

.. testcode::

    import jax
    import jax.numpy as jnp
    from audiotree import AudioTree
    from audiotree.transforms import jax as jax_transforms

    # The same transform as the NumPy backend, but JAX-native (takes a jax.random.key).
    augment = jax_transforms.volume_change(min_db=-6, max_db=6)

    @jax.jit
    def train_step(batch: AudioTree, key):
        # Augment on-device, then run your model. A trivial energy statistic
        # stands in for the model and loss here.
        batch = augment.random_map(batch, key)
        return jnp.mean(batch.waveform ** 2)

    batch = AudioTree(jnp.ones((8, 2, 16_000)) * 0.5, 16_000)

    # Split a fresh key each step so every step augments differently.
    key = jax.random.key(0)
    for _ in range(3):
        key, subkey = jax.random.split(key)
        loss = train_step(batch, subkey)

    print(loss.shape)     # a scalar
    print(bool(loss > 0))

.. testoutput::

    ()
    True

Chain several augmentations by splitting a key per random transform; map
transforms such as ``trim`` and ``resample`` need no key and compose the same way:

.. testcode::

    gain = jax_transforms.volume_change(min_db=-6, max_db=6)
    phase = jax_transforms.invert_phase(prob=0.5)
    resize = jax_transforms.trim(length=0.5)   # a map transform

    @jax.jit
    def augment_batch(batch: AudioTree, key):
        k1, k2 = jax.random.split(key)
        batch = gain.random_map(batch, k1)
        batch = phase.random_map(batch, k2)
        batch = resize.map(batch)
        return batch

    out = augment_batch(batch, jax.random.key(1))
    print(out.waveform.shape)

.. testoutput::

    (8, 2, 8000)

Pair this with :func:`grain.experimental.device_put` (see
:ref:`streaming-device-put`) to stream host batches onto the accelerator and
augment them in the same jitted step that trains your model.

.. note::
   The full catalog of transforms — their shared parameters (``prob``,
   ``split_seed``, ``scope``, ``output_key``), which ones invalidate the cached
   ``lufs``, and the decorators for writing your own — lives in the
   :mod:`audiotree.transforms` API reference. Both backends expose the same names
   (``choose`` is the one exception — it branches in Python, so it cannot be
   traced), so anything you configure for a Grain pipeline works here by importing
   it from ``audiotree.transforms.jax``.

   ``encode_with_codec`` and ``encode_latents`` are a special case in the other
   direction: the two namespaces export the *same* objects, because the
   arithmetic lives in the codec you supply rather than in audiotree. Neither
   wraps the codec in ``jax.jit``, so a JAX codec traces into a step like the one
   above. See :ref:`codecs`.

.. note::
   **Same names, same semantics, not bit-identical results.** The two backends
   agree on what a transform *means* — and ``tests/transforms/test_backend_parity.py``
   pins that agreement — but two implementations of the same DSP do not produce
   identical floats:

   - ``resample`` uses librosa/soxr on NumPy and a Julius-style sinc filter on
     JAX. On band-limited content they agree to about ``8.5e-5`` absolute; on
     broadband noise only to ~24 dB SNR, essentially all of it in the
     anti-aliasing filter's transition band, where the two filter designs roll
     off differently.
   - ``volume_norm`` uses FIR K-weighting (pyloudnorm-style) on NumPy and IIR
     K-weighting on JAX. Measured loudness differs by up to 0.031 dB, so the
     output differs from the other backend's by a pure scalar gain of about
     0.036 dB.

   Neither is a bug, but do not expect a NumPy-augmented run and a
   JAX-augmented run to reproduce each other sample-for-sample.

Next
----

With data loaded, augmented, and ready for training, the final Getting-started
chapter, :ref:`writer`, shows how to write prepared datasets back to disk.
