.. role:: python(code)
     :language: python
     :class: highlight

.. :tocdepth: 2

.. _codecs:

Neural Codecs
=============

Two transforms hand an :class:`~audiotree.AudioTree` to a *neural audio
codec* and store what comes back on the tree itself:

- :func:`~audiotree.transforms.encode_with_codec` — discrete tokens, stored on
  ``AudioTree.codes``
- :func:`~audiotree.transforms.encode_latents` — continuous embeddings, stored
  on ``AudioTree.latents``

Neither one ships a codec. Both take a *codec object* you supply — a wrapper
around DAC, EnCodec, SoundStream, or your own trained model — and audiotree
never imports it, so the heavyweight dependency stays yours. What audiotree
fixes is the interface: two one-method protocols that say what the transforms
will call and what they expect back.

.. testsetup::

    # Hidden setup: a small synthetic corpus so the Grain pipeline example
    # further down runs.
    import os
    import tempfile
    import numpy as np
    import soundfile

    _g = np.random.default_rng(0)
    _corpus = tempfile.mkdtemp()
    for _i in range(4):
        soundfile.write(
            os.path.join(_corpus, f"{_i}.wav"),
            (0.1 * _g.standard_normal((44_100, 2))).astype(np.float32),
            44_100,
        )

The protocols
-------------

Both are ``@runtime_checkable`` :class:`typing.Protocol`\ s, so conformance is
structural: implement the method and your class satisfies the protocol without
inheriting from anything.

.. list-table::
   :header-rows: 1
   :widths: 26 40 34

   * - Protocol
     - Required method
     - Consumed by
   * - ``AudioCodec``
     - ``encode(AudioTree) -> codes``
     - ``encode_with_codec``
   * - ``LatentAudioCodec``
     - ``encode_to_latent(AudioTree) -> latents``
     - ``encode_latents``

They are deliberately separate. A codec that only tokenizes satisfies
``AudioCodec`` alone; one that also exposes its continuous bottleneck satisfies
both, and the same object can then be handed to either transform.

What the protocol requires of *you*
-----------------------------------

The method signature is the whole contract, but it leaves two
responsibilities with the codec:

**The codec owns its input conventions.** ``encode`` is handed the tree exactly
as the pipeline produced it — at whatever sample rate and channel count the
loader used. Resampling to the codec's own rate, and deciding what a stereo
input means to a mono encoder (fold the channels into the batch, downmix,
refuse), is the codec's job. audiotree does not insert a ``resample`` or
``mono`` step for you, because only the codec knows what its weights were
trained on.

**The codec owns its output shapes.** ``codes`` is stored verbatim. A
residual-vector-quantizer codec typically returns ``(batch, codebooks,
frames)``; a single-codebook tokenizer might return ``(batch, frames)``.
audiotree does not reshape, transpose, or validate it — only the leading batch
axis matters, and only because everything else on the tree is batched that way
too. The codes are also all that is stored: a codec that keeps extra state
(say, a loudness-normalization factor applied before quantizing that its
decoder needs back) manages that state itself, since audiotree stores only
what ``encode`` returns.

Writing a conforming codec
--------------------------

Here is a complete, deliberately tiny one. It peak-normalizes, pools the
waveform into frames, and quantizes each frame to 8 bits — no neural network in
sight, but it exercises every part of the contract, including resampling to its
own rate and folding a stereo input down to mono. A real wrapper differs only
in what happens between the resample and the return.

.. testcode::

    import numpy as np
    from audiotree import AudioTree


    class ToyCodec:
        """A stand-in codec: peak-normalize, mean-pool, quantize to 8 bits."""

        sample_rate = 16_000  # what this codec was "trained" at
        hop_length = 320  # 50 frames per second
        num_levels = 256

        def encode(self, audio: AudioTree):
            # 1. The codec owns resampling and channel handling.
            audio = audio.to_mono().resample(self.sample_rate)
            waveform = np.asarray(audio.waveform)  # (B, 1, T)

            # 2. Peak-normalize. The factor is this codec's own business: keep
            #    it yourself if your decoder needs it back.
            scale = np.maximum(np.abs(waveform).max(axis=(1, 2)), 1e-8)
            normalized = waveform / scale[:, None, None]

            # 3. Pool into frames and quantize. The shape is the codec's choice;
            #    only the leading batch axis is fixed.
            batch = waveform.shape[0]
            frames = normalized.shape[-1] // self.hop_length
            pooled = (
                normalized[..., : frames * self.hop_length]
                .reshape(batch, 1, frames, self.hop_length)
                .mean(axis=-1)
            )  # (B, 1, frames)
            return np.clip(
                np.rint((pooled + 1.0) * 0.5 * (self.num_levels - 1)),
                0,
                self.num_levels - 1,
            ).astype(np.int32)

        def encode_to_latent(self, audio: AudioTree):
            """The same pooling, left continuous — this codec satisfies both."""
            codes = self.encode(audio)
            return codes.astype(np.float32) / (self.num_levels - 1)

Because the protocols are ``runtime_checkable``, you can assert conformance
without importing anything from the codec's own package:

.. testcode::

    from audiotree.transforms import AudioCodec

    print(isinstance(ToyCodec(), AudioCodec))
    print(isinstance(object(), AudioCodec))

.. testoutput::

    True
    False

.. note::
   ``isinstance`` against a runtime-checkable protocol only checks that the
   *method exists*, not that its signature or its return type match. It catches
   the typo, not the wrong shape.

Using the transforms
--------------------

Both are deterministic ``Map`` transforms: build one from a codec instance,
then use ``.map()`` — never ``.random_map()``, since there is nothing random to
seed.

.. testcode::

    from audiotree.transforms import encode_latents, encode_with_codec

    codec = ToyCodec()
    audio = AudioTree(np.full((4, 2, 44_100), 0.25, dtype=np.float32), 44_100)

    encoded = encode_with_codec(codec).map(audio)
    print(encoded.codes.shape)  # the codec's own convention
    print(encoded.codes.dtype)

    latent = encode_latents(codec).map(audio)
    print(latent.latents.shape)

.. testoutput::

    (4, 1, 50)
    int32
    (4, 1, 50)

The input was 44.1 kHz stereo and the codec resampled it to 16 kHz mono itself:
one second at ``hop_length=320`` is 50 frames. None of that was audiotree's
doing.

In a Grain pipeline the transform is one more ``.map()`` stage:

.. testcode::

    from audiotree.sources import create_audio_dataset
    from audiotree.transforms import volume_norm

    ds = create_audio_dataset(_corpus, sample_rate=44100, duration=1.0)
    ds = ds.seed(0)
    ds = ds.random_map(volume_norm(min_db=-20, max_db=-15))  # augment first…
    ds = ds.map(encode_with_codec(ToyCodec()))  # …tokenize last

    iter_ds = ds.to_iter_dataset().batch(4, batch_fn=AudioTree.batch)
    item = next(iter(iter_ds))
    print(item.codes.shape)

.. testoutput::

    (4, 1, 50)

Encode last
-----------

That ordering is not stylistic. ``codes`` and ``latents`` are *derived*
fields: they describe one particular
waveform, at one particular length, rate, channel count and level. Every
audiotree operation that changes the audio in one of those ways clears them,
rather than leaving tokens behind that silently describe the previous audio:

.. testcode::

    encoded = encode_with_codec(ToyCodec()).map(audio)
    print(encoded.codes is None)

    resampled = encoded.resample(22_050)
    print(resampled.codes is None)

.. testoutput::

    False
    True

So put the codec stage after every augmentation that touches the waveform;
otherwise you pay for an encode whose result is thrown away. The same rule
governs ``lufs`` — see :ref:`transform_chaining`.

The mirror image of that rule is a **pass-through**: a tree that *already* has
``codes`` is returned unchanged by ``encode_with_codec``, and one that already
has ``latents`` is returned unchanged by ``encode_latents``. That makes the
stage idempotent and cheap to re-apply — but it also means a second codec
cannot overwrite the first one's output. To re-encode with a different codec,
clear the field first:

.. testcode::

    already = encode_with_codec(ToyCodec()).map(audio)
    again = encode_with_codec(ToyCodec()).map(already)
    print(again.codes is already.codes)  # untouched, not recomputed

    fresh = encode_with_codec(ToyCodec()).map(already.replace(codes=None))
    print(fresh.codes.shape)

.. testoutput::

    True
    (4, 1, 50)

``scope`` and ``output_key``
----------------------------

Both transforms take the standard keyword-only ``scope`` and ``output_key``
arguments described in :ref:`dict_batches`, and both honour them exactly as
every other map transform does. What they do **not** do is worth stating
plainly, because the names invite a wrong guess.

``scope`` picks which leaves of a *dict element* get encoded. It has nothing to
say about which AudioTree field is written:

.. testcode::

    element = {"dry": audio, "wet": audio}
    result = encode_with_codec(ToyCodec(), scope=["wet"]).map(element)

    print(result["wet"].codes is None)
    print(result["dry"].codes is None)

.. testoutput::

    False
    True

``output_key`` writes the encoded tree under a **new dict key**, leaving the
input leaf as it was. It renames the *leaf*, not the field — the tokens still
land on ``.codes`` and the latents still land on ``.latents``:

.. testcode::

    result = encode_latents(ToyCodec(), scope=["dry"], output_key="dry_latents").map(
        {"dry": audio}
    )

    print(sorted(result))
    print(result["dry"].latents is None)  # input leaf untouched
    print(result["dry_latents"].latents.shape)  # still the `latents` field

.. testoutput::

    ['dry', 'dry_latents']
    True
    (4, 1, 50)

Two consequences follow:

- **The destination field is fixed.** No argument makes ``encode_with_codec``
  write anywhere other than ``codes``. If you want two codecs' tokens side by
  side, give each its own dict leaf with ``scope`` / ``output_key`` and let each
  leaf carry its own ``codes``.
- **``output_key`` needs a dict element.** Passing it while mapping over a bare
  :class:`~audiotree.AudioTree` is an error — there is no dict to add a key
  to. Use it only on the dict-of-AudioTree elements of :ref:`dict_batches`.

Both backends, one codec
------------------------

``audiotree.transforms`` and ``audiotree.transforms.jax`` export the *same*
``encode_with_codec``, ``encode_latents`` and ``AudioCodec`` objects — unlike
the augmentations, there is no separate NumPy and JAX implementation to choose
between, because all the arithmetic lives in your codec. Import them from
whichever namespace the rest of the file uses.

Neither transform wraps the codec call in ``jax.jit``. That is a deliberate
omission: the caller decides where the JIT boundary is, so a codec built from
JAX ops traces cleanly as part of a larger jitted step instead of being
compiled behind your back as a program of its own.

.. testcode::

    import jax
    import jax.numpy as jnp
    from audiotree.transforms.jax import encode_with_codec as jax_encode_with_codec


    class JitCodec:
        """The same idea in JAX ops, at whatever rate the pipeline hands it."""

        hop_length = 320

        def encode(self, audio: AudioTree):
            waveform = audio.waveform.mean(axis=1, keepdims=True)  # (B, 1, T)
            batch, _, samples = waveform.shape
            frames = samples // self.hop_length
            pooled = (
                waveform[..., : frames * self.hop_length]
                .reshape(batch, 1, frames, self.hop_length)
                .mean(axis=-1)
            )
            return jnp.round(pooled * 127).astype(jnp.int32)


    tokenize = jax_encode_with_codec(JitCodec())


    @jax.jit
    def encode_step(batch: AudioTree) -> AudioTree:
        return tokenize.map(batch)


    out = encode_step(AudioTree(jnp.full((8, 2, 16_000), 0.5), 16_000))
    print(out.codes.shape)
    print(out.codes.dtype)

.. testoutput::

    (8, 1, 50)
    int32

Keeping the tokens
------------------

``codes`` and ``latents`` are ordinary AudioTree fields, so both writers persist
them: :class:`~audiotree.writer.AudioWriter` records them as manifest columns
and :class:`~audiotree.tree_writer.TreeWriter` stores them as leaves of the
on-disk tree. That is the usual reason to run a codec inside a data pipeline at
all — tokenize a corpus once, then train from the tokens. See :ref:`writer`.

See Also
--------

- :ref:`transform_chaining` — where a codec stage belongs in a pipeline
- :ref:`dict_batches` — ``scope`` and ``output_key`` in full
- :ref:`writer` — persisting ``codes`` and ``latents`` to disk
- :func:`~audiotree.transforms.encode_with_codec`,
  :func:`~audiotree.transforms.encode_latents` — API reference
