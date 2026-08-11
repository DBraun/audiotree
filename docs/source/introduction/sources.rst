.. role:: python(code)
     :language: python
     :class: highlight

.. :tocdepth: 2

.. _sources:

Data Sources
=========================

..

.. ---------------------------

``audiotree.sources`` provides `Grain`_ `data sources <https://google-grain.readthedocs.io/en/latest/data_sources/protocol.html>`_
built for audio: :func:`~audiotree.sources.create_audio_dataset` for simple
loading, and :func:`~audiotree.sources.create_balanced_audio_dataset` for
balanced multi-group sampling.

.. testsetup::

    # Hidden setup: stand in for the "/data/..." directories below with temp
    # dirs of small synthetic WAVs, so the Quick Start example runs.
    import os
    import tempfile
    import numpy as np
    import soundfile

    _g = np.random.default_rng(0)
    _speech_dir, _music_dir = tempfile.mkdtemp(), tempfile.mkdtemp()
    _more_speech_dir, _musdb_dir = tempfile.mkdtemp(), tempfile.mkdtemp()
    for _d in (_speech_dir, _music_dir, _more_speech_dir):
        for _i in range(3):
            soundfile.write(
                os.path.join(_d, f"{_i}.wav"),
                (0.1 * _g.standard_normal((44_100, 1))).astype(np.float32),
                44_100,
            )
    # A stems-style layout for the glob example: train/<track>/mixture.wav.
    for _track in ("track_a", "track_b"):
        _track_dir = os.path.join(_musdb_dir, "train", _track)
        os.makedirs(_track_dir)
        soundfile.write(
            os.path.join(_track_dir, "mixture.wav"),
            (0.1 * _g.standard_normal((44_100, 1))).astype(np.float32),
            44_100,
        )

**Quick Start**

.. testcode::

    from audiotree.sources import create_balanced_audio_dataset

    # Simple balanced dataset
    ds = create_balanced_audio_dataset(
        sources={
            "speech": [_speech_dir],
            "music": [_music_dir],
        },
        sample_rate=44100,
        duration=3.0,
    )

    print(ds[0].waveform.shape)

.. testoutput::

    (1, 1, 132300)

A group's list can mix several directories, individual files, and glob
patterns (recursive ``**`` included). Every entry is expanded by
:func:`~audiotree.sources.find_audio_files`, which searches directories
recursively and returns a **sorted, de-duplicated** list, so the corpus (and
therefore seeded shuffling) is deterministic across machines:

.. testcode::

    import os

    from audiotree.sources import find_audio_files

    # Two directories and a glob pattern, freely mixed. This is the expansion
    # the dataset builders apply to each entry of a group's list.
    corpus = find_audio_files([_speech_dir, f"{_musdb_dir}/train/*/mixture.wav"])
    print(sorted(os.path.basename(path) for path in corpus))

    # The same entries are valid inside `sources=`:
    ds = create_balanced_audio_dataset(
        sources={
            "speech": [_speech_dir, _more_speech_dir],
            "music": [f"{_musdb_dir}/train/*/mixture.wav"],
        },
        sample_rate=44100,
        duration=3.0,
    )

.. testoutput::

    ['0.wav', '1.wav', '2.wav', 'mixture.wav', 'mixture.wav']

For detailed information on balanced datasets, hierarchical directories, and weight-based sampling,
see :ref:`balanced_datasets`.

For parallel data loading with multiprocessing, see :ref:`multiprocessing`.

Excerpt Selection
-----------------

Which part of each file an item comes from is decided by the ``excerpt=``
argument, an :class:`~audiotree.ExcerptConfig` with one named ``strategy``.
The default — ``ExcerptConfig()``, i.e. ``strategy="random"`` — draws a
uniformly random offset on every read, with no loudness measured, so it costs
one read per item. Because the offset is drawn inside grain's ``random_map``,
a file visited again on a later epoch yields a *different* excerpt: files can
repeat, excerpts effectively never do. Pass ``strategy="start"`` to always
read from offset 0 instead — the right choice for a validation set that
should see identical audio every epoch.

The third strategy is a loudness search. Use it to select louder sections of
audio from corpora with long quiet stretches:

.. code-block:: python

    from audiotree.sources import create_balanced_audio_dataset
    from audiotree.sources import ExcerptConfig

    excerpt = ExcerptConfig(
        strategy="loudest",
        lufs_cutoff=-40,  # Only select sections above -40 LUFS
        num_tries=10,
    )

    ds = create_balanced_audio_dataset(
        sources={
            "speech": ["/data/speech"],
            "music": ["/data/music"],
        },
        excerpt=excerpt,
        sample_rate=44100,
        duration=3.0,
    )

The search is best-effort per file: after ``num_tries`` candidates the loudest
one wins, even if still below ``lufs_cutoff``. Pass ``on_failure="skip"`` to
drop such files instead (the loader returns ``None``, which grain skips at
iteration); see :ref:`balanced_datasets` for the details.

The same saliency search is available as a classmethod,
:meth:`~audiotree.AudioTree.loudest_excerpt`, for pulling a single loud
excerpt straight from a file path without building a dataset:

.. code-block:: python

    import numpy as np
    from audiotree import AudioTree
    from audiotree.sources import ExcerptConfig

    audio = AudioTree.loudest_excerpt(
        "/data/song.wav",
        rng=np.random.default_rng(0),
        excerpt=ExcerptConfig(strategy="loudest", lufs_cutoff=-40, num_tries=10),
        sample_rate=44100,
        duration=3.0,  # required
    )

It takes the same :class:`~audiotree.ExcerptConfig` as the dataset loaders and
forwards ``sample_rate`` / ``duration`` / ``mono`` to
:meth:`~audiotree.AudioTree.from_file`.

File Extensions
---------------

Customize which file types to load:

.. code-block:: python

    ds = create_balanced_audio_dataset(
        sources={"audio": ["/data/audio"]},
        extensions=[".wav", ".flac", ".mp3", ".ogg"],
        sample_rate=44100,
        duration=3.0,
    )

.. _unreadable-files:

Unreadable Files
----------------

One truncated, zero-byte or permission-denied file in a 100k-file corpus
otherwise ends a multi-hour run. ``on_read_error`` decides what happens
instead. It is accepted by :func:`~audiotree.sources.create_audio_dataset`,
:func:`~audiotree.sources.create_balanced_audio_dataset`, and
:class:`~audiotree.sources.AudioDataSource`.

.. list-table::
   :header-rows: 1
   :widths: 16 84

   * - ``on_read_error``
     - Behavior
   * - ``"raise"``
     - **Default.** Raise ``AudioReadError`` naming the file.
   * - ``"warn"``
     - Drop the item, and emit a :class:`UserWarning` naming the file and the
       original error.
   * - ``"skip"``
     - Drop the item without warning, for a corpus already known to contain
       junk, where a warning per epoch per bad file is just noise.

Failing loudly, by name
~~~~~~~~~~~~~~~~~~~~~~~

Under the default policy, every read failure surfaces as ``AudioReadError``
regardless of which layer of the decoding stack noticed. That matters because
the underlying exception frequently does not identify the file: a truncated
header sends librosa down its ``audioread`` fallback, whose ``EOFError`` or
``NoBackendError`` has an empty ``str()`` — a blank traceback line at the end of
a long run.

``AudioReadError`` subclasses :class:`OSError`, carries the path as
``file_path``, and keeps whatever the decoder raised as ``__cause__``:

.. testsetup:: readerrors

    # Hidden setup: a corpus of two readable files and one that is not. A
    # zero-byte ``.wav`` is the cheapest way to make a file that the corpus
    # scan finds and the decoder cannot open.
    import os
    import tempfile
    import numpy as np
    import soundfile

    _rng = np.random.default_rng(0)
    _mixed_dir = tempfile.mkdtemp()
    for _i in range(2):
        soundfile.write(
            os.path.join(_mixed_dir, f"good_{_i}.wav"),
            (0.1 * _rng.standard_normal((44_100, 1))).astype(np.float32),
            44_100,
        )
    open(os.path.join(_mixed_dir, "broken.wav"), "wb").close()

.. testcode:: readerrors

    import os
    from audiotree.sources import create_audio_dataset

    strict = create_audio_dataset(_mixed_dir, sample_rate=44100, duration=1.0)

    try:
        items = [strict[i] for i in range(len(strict))]
    except OSError as err:  # AudioReadError is an OSError
        print(type(err).__name__)
        print(os.path.basename(err.file_path))

.. testoutput:: readerrors

    AudioReadError
    broken.wav

Dropped, the grain way
~~~~~~~~~~~~~~~~~~~~~~

The two non-raising policies drop the broken item using grain's own
convention: the loader returns ``None``, later ``.map()`` stages are never
called on it, and ``to_iter_dataset()`` skips it, so batches refill with real
audio. Nothing synthetic ever enters the pipeline, and no marker needs to
travel with the good items.

The consequence to know about is on the *random access* side: under a
non-raising policy ``ds[i]`` returns ``None`` for a broken file, so a loop
that indexes the dataset directly must expect it. The iteration path needs no
handling at all:

.. testcode:: readerrors

    tolerant = create_audio_dataset(
        _mixed_dir, sample_rate=44100, duration=1.0, on_read_error="skip"
    )

    # Random access: the broken file's slot reads as None.
    items = [tolerant[i] for i in range(len(tolerant))]
    print([item is None for item in items])

    # Iteration: grain skips the None slots by itself.
    print(sum(1 for item in tolerant.to_iter_dataset()))

.. testoutput:: readerrors

    [True, False, False]
    2

One combination is refused up front with a ``ValueError`` rather than
half-supported: ``window`` with a non-raising policy. Windowed sampling reads
every file's duration before the first item to build its window index, and a
file that cannot be read has no duration to contribute; see
:ref:`windowed_datasets`.

:class:`~audiotree.sources.AudioDataSource` takes the same keyword-only
argument, exposes it as ``self.on_read_error``, and its ``filter*()`` views
inherit it.

Time-Aligned Annotations
------------------------

When you take a random excerpt of a long recording, you often need the matching
slice of a time-aligned annotation — a pianoroll, MIDI, F0 curve, or label
track. Each loaded ``AudioTree`` records where its excerpt came from: the
``.offset`` provenance (start time in seconds) and ``.filepath``; the excerpt
length is ``samples / sample_rate``. A `Grain`_ ``.map`` step can
use those to load and slice the aligned annotation.

In this example each ``<name>.wav`` has a sibling ``<name>.npy`` holding a
``(128, frames)`` pianoroll at a known frame rate:

.. testsetup:: pianoroll

    import os
    import tempfile
    import numpy as np
    import soundfile

    # A few fake recordings, each with a sibling ``<name>.npy`` pianoroll. The
    # pianoroll's first row stores the frame index purely so this example can
    # confirm the slice lines up with the excerpt's offset; real pianorolls hold
    # note activity.
    _piano_dir = tempfile.mkdtemp()
    _rng = np.random.default_rng(0)
    _SR, _TOTAL_S, _FPS = 16000, 10.0, 100
    for _i in range(3):
        soundfile.write(
            os.path.join(_piano_dir, f"{_i}.wav"),
            (0.1 * _rng.standard_normal(int(_SR * _TOTAL_S))).astype(np.float32),
            _SR,
        )
        _roll = np.zeros((128, round(_TOTAL_S * _FPS)), np.float32)
        _roll[0, :] = np.arange(round(_TOTAL_S * _FPS))
        np.save(os.path.join(_piano_dir, f"{_i}.npy"), _roll)

.. testcode:: pianoroll

    import numpy as np
    from audiotree import AudioTree
    from audiotree.sources import ExcerptConfig
    from audiotree.sources import create_audio_dataset

    PIANOROLL_FPS = 100  # frame rate of the .npy pianorolls

    # Random 1-second excerpts. Each loaded AudioTree records where its excerpt
    # came from in the ``offset`` (seconds) and ``filepath`` provenance.
    ds = create_audio_dataset(
        sources=_piano_dir,
        sample_rate=16000,
        duration=1.0,
        excerpt=ExcerptConfig(strategy="random"),
    )


    def attach_pianoroll(audio: AudioTree) -> AudioTree:
        offset = float(audio.offset[0])  # excerpt start (s)
        duration = audio.samples / audio.sample_rate
        roll = np.load(audio.filepath[0].replace(".wav", ".npy"))  # (128, frames)
        start = round(offset * PIANOROLL_FPS)
        n = round(duration * PIANOROLL_FPS)
        excerpt = roll[:, start : start + n]
        # Store it in extras with a leading batch axis so it batches and
        # indexes alongside the waveform.
        return audio.replace_extras(pianoroll=excerpt[None])


    ds = ds.map(attach_pianoroll)

    item = ds[0]
    offset = float(item.offset[0])
    print("pianoroll shape:", item.extras["pianoroll"].shape)
    print(
        "aligned:", int(item.extras["pianoroll"][0, 0, 0]) == round(offset * PIANOROLL_FPS)
    )

.. testoutput:: pianoroll

    pianoroll shape: (1, 128, 100)
    aligned: True

Because ``pianoroll`` lives in ``extras`` — an active pytree node, not just
static description — :meth:`~audiotree.AudioTree.batch` stacks the
per-excerpt pianorolls into ``(batch, 128, frames)`` right alongside the
waveform, and indexing or slicing the batch keeps them aligned.

Resumable Training
------------------

Because ``audiotree`` pipelines are Grain datasets, their iterators are
checkpointable: you can snapshot the exact read position — shuffle order and
per-excerpt RNG — and resume mid-epoch after an interruption instead of
restarting the epoch. Grain seeds each excerpt deterministically from its
element index, so restoring the position reproduces the exact same subsequent
``AudioTree`` batches.

The recommended approach is `Orbax`_, which checkpoints your model **and** the
data pipeline together (and handles distributed-training edge cases). Pass the
dataset iterator to ``grain.checkpoint.CheckpointSave`` / ``CheckpointRestore``:

.. skip-snippet-exec: writes checkpoints to an absolute path outside the corpus.

.. code-block:: python

    import grain
    import orbax.checkpoint as ocp
    from audiotree import AudioTree
    from audiotree.sources import create_audio_dataset

    ds = create_audio_dataset("/data/audio", num_epochs=None, duration=3.0)
    it = iter(ds.to_iter_dataset().batch(32, batch_fn=AudioTree.batch))

    mngr = ocp.CheckpointManager("/checkpoints")

    for step, batch in enumerate(it):
        train_step(batch)  # your training step
        if step % 1000 == 0:
            mngr.save(step, args=grain.checkpoint.CheckpointSave(it), force=True)
            mngr.wait_until_finished()  # saving is async by default

    # After a crash, rebuild the same pipeline and resume in place:
    it = iter(ds.to_iter_dataset().batch(32, batch_fn=AudioTree.batch))
    mngr.restore(mngr.latest_step(), args=grain.checkpoint.CheckpointRestore(it))

To checkpoint your model in the same step, combine the data iterator with your model state
using Orbax's ``Composite`` args. For the full reference, see Grain's `checkpointing tutorial
<https://google-grain.readthedocs.io/en/latest/tutorials/dataset_advanced_tutorial.html#checkpointing>`_.

.. _streaming-device-put:

Prefetching Batches onto an Accelerator
---------------------------------------

Iterating a batched pipeline yields host (NumPy-backed) AudioTrees. To feed the
model batches that already live on the GPU/TPU — and to overlap that transfer with
the training step — wrap the ``IterDataset`` with
:func:`grain.experimental.device_put`. It double-buffers: while the current batch
trains, the next is staged on the device. Because an AudioTree is a Pytree, every
array leaf (``waveform``, each ``extras`` array, and the encoded provenance
container) arrives on-device as a ``jax.Array``:

.. testcode::

    import jax
    import grain
    from audiotree import AudioTree
    from audiotree.sources import create_audio_dataset

    ds = create_audio_dataset(sources=_speech_dir, sample_rate=44100, duration=1.0)
    iter_ds = ds.to_iter_dataset().batch(2, batch_fn=AudioTree.batch)

    # device=None targets the default JAX device (a GPU/TPU when present).
    # cpu_buffer_size / device_buffer_size set how many batches are staged in host
    # and device memory.
    device_ds = grain.experimental.device_put(
        iter_ds, device=None, cpu_buffer_size=4, device_buffer_size=2
    )

    for batch in device_ds:
        print(isinstance(batch.waveform, jax.Array))
        print(batch.waveform.shape)
        break

.. testoutput::

    True
    (2, 1, 44100)

Add it as the final stage of the pipeline (after ``.batch()`` and
``.to_iter_dataset()``), in place of manually calling :func:`jax.device_put` on
each batch inside the training loop.

.. _reproducibility:

Reproducibility
---------------

What a fixed seed buys you:

* For a fixed corpus, a fixed seed, and pinned dependency versions, ``ds[i]``
  returns the same audio every time. Grain derives a per-index seed rather
  than advancing global state, and
  :func:`~audiotree.sources.find_audio_files` returns a sorted, de-duplicated
  list, so the file order that seeding is applied to does not depend on the
  filesystem or the machine.
* ``shuffle_seed`` and ``excerpt_seed`` are separable: two datasets with the
  same ``shuffle_seed`` and different ``excerpt_seed`` visit files in the same
  order and draw different excerpts.

What it does not buy you:

* **Identical bytes across versions.** An audiotree, ``grain``,
  ``librosa``/``soxr``, ``jaxloudnorm`` or ``scipy`` upgrade may change
  decoded audio, resampled output, or loudness in the last bits — the
  resamplers and loudness meters are third-party. Pin your environment if you
  need byte-stability; hash your own pre-rendered dataset if you need it
  *checked*.
* **NumPy/JAX engine equality.** The two loudness engines selectable with
  ``replace_lufs(engine=...)`` agree closely but not bit-for-bit (the exact
  IIR K-weighting meter on the CPU versus ``jaxloudnorm``'s FIR
  approximation), and the same caution applies to the two transform backends
  generally.
* **Batch composition under multiprocessing.** With ``.batch()`` upstream of
  ``.mp_prefetch()``, which items share a batch depends on ``num_workers``;
  batching after ``mp_prefetch`` keeps composition independent of the worker
  count (see :ref:`transform_chaining`). The set of items seen per epoch is
  unaffected either way.
* **Cross-platform bit equality.** Different CPUs, XLA backends, and BLAS
  builds are not expected to agree bit for bit.

If a result must be reproducible years later, pre-render it with
:class:`~audiotree.tree_writer.TreeWriter` and keep the bytes, rather than
relying on re-running the pipeline.

Next
----

You can now load AudioTrees from disk into batched, device-ready pipelines. The
next chapter, :ref:`transform_chaining`, adds augmentations — the transforms you
chain onto a dataset with ``.map()`` and ``.random_map()``.

.. _Grain: https://github.com/google/grain
.. _Orbax: https://orbax.readthedocs.io/en/latest/
