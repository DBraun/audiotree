.. role:: python(code)
     :language: python
     :class: highlight

.. :tocdepth: 2

.. _sources:

Data Sources
=========================

..

.. ---------------------------

Data Sources in ``audiotree.sources`` are `Grain`_ `data sources <https://github.com/google/grain/blob/main/docs/data_sources.md>`_
that are specially designed for audio. Grain is a new library for dataset operations in JAX with no TensorFlow dependency.

The main functions are :func:`~audiotree.sources.create_audio_dataset` for simple loading and
:func:`~audiotree.sources.create_balanced_audio_dataset` for balanced multi-group sampling.

.. testsetup::

    # Hidden setup: stand in for the "/data/..." directories below with temp
    # dirs of small synthetic WAVs, so the Quick Start example runs.
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

For detailed information on balanced datasets, hierarchical directories, and weight-based sampling,
see :ref:`balanced_datasets`.

For parallel data loading with multiprocessing, see :ref:`multiprocessing`.

Additional Features
-------------------

**Saliency-Based Loading**

Use saliency to select louder sections of audio:

.. code-block:: python

    from audiotree.sources import create_balanced_audio_dataset
    from audiotree.core import SaliencyParams

    saliency_params = SaliencyParams(
        enabled=True,
        loudness_cutoff=-40,  # Only select sections above -40 LUFS
        num_tries=10,
    )

    ds = create_balanced_audio_dataset(
        sources={
            "speech": ["/data/speech"],
            "music": ["/data/music"],
        },
        saliency_params=saliency_params,
        sample_rate=44100,
        duration=3.0,
    )

**File Extensions**

Customize which file types to load:

.. code-block:: python

    ds = create_balanced_audio_dataset(
        sources={"audio": ["/data/audio"]},
        extensions=[".wav", ".flac", ".mp3", ".ogg"],
        sample_rate=44100,
        duration=3.0,
    )

**Multiple Directories per Group**

.. code-block:: python

    ds = create_balanced_audio_dataset(
        sources={
            "speech": [
                "/data/vctk",
                "/data/librispeech",
                "/data/common_voice",
            ],
            "music": [
                "/data/musdb/train",
                "/data/jamendo",
            ],
        },
        weights={"speech": 0.7, "music": 0.3},
        sample_rate=44100,
        duration=3.0,
    )

Time-Aligned Annotations
------------------------

When you take a random excerpt of a long recording, you often need the matching
slice of a time-aligned annotation — a pianoroll, MIDI, F0 curve, or label
track. Each loaded ``AudioTree`` records where its excerpt came from:
``metadata["offset"]`` (start time in seconds) and ``filepath``; the excerpt
length is ``samples / sample_rate``. A `Grain`_ ``.map`` step can use those to
load and slice the aligned annotation.

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
    from audiotree.core import SaliencyParams
    from audiotree.sources import create_audio_dataset

    PIANOROLL_FPS = 100  # frame rate of the .npy pianorolls

    # Random 1-second excerpts. Each loaded AudioTree records where its excerpt
    # came from in ``metadata["offset"]`` (seconds) and ``filepath``.
    ds = create_audio_dataset(
        sources=_piano_dir,
        sample_rate=16000,
        duration=1.0,
        saliency_params=SaliencyParams(enabled=True, loudness_cutoff=None),
    )

    def attach_pianoroll(audio_tree: AudioTree) -> AudioTree:
        offset = float(audio_tree.metadata["offset"][0])        # excerpt start (s)
        duration = audio_tree.samples / audio_tree.sample_rate
        roll = np.load(audio_tree.filepath[0].replace(".wav", ".npy"))  # (128, frames)
        start = round(offset * PIANOROLL_FPS)
        n = round(duration * PIANOROLL_FPS)
        excerpt = roll[:, start : start + n]
        # Store it in metadata with a leading batch axis so it batches and
        # indexes alongside the waveform.
        return audio_tree.replace(
            metadata={**audio_tree.metadata, "pianoroll": excerpt[None]}
        )

    ds = ds.map(attach_pianoroll)

    item = ds[0]
    offset = float(item.metadata["offset"][0])
    print("pianoroll shape:", item.metadata["pianoroll"].shape)
    print("aligned:", int(item.metadata["pianoroll"][0, 0, 0]) == round(offset * PIANOROLL_FPS))

.. testoutput:: pianoroll

    pianoroll shape: (1, 128, 100)
    aligned: True

Because ``pianoroll`` lives in ``metadata`` — an active pytree node, not just
static description — :meth:`~audiotree.core.AudioTree.batch` stacks the
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

.. code-block:: python

    import grain
    import orbax.checkpoint as ocp
    from audiotree import AudioTree
    from audiotree.sources import create_audio_dataset

    ds = create_audio_dataset("/data/audio", repeat=True, duration=3.0)
    it = iter(ds.to_iter_dataset().batch(32, batch_fn=AudioTree.batch))

    mngr = ocp.CheckpointManager("/checkpoints")

    for step, batch in enumerate(it):
        train_step(batch)                       # your training step
        if step % 1000 == 0:
            mngr.save(step, args=grain.checkpoint.CheckpointSave(it), force=True)
            mngr.wait_until_finished()          # saving is async by default

    # After a crash, rebuild the same pipeline and resume in place:
    it = iter(ds.to_iter_dataset().batch(32, batch_fn=AudioTree.batch))
    mngr.restore(mngr.latest_step(), args=grain.checkpoint.CheckpointRestore(it))

``orbax-checkpoint`` ships transitively with the JAX stack, so no extra
dependency is required. To checkpoint your model in the same step, combine the
data iterator with your model state using Orbax's ``Composite`` args. For the
full reference, see Grain's `checkpointing tutorial
<https://github.com/google/grain/blob/main/docs/tutorials/dataset_advanced_tutorial.md>`_.

External Examples
-----------------

For production usage examples, see `DAC-JAX's input_pipeline.py <https://github.com/DBraun/DAC-JAX/blob/main/scripts/input_pipeline.py>`_.

.. _ArgBind: https://github.com/pseeth/argbind/
.. _DAC-JAX: https://github.com/DBraun/DAC-JAX
.. _Grain: https://github.com/google/grain
.. _Orbax: https://orbax.readthedocs.io/en/latest/
