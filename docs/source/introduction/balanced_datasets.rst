.. role:: python(code)
     :language: python
     :class: highlight

.. :tocdepth: 2

.. _balanced_datasets:

Balanced Datasets
=================

:func:`~audiotree.sources.create_balanced_audio_dataset` builds a dataset that
samples from several groups of files at weights you choose, regardless of how
many files each group actually contains.

.. testsetup::

    # Hidden setup: stand in for the "/data/..." directories used below with
    # temp dirs of small synthetic WAVs, so the executable examples run.
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

Basic Usage
-----------

Create a balanced dataset from two audio directories:

.. code-block:: python

    from collections import Counter

    from audiotree.sources import create_balanced_audio_dataset

    ds = create_balanced_audio_dataset(
        sources={
            "speech": ["/data/speech"],
            "music": ["/data/music"],
        },
        sample_rate=44100,
        duration=3.0,
    )

With equal weights (default), both groups will appear equally in the dataset, regardless of how many files each directory contains.

Custom Weights
--------------

Control the proportion of each group:

.. code-block:: python

    ds = create_balanced_audio_dataset(
        sources={
            "speech": ["/data/speech"],
            "music": ["/data/music"],
            "effects": ["/data/sound_effects"],
        },
        weights={"speech": 0.5, "music": 0.3, "effects": 0.2},
        sample_rate=44100,
        duration=3.0,
    )

This creates a dataset with 50% speech, 30% music, and 20% sound effects.

Hierarchical Directories
-------------------------

The function automatically discovers all audio files in nested subdirectories:

.. code-block:: none

    /data/speech/
    ├── speaker1/
    │   ├── session1/
    │   │   ├── audio_001.wav
    │   │   └── audio_002.wav
    │   └── session2/
    │       ├── audio_003.wav
    │       └── audio_004.wav
    └── speaker2/
        ├── recording1.wav
        └── recording2.wav

All files under ``/data/speech/`` are aggregated into the "speech" group:

.. code-block:: python

    ds = create_balanced_audio_dataset(
        sources={
            "speech": ["/data/speech"],  # Finds all .wav and .flac files recursively
        },
        sample_rate=44100,
        duration=3.0,
    )

Handling Unbalanced Groups
---------------------------

The balancing works even with drastically different file counts:

.. code-block:: python

    # Assume:
    # /data/large_dataset has 10,000 files
    # /data/small_dataset has 100 files

    ds = create_balanced_audio_dataset(
        sources={
            "large": ["/data/large_dataset"],
            "small": ["/data/small_dataset"],
        },
        weights={"large": 0.5, "small": 0.5},  # 50/50 split
        sample_rate=44100,
        duration=3.0,
    )

Files from the small dataset will be repeated ~10x to match the large dataset's representation.

Multiple Directories per Group
-------------------------------

Combine multiple directories into a single group:

.. code-block:: python

    ds = create_balanced_audio_dataset(
        sources={
            "speech": [
                "/data/vctk",
                "/data/librispeech",
                "/data/common_voice",
            ],
            "music": [
                "/data/musdb",
                "/data/fma",
            ],
        },
        weights={"speech": 0.7, "music": 0.3},
        sample_rate=44100,
        duration=3.0,
    )

All files from the three speech datasets are aggregated as one "speech" group.
Entries can also be individual files or glob patterns — anything
:func:`~audiotree.sources.find_audio_files` accepts, e.g.
``"/data/musdb18hq/train/*/mixture.wav"`` or a recursive ``"/data/**/vocals.wav"``.

Source Tracking
---------------

Each loaded :class:`~audiotree.AudioTree` has a ``source`` property indicating which group it came from:

.. testcode::

    from audiotree.sources import create_balanced_audio_dataset

    ds = create_balanced_audio_dataset(
        sources={"speech": [_speech_dir], "music": [_music_dir]},
        sample_rate=44100,
        duration=1.0,
    )

    item = ds[0]
    print(item.source)  # which group this item came from
    # item.filepath holds the absolute path(s), e.g., ['/data/speech/.../audio_001.wav']

.. testoutput::

    ['speech']

``source`` is useful for logging during training, for branching on the group in
a later ``.map()``, and for checking that the balancing is doing what you asked.

Deterministic Sampling
----------------------

Use ``shuffle=False`` and a fixed ``excerpt_seed`` for reproducible iteration.
The seed is split in two: ``shuffle_seed`` fixes the file order (and does nothing
once ``shuffle=False``), while ``excerpt_seed`` fixes which excerpt is drawn from
each file. ``excerpt_seed`` defaults to ``shuffle_seed``, so passing one seed is
usually enough:

.. code-block:: python

    ds = create_balanced_audio_dataset(
        sources={"speech": ["/data/speech"], "music": ["/data/music"]},
        shuffle=False,
        excerpt_seed=42,
        sample_rate=44100,
        duration=3.0,
    )

This is useful for pre-rendering datasets or validation sets.

Mixing with Pre-Constructed Datasets
-------------------------------------

Combine file-based sources with existing Grain datasets:

.. code-block:: python

    from audiotree.sources import create_audio_dataset, create_balanced_audio_dataset

    # Pre-construct a dataset (perhaps with custom processing)
    preprocessed_ds = create_audio_dataset(
        sources="/data/preprocessed",
        num_epochs=None,
        sample_rate=44100,
        duration=3.0,
    )

    # Mix with file-based sources
    ds = create_balanced_audio_dataset(
        sources={"speech": ["/data/speech"]},
        datasets={"preprocessed": preprocessed_ds},
        weights={"speech": 0.7, "preprocessed": 0.3},
    )

Every item of a pre-built dataset is stamped with the same per-item provenance
schema the file-based groups carry, so batches spanning both collate: its
``source`` becomes the group name, and — for items built without file
provenance, e.g. with :meth:`~audiotree.AudioTree.create` — a missing
``filepath`` is filled with empty strings and a missing ``offset`` with
``NaN``, the honest spellings of "no source file".

Excerpt Selection
-----------------

The ``excerpt=`` argument works here exactly as it does for
:func:`~audiotree.sources.create_audio_dataset` (see :ref:`sources`): the
default draws a uniformly random offset per read. For corpora with long quiet
stretches, the ``"loudest"`` strategy searches for a loud section instead:

.. code-block:: python

    from audiotree import AudioTree
    from audiotree.sources import create_balanced_audio_dataset
    from audiotree.sources import ExcerptConfig

    excerpt = ExcerptConfig(
        strategy="loudest",
        lufs_cutoff=-40,  # stop searching once a section exceeds -40 LUFS
        num_tries=10,  # candidate offsets per read
        on_failure="skip",  # drop files whose best candidate stays below it
    )

    ds = create_balanced_audio_dataset(
        sources={"speech": ["/data/speech"], "music": ["/data/music"]},
        excerpt=excerpt,
        sample_rate=44100,
        duration=3.0,
    )
    iter_ds = ds.to_iter_dataset().batch(32, batch_fn=AudioTree.batch)

The search is best-effort per file: after ``num_tries`` candidates the loudest
one wins. ``on_failure`` decides what happens when even that one is below the
cutoff. The default ``"keep"`` returns it anyway; ``"skip"`` returns ``None``,
which grain drops at iteration, so batches hold only excerpts that genuinely
clear the floor; ``"raise"`` treats it as an error. For predicates beyond the
search's own cutoff, the general tool is a filter on the measured loudness
(the search fills ``.lufs`` on every excerpt it returns), e.g.
``ds.filter(lambda audio: audio.lufs[0] > -35)``.

This matters most for long files, where a purely random excerpt has a real
chance of landing in silence.

File Extensions
---------------

By default, ``.wav`` and ``.flac`` files are discovered. Customize this:

.. code-block:: python

    ds = create_balanced_audio_dataset(
        sources={"audio": ["/data/audio"]},
        extensions=[".wav", ".flac", ".mp3", ".ogg"],
        sample_rate=44100,
        duration=3.0,
    )

Statistical Validation
----------------------

Over large sample sizes, the actual distribution closely matches requested weights:

.. skip-snippet-exec: ``len()`` of a repeated dataset is unbounded.

.. code-block:: python

    ds = create_balanced_audio_dataset(
        sources={
            "group1": ["/data/group1"],
            "group2": ["/data/group2"],
            "group3": ["/data/group3"],
        },
        weights={"group1": 0.5, "group2": 0.3, "group3": 0.2},
        sample_rate=44100,
        duration=3.0,
    )

    # Count occurrences
    sources = [ds[i].source[0] for i in range(len(ds))]
    counts = Counter(sources)

    # Verify proportions (should be within ±1% for 10k samples)
    print(counts)
    # {'group1': ~5000, 'group2': ~3000, 'group3': ~2000}

When choosing groups, make them semantic (genre, speaker type, recording
quality), not whatever the directory layout happens to be, and pick weights by
how much each group should matter to the model rather than by how much data you
happen to have; correcting for that imbalance is the whole point of the
function. During training, an occasional ``Counter`` over ``item.source`` is a
cheap check that the actual distribution matches the weights you asked for.

See Also
--------

- :func:`~audiotree.sources.create_audio_dataset` - For simple, unbalanced loading
- :class:`~audiotree.ExcerptConfig` - For saliency-based excerpt selection
- :ref:`multiprocessing` - For parallel data loading
