.. role:: python(code)
     :language: python
     :class: highlight

.. :tocdepth: 2

.. _balanced_datasets:

Balanced Datasets
=================

The :func:`~audiotree.sources.create_balanced_audio_dataset` function creates datasets that sample from multiple groups with specified weights, making it ideal for training ML models on diverse audio data.

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

Source Tracking
---------------

Each loaded :class:`~audiotree.core.AudioTree` has a ``source`` property indicating which group it came from:

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

This is useful for:

- Logging during training
- Conditional processing based on source type
- Debugging data pipeline issues

Deterministic Sampling
----------------------

Use ``shuffle=False`` and a fixed ``seed`` for reproducible iteration:

.. code-block:: python

    ds = create_balanced_audio_dataset(
        sources={"speech": ["/data/speech"], "music": ["/data/music"]},
        shuffle=False,
        seed=42,
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
        repeat=True,
        sample_rate=44100,
        duration=3.0,
    )

    # Mix with file-based sources
    ds = create_balanced_audio_dataset(
        sources={"speech": ["/data/speech"]},
        datasets={"preprocessed": preprocessed_ds},
        weights={"speech": 0.7, "preprocessed": 0.3},
    )

Saliency-Based Loading
-----------------------

Use saliency to randomly select louder sections of audio:

.. code-block:: python

    from audiotree import AudioTree
    from audiotree.sources import create_balanced_audio_dataset
    from audiotree.core import SaliencyParams

    loudness_cutoff = -40  # Only select sections above -40 LUFS
    saliency_params = SaliencyParams(
        enabled=True,
        loudness_cutoff=loudness_cutoff,
        num_tries=10,  # Try up to 10 random positions
    )

    ds = create_balanced_audio_dataset(
        sources={"speech": ["/data/speech"], "music": ["/data/music"]},
        saliency_params=saliency_params,
        sample_rate=44100,
        duration=3.0,
    )

    # Saliency is best-effort: after num_tries it returns the loudest excerpt it
    # found, even if still below the cutoff. Filter to *guarantee* the floor,
    # dropping files that never clear it. (Saliency populates .lufs on each excerpt.)
    ds = ds.filter(lambda audio_tree: audio_tree.lufs[0] > loudness_cutoff)
    ds = ds.to_iter_dataset()
    ds = ds.batch(32, batch_fn=AudioTree.batch)

This is particularly useful for training on long audio files where you want to avoid silent sections.

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
    from collections import Counter
    sources = [ds[i].source[0] for i in range(len(ds))]
    counts = Counter(sources)

    # Verify proportions (should be within ±1% for 10k samples)
    print(counts)
    # {'group1': ~5000, 'group2': ~3000, 'group3': ~2000}

Best Practices
--------------

1. **Organize by Semantic Groups**: Group files by meaningful categories (genre, speaker type, recording quality), not just directory structure
2. **Use Hierarchical Directories**: Organize subdirectories by speaker, session, or other metadata
3. **Set Appropriate Weights**: Balance based on importance, not just available data
4. **Monitor Source Distribution**: Track ``source`` property during training to verify balancing
5. **Use Saliency for Long Files**: Enable saliency when working with files longer than your target duration

See Also
--------

- :func:`~audiotree.sources.create_audio_dataset` - For simple, unbalanced loading
- :class:`~audiotree.core.SaliencyParams` - For saliency-based excerpt selection
- :ref:`multiprocessing` - For parallel data loading
