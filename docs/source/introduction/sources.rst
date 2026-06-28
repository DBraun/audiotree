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

External Examples
-----------------

For production usage examples, see `DAC-JAX's input_pipeline.py <https://github.com/DBraun/DAC-JAX/blob/main/scripts/input_pipeline.py>`_.

.. _ArgBind: https://github.com/pseeth/argbind/
.. _DAC-JAX: https://github.com/DBraun/DAC-JAX
.. _Grain: https://github.com/google/grain
