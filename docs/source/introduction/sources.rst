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

**Quick Start**

.. code-block:: python

    from audiotree.sources import create_balanced_audio_dataset

    # Simple balanced dataset
    ds = create_balanced_audio_dataset(
        sources={
            "speech": ["/data/speech"],
            "music": ["/data/music"],
        },
        sample_rate=44100,
        duration=3.0,
    )

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