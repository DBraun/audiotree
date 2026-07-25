.. role:: python(code)
     :language: python
     :class: highlight

.. :tocdepth: 2

.. _windowed_datasets:

Windowed (Length-Aware) Datasets
================================

:func:`~audiotree.sources.create_windowed_audio_dataset` samples **windows**
instead of files. It solves a tension that appears when a corpus mixes short and
long files (e.g. 1-minute songs alongside hour-long DJ mixes):

* :func:`~audiotree.sources.create_audio_dataset` visits each file once per epoch
  and draws **one random excerpt** -- so a long file is under-covered (its content
  is sampled with replacement, unevenly) yet counts the same as a short file.
* Naively enumerating every window of a file covers it evenly but emits those
  windows back-to-back, so the loader **lingers** on one file for many batches and
  batch diversity collapses.

The windowed dataset gets both even coverage *and* diverse batches by flattening
every file's windows into one index and letting Grain globally shuffle it.

.. testsetup::

    # Hidden setup: a small corpus with one short and one long file, standing in
    # for "/data/audio" below so the executable examples run.
    import os
    import tempfile
    import numpy as np
    import soundfile

    _sr = 16000
    audio_dir = tempfile.mkdtemp()
    _g = np.random.default_rng(0)
    for _name, _dur in {"short": 4.0, "long": 40.0}.items():
        soundfile.write(
            os.path.join(audio_dir, f"{_name}.wav"),
            (0.1 * _g.standard_normal((int(_dur * _sr), 1))).astype(np.float32),
            _sr,
        )

Basic Usage
-----------

.. testcode::

    from audiotree.sources import create_windowed_audio_dataset

    ds = create_windowed_audio_dataset(
        sources=audio_dir,              # one short (4s) + one long (40s) file
        duration=1.0,                   # 1-second excerpts
        hop=1.0,                        # non-overlapping windows (the default)
        alpha=1.0,                      # sample frequency proportional to length
        sample_rate=16000,
        repeat=False,                   # one finite epoch
    )

    # One epoch contains every window exactly once: 4 + 40 = 44 slots.
    print(len(ds))
    print(ds[0].waveform.shape)

.. testoutput::

    44
    (1, 1, 16000)

The ``alpha`` Knob
------------------

``alpha`` controls how a file's sampling frequency scales with its length. Each
file contributes ``m_i = round(n_i ** alpha)`` slots, where ``n_i`` is its number
of natural ``hop`` windows:

* ``alpha = 0`` -- one slot per file (**uniform per file**, like the file-as-unit
  default).
* ``alpha = 1`` -- one slot per window (**proportional to length**; even coverage
  per second of audio).
* ``0 < alpha < 1`` -- in between (``alpha = 0.5`` is ``sqrt(length)``).

.. testcode::

    for alpha in (0.0, 0.5, 1.0):
        ds = create_windowed_audio_dataset(
            sources=audio_dir, duration=1.0, hop=1.0, alpha=alpha,
            sample_rate=16000, repeat=False,
        )
        print(alpha, len(ds))

.. testoutput::

    0.0 2
    0.5 8
    1.0 44

At ``alpha = 1`` the 40-second file's 40 windows are **scattered uniformly**
across the epoch by Grain's global shuffle, so batches stay diverse instead of
lingering on that file.

Jitter and Coverage
-------------------

Each slot owns a contiguous ``stride`` of its file's timeline. With
``jitter=True`` (the default) every draw picks a random offset *within* its
stride, so the slots tile the whole file while excerpts never repeat their exact
boundaries across epochs -- continuous variety with even coverage at any
``alpha``. Offsets are confined to ``[0, file_duration - duration]``, so windows
never run past end-of-file (clamping is built in).

Caching Durations
-----------------

Building the slot index needs each file's duration. These are read from headers
(no decode) and can be cached so retuning ``alpha``/``hop``/``duration`` never
re-reads the corpus:

.. code-block:: python

    from audiotree.sources import scan_durations, create_windowed_audio_dataset

    durations = scan_durations(filepaths)        # {path: seconds}, header-only
    # persist `durations` (e.g. JSON) and pass it back on later runs:
    ds = create_windowed_audio_dataset(filepaths=filepaths, durations=durations)

Loudness-Based Saliency (Optional)
----------------------------------

For uncurated corpora you can drop quiet windows up front. Precompute a per-file
**windowed-LUFS** cache once (computed on the CPU, no JAX/GPU work), then point
the dataset at it -- slots below ``lufs_cutoff`` are removed at build time,
so no loudness search happens during training:

.. code-block:: python

    from audiotree.sources import build_window_lufs_cache, create_windowed_audio_dataset

    # One-time offline pass. Ragged per-file LUFS arrays are stored as bagz.
    build_window_lufs_cache(
        filepaths,
        lufs_window_sec=1.0,
        out_dir="lufs_cache/",
        sample_rate=44100,   # match the dataset so LUFS reflects the loaded audio
        mono=True,
    )

    ds = create_windowed_audio_dataset(
        sources="/data/audio",
        duration=1.0,
        alpha=0.5,
        sample_rate=44100,
        mono=True,
        lufs_cache="lufs_cache/",   # supplies durations + LUFS + window
        lufs_cutoff=-40.0,              # keep windows at or above -40 LUFS
    )

The cache also stores durations (so it doubles as the duration cache) and the
``sample_rate``/``mono`` it was measured with; the dataset raises if those don't
match what it loads, since the cutoff would otherwise apply to a different signal.
Well-curated data needs none of this -- omit ``lufs_cache`` and no filtering
is done.

Composing with Balanced Datasets
--------------------------------

Pass a :class:`~audiotree.sources.WindowParams` to
:func:`~audiotree.sources.create_balanced_audio_dataset` to build every group
with windowed sampling. The group ``weights`` balance *across* categories while
``alpha`` controls length bias *within* each category -- the two compose
multiplicatively:

.. code-block:: python

    from audiotree.sources import create_balanced_audio_dataset, WindowParams

    ds = create_balanced_audio_dataset(
        sources={"speech": ["/data/speech"], "music": ["/data/music"]},
        weights={"speech": 0.7, "music": 0.3},   # across-group balance
        sample_rate=44100,
        mono=True,
        window_params=WindowParams(duration=1.0, alpha=0.5),  # within-group coverage
    )
