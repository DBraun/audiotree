.. role:: python(code)
     :language: python
     :class: highlight

audiotree.sources
===========================

..

.. ---------------------------

.. Documented against the ``audiotree.sources`` package rather than the
   ``audiotree.sources.core`` / ``.windowed`` submodules the objects live in, so
   that the public spelling — ``audiotree.sources.create_audio_dataset`` — is the
   one cross-references resolve against. ``ExcerptConfig`` is deliberately
   absent: it is documented once, on the :doc:`core` page.

.. automodule:: audiotree.sources
   :imported-members:
   :members: find_audio_files, create_audio_dataset,
      create_balanced_audio_dataset, create_windowed_audio_dataset,
      WindowParams, WindowLufsCache, build_window_lufs_cache,
      precompute_window_lufs, save_window_lufs, load_window_lufs,
      scan_durations
