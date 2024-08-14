.. role:: python(code)
     :language: python
     :class: highlight

.. :tocdepth: 2

.. _datasources:

Data Sources
=========================

..  

.. ---------------------------

Data Sources in ``audiotree.datasources`` are `Grain`_ `data sources <https://github.com/google/grain/blob/main/docs/data_sources.md>`_
that are specially designed for audio. Grain is a new library for dataset operations in JAX with no TensorFlow dependency.

For now, there are only two types of Data Sources, but they fit many needs.
You can take a look at DAC-JAX's `input_pipeline.py <https://github.com/DBraun/DAC-JAX/blob/main/scripts/input_pipeline.py>`_ to see how they're used.

Both :class:`~audiotree.datasources.core.AudioDataSimpleSource` and :class:`~audiotree.datasources.core.AudioDataBalancedSource` are initialized with a dictionary of ``sources``.
For example, with ArgBind, the YAML might be this (adapted from `DAC <https://github.com/descriptinc/descript-audio-codec/blob/main/conf/base.yml>`_):

.. code-block:: yaml
    
    train/AudioDataSimpleSource.sources:
        speech_fb:
            - /data/daps/train
        speech_hq:
            - /data/vctk
            - /data/vocalset
            - /data/read_speech
            - /data/french_speech
        speech_uq:
            - /data/emotional_speech/
            - /data/common_voice/
            - /data/german_speech/
            - /data/russian_speech/
            - /data/spanish_speech/
        music_hq:
            - /data/musdb/train
        music_uq:
            - /data/jamendo
        general:
            - /data/audioset/data/unbalanced_train_segments/
            - /data/audioset/data/balanced_train_segments/

The second thing to know is that both :class:`~audiotree.datasources.core.AudioDataSimpleSource` and :class:`~audiotree.datasources.core.AudioDataBalancedSource`
can be initialized with an instance of :class:`~audiotree.datasources.core.SaliencyParams`. If :class:`~audiotree.datasources.core.SaliencyParams` has ``enabled``
set to ``True``, then a random section of an audio file will be selected until it meets a specified minimum loudness.

.. _ArgBind: https://github.com/pseeth/argbind/
.. _DAC-JAX: https://github.com/DBraun/DAC-JAX
.. _Grain: https://github.com/google/grain