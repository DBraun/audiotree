from pathlib import Path
from typing import Union

from flax import struct
from jax import numpy as jnp
import librosa
import numpy as np
import soundfile

from .loudness import jit_integrated_loudness
from .resample import resample


@struct.dataclass
class AudioTree:
    """A ``flax.struct.dataclass`` for holding audio information including a waveform, sample rate, and metadata.

    The ``AudioTree`` class is inspired by Descript AudioTools's `AudioSignal`_.
        .. _AudioSignal: https://github.com/descriptinc/audiotools/blob/master/audiotools/core/audio_signal.py

    Args:
        audio_data (jnp.ndarray): Audio waveform data in JAX numpy tensor shaped (Batch, Channels, Time)
        sample_rate (int): Sample rate of audio_data, such as 44100 Hz.
        loudness (float, optional): Loudness of the audio waveform in LUFs. Don't set this when initializing. Instead,
            use ``replace_loudness()`` to create a new AudioTree with ``loudness`` calculated.
        pitch (float, optional): The MIDI pitch where 60 is middle C.
        velocity (float, optional): The MIDI velocity between 0 and 127.
        duration (float, optional): The duration of the audio waveform in seconds.
        codes (jnp.ndarray): The neural audio codec tokens for the audio.
        metadata (dict): Any extra metadata can be placed here.
    """

    audio_data: jnp.ndarray
    sample_rate: int = struct.field(pytree_node=False)
    loudness: float = None
    pitch: float = None
    velocity: float = None
    duration: float = None
    codes: jnp.ndarray = None
    metadata: dict = struct.field(pytree_node=True, default_factory=dict)

    def replace_loudness(self):
        """Replace ``loudness`` property with a JAX scalar."""
        loudness = jit_integrated_loudness(self.audio_data, self.sample_rate, zeros=512)
        return self.replace(loudness=loudness)

    @classmethod
    def from_file(
        cls,
        audio_path: str,
        sample_rate: int = None,
        offset: float = 0,
        duration: float = None,
        mono: bool = False,
    ):
        """Create an AudioTree from an audio file path.

        Args:
            audio_path (str): Path to audio file.
            sample_rate (int, optional): Sample rate of audio data, such as 44100 Hz. If left as ``None``, the file's
                original sample rate will be used.
            offset (float, optional): Offset in seconds to audio data.
            duration (float, optional): Duration in seconds of audio data. The audio data will be trimmed or extended as
                necessary.
            mono (bool, optional): Whether to force the audio data to be single-channel.
        """
        data, sr = librosa.load(
            audio_path, sr=sample_rate, offset=offset, duration=duration, mono=mono
        )
        assert sr == sample_rate
        data = jnp.array(data, dtype=jnp.float32)
        if data.ndim == 1:
            data = data[None, None, :]  # Add batch and channel dimension
        elif data.ndim == 2:
            data = data[None, :, :]  # Add batch dimension

        if duration is not None and data.shape[-1] < round(duration * sample_rate):
            pad_right = round(duration * sample_rate) - data.shape[-1]
            data = jnp.pad(data, ((0, 0), (0, 0), (0, pad_right)))

        return cls(
            audio_data=data,
            sample_rate=sr,
            metadata={"offset": offset, "duration": duration},
        )

    @classmethod
    def from_array(cls, audio_data: np.ndarray, sample_rate: int):
        """Create an AudioTree from an audio array and a sample rate.

        Args:
            audio_data (jnp.ndarray): Audio data shaped (Batch, Channels, Time)
            sample_rate (int): Sample rate of audio data, such as 44100 Hz.
        """
        audio_data = jnp.array(audio_data, dtype=jnp.float32)
        if audio_data.ndim == 1:
            audio_data = audio_data[None, None, :]  # Add batch and channel dimension
        elif audio_data.ndim == 2:
            audio_data = audio_data[None, :, :]  # Add batch dimension
        return cls(audio_data=audio_data, sample_rate=sample_rate)

    @classmethod
    def excerpt(
        cls,
        audio_path: str,
        rng: np.random.RandomState,
        offset: float = None,
        duration: float = None,
        **kwargs,
    ):
        assert duration is not None and duration > 0
        info = soundfile.info(audio_path)
        total_duration = info.duration  # seconds

        lower_bound = 0 if offset is None else offset
        upper_bound = max(total_duration - duration, 0)
        offset = rng.uniform(lower_bound, upper_bound)

        audio_signal = cls.from_file(
            audio_path=audio_path, offset=offset, duration=duration, **kwargs
        )

        return audio_signal

    @classmethod
    def salient_excerpt(
        cls,
        audio_path: Union[str, Path],
        rng: np.random.RandomState,
        loudness_cutoff: float = None,
        num_tries: int = 8,
        **kwargs,
    ):
        assert (
            "offset" not in kwargs
        ), "``salient_excerpt`` cannot be used with kwarg ``offset``."
        assert (
            "duration" in kwargs
        ), "``salient_excerpt`` must be used with kwarg ``duration``."
        if loudness_cutoff is None:
            excerpt = cls.excerpt(audio_path, rng=rng, **kwargs)
        else:
            loudness = -np.inf
            current_try = 0
            while loudness <= loudness_cutoff:
                current_try += 1
                new_excerpt = cls.excerpt(
                    audio_path, rng=rng, **kwargs
                ).replace_loudness()
                if current_try == 1 or new_excerpt.loudness > loudness:
                    excerpt = new_excerpt
                    loudness = new_excerpt.loudness
                if num_tries is not None and current_try >= num_tries:
                    break
        return excerpt

    def to_mono(self):
        """
        Reduce the ``audio_data`` to mono.
        """
        audio_data = jnp.mean(self.audio_data, axis=1, keepdims=True)
        return self.replace(audio_data=audio_data, loudness=None)

    def resample(
        self,
        sample_rate: int,
        zeros: int = 24,
        rolloff: float = 0.945,
        output_length: int = None,
        full: bool = False,
    ):
        """
        Resample the AudioTree's ``audio_data`` to a new sample rate. The algorithm is a JAX port of ``ResampleFrac``
        from the PyTorch library `Julius`_.

        .. _Julius: https://github.com/adefossez/julius/blob/main/julius/resample.py

        Args:
            sample_rate (int): The new sample rate of audio data, such as 44100 Hz.
            zeros (int, optional): number of zero crossing to keep in the sinc filter.
            rolloff (float): use a lowpass filter that is ``rolloff * sample_rate / 2``,
                to ensure sufficient margin due to the imperfection of the FIR filter used.
                Lowering this value will reduce antialiasing, but will reduce some of the
                highest frequencies.
            output_length (None or int): This can be set to the desired output length (last dimension).
                Allowed values are between 0 and ``ceil(length * sample_rate / old_sr)``. When None (default) is
                specified, the floored output length will be used. In order to select the largest possible
                size, use the `full` argument.
            full (bool): return the longest possible output from the input. This can be useful
                if you chain resampling operations, and want to give the ``output_length`` only
                for the last one, while passing ``full=True`` to all the other ones.
        """
        if sample_rate == self.sample_rate:
            return self
        audio_data = resample(
            self.audio_data,
            self.sample_rate,
            sample_rate,
            zeros=zeros,
            rolloff=rolloff,
            output_length=output_length,
            full=full,
        )
        return self.replace(
            audio_data=audio_data, sample_rate=sample_rate, loudness=None
        )
