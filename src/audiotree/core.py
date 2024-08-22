from dataclasses import field
from functools import partial
from pathlib import Path
from typing import Callable, List, Union
from typing_extensions import Self

from flax import struct
from jax import numpy as jnp
import librosa
import numpy as np
import soundfile

from .loudness import jit_integrated_loudness
from .resample import resample


@struct.dataclass
class SaliencyParams:
    """
    The parameters for saliency detection.

    Args:
        enabled (bool): Whether to enable saliency detection.
        num_tries (int): Maximum number of attempts to find a salient section of audio (default 8).
        loudness_cutoff (float): Minimum loudness cutoff in decibels for determining salient audio (default -40).
        search_function (Union[Callable, str]): The search function for determining the random offset. The default is
            ``SaliencyParams.search_uniform``. Another option is ``SaliencyParams.search_bias_early`` which gradually
            searches earlier in the file as more attempts are made.
    """

    enabled: bool = field(default=False)
    num_tries: int = 8
    loudness_cutoff: float = -40

    # Note: Although Union[Callable, str] would be a better type annotation, it doesn't work well with argbind
    search_function: str = "SaliencyParams.search_uniform"

    @staticmethod
    def search_uniform(
        rng: np.random.Generator,
        offset: float,
        duration: float,
        total_duration: float,
        attempt: int,
        max_attempts: int,
    ):
        lower_bound = max(0.0, offset)
        upper_bound = max(total_duration - duration, lower_bound)
        return rng.uniform(lower_bound, upper_bound)

    @staticmethod
    def search_bias_early(
        rng: np.random.Generator,
        offset: float,
        duration: float,
        total_duration: float,
        attempt: int,
        max_attempts: int,
    ):
        lower_bound = max(0.0, offset)
        upper_bound1 = max(total_duration - duration, lower_bound)
        # linearly interpolate the upper bound based on number of attempts so far
        alpha = attempt / (max_attempts - 1) if max_attempts > 1 else 0
        upper_bound2 = min(upper_bound1, lower_bound + duration)
        upper_bound = upper_bound1 * (1 - alpha) + upper_bound2 * alpha
        return rng.uniform(lower_bound, upper_bound)


_str_max_length = 256


@struct.dataclass
class AudioTree:
    """
    A `flax.struct.dataclass`_ for holding audio information including a waveform, sample rate, and metadata.

    The ``AudioTree`` class is inspired by Descript AudioTools's `AudioSignal`_.
        .. _AudioSignal: https://github.com/descriptinc/audiotools/blob/master/audiotools/core/audio_signal.py
        .. _flax.struct.dataclass: https://flax.readthedocs.io/en/latest/api_reference/flax.struct.html#flax.struct.dataclass

    Args:
        audio_data (jnp.ndarray): Audio waveform data in JAX numpy tensor shaped ``(Batch, Channels, Samples)``
        sample_rate (int): Sample rate of ``audio_data``, such as 44100 Hz.
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

    def replace_loudness(self) -> Self:
        """Replace ``loudness`` property with a JAX scalar."""
        loudness = jit_integrated_loudness(self.audio_data, self.sample_rate, zeros=512)
        return self.replace(loudness=loudness)

    @staticmethod
    def _encode_string(s: str):
        # Convert string to list of ASCII values and pad with 0
        encoded = [ord(char) for char in s] + [0] * (_str_max_length - len(s))
        return jnp.array([encoded])  # [1, _str_max_length]

    @staticmethod
    def _decode_string(encoded_array: jnp.ndarray):
        # Convert list of integers to characters and join them into a string
        decoded = "".join([chr(val) for val in encoded_array if val != 0])
        return decoded

    @property
    def filepath(self) -> List[str]:
        """Return a list of filepaths assuming information exists in ``metadata['filepath']]``"""
        return [self._decode_string(data) for data in self.metadata["filepath"]]

    @classmethod
    def from_file(
        cls,
        audio_path: str,
        sample_rate: int = None,
        offset: float = 0.0,
        duration: float = None,
        mono: bool = False,
        cpu: bool = False,
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

        Returns:
            AudioTree: An instance of ``AudioTree``.
        """
        data, sr = librosa.load(
            audio_path, sr=sample_rate, offset=offset, duration=duration, mono=mono
        )
        assert sr == sample_rate
        if data.ndim == 1:
            data = data[None, None, :]  # Add batch and channel dimension
        elif data.ndim == 2:
            data = data[None, :, :]  # Add batch dimension

        if duration is not None and data.shape[-1] < round(duration * sample_rate):
            pad_right = round(duration * sample_rate) - data.shape[-1]
            data = np.pad(data, ((0, 0), (0, 0), (0, pad_right)))

        if not cpu:
            data = jnp.array(data, dtype=jnp.float32)

        return cls(
            audio_data=data,
            sample_rate=sr,
            metadata={
                "filepath": cls._encode_string(audio_path),
                "offset": offset,
                "duration": duration,
            },
        )

    @classmethod
    def from_array(cls, audio_data: np.ndarray, sample_rate: int) -> Self:
        """Create an AudioTree from an audio array and a sample rate.

        Args:
            audio_data (jnp.ndarray): Audio data shaped ``(Batch, Channels, Samples)``
            sample_rate (int): Sample rate of audio data, such as 44100 Hz.

        Returns:
            AudioTree: An instance of ``AudioTree``.
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
        rng: np.random.Generator,
        offset: float = 0.0,
        duration: float = None,
        search_function: Callable = None,
        **kwargs,
    ) -> Self:
        """Create an AudioTree from a random section of audio from a file path.

        Args:
            audio_path (str): Path to audio file.
            rng (np.random.Generator): Random number generator.
            offset (float, optional): Offset in seconds to audio data.
            duration (float, optional): Duration in seconds of audio data. The audio data will be trimmed or lengthened
                as necessary.
            search_function (Callable, optional): A function that determines the random offset.
            **kwargs: Keyword arguments passed to ``AudioTree.__init__``.

        Returns:
            AudioTree: An instance of ``AudioTree``.
        """
        assert duration is not None and duration > 0
        info = soundfile.info(audio_path)
        total_duration = info.duration  # seconds

        if search_function is None:
            search_function = partial(
                SaliencyParams.search_uniform, attempts=0, max_attempts=1
            )

        random_offset = search_function(rng, offset, duration, total_duration)

        audio_signal = cls.from_file(
            audio_path=audio_path, offset=random_offset, duration=duration, **kwargs
        )

        return audio_signal

    @classmethod
    def salient_excerpt(
        cls,
        audio_path: Union[str, Path],
        rng: np.random.Generator,
        saliency_params: SaliencyParams,
        **kwargs,
    ) -> Self:
        """Create an AudioTree from a salient section of audio from a file path.

        Args:
            audio_path (str): Path to audio file.
            rng (np.random.Generator): Random number generator such as ``np.random.default_rng(42)``.
            saliency_params (SaliencyParams): Saliency parameters to use to find a salient section.
            **kwargs: Keyword arguments passed to ``AudioTree.__init__``.

        Returns:
            AudioTree: An instance of ``AudioTree``.
        """
        assert (
            "offset" not in kwargs
        ), "``salient_excerpt`` cannot be used with kwarg ``offset``."
        assert (
            "duration" in kwargs
        ), "``salient_excerpt`` must be used with kwarg ``duration``."
        if (
            not saliency_params.enabled
            or saliency_params.loudness_cutoff is None
            or np.isnan(saliency_params.loudness_cutoff)
        ):
            excerpt = cls.excerpt(audio_path, rng=rng, **kwargs)
        else:
            loudness = -np.inf
            current_try = 0
            num_tries = saliency_params.num_tries
            if isinstance(saliency_params.search_function, str):
                _search_function = eval(saliency_params.search_function)
            else:
                _search_function = saliency_params.search_function
            while loudness <= saliency_params.loudness_cutoff:
                search_function = partial(
                    _search_function,
                    attempt=current_try,
                    max_attempts=num_tries,
                )
                new_excerpt = cls.excerpt(
                    audio_path, rng=rng, search_function=search_function, **kwargs
                ).replace_loudness()
                if current_try == 0 or new_excerpt.loudness > loudness:
                    excerpt = new_excerpt
                    loudness = new_excerpt.loudness
                current_try += 1
                if num_tries is not None and current_try >= num_tries:
                    break
        return excerpt

    def to_mono(self) -> Self:
        """Reduce the ``audio_data`` to mono.

        Returns:
            AudioTree: An instance of ``AudioTree``.
        """
        audio_data = self.audio_data.mean(axis=1, keepdims=True)
        return self.replace(audio_data=audio_data, loudness=None)

    def resample(
        self,
        sample_rate: int,
        zeros: int = 24,
        rolloff: float = 0.945,
        output_length: int = None,
        full: bool = False,
    ) -> Self:
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
                Allowed values are between 0 and ``ceil(length * sample_rate / old_sr)``. When ``None`` (default) is
                specified, the floored output length will be used. In order to select the largest possible
                size, use the `full` argument.
            full (bool): return the longest possible output from the input. This can be useful
                if you chain resampling operations, and want to give the ``output_length`` only
                for the last one, while passing ``full=True`` to all the other ones.

        Returns:
            AudioTree: An instance of ``AudioTree``.
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
