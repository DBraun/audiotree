from dataclasses import field
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Self, Sequence, Union

from absl import logging
from flax import struct
from jax import numpy as jnp, tree_util
import librosa
import loudness
import numpy as np
import soundfile

from .loudness import jit_integrated_loudness
from .resample import resample


@struct.dataclass
class SaliencyParams:
    """
    The parameters for saliency detection.

    When enabled, this controls how audio excerpts are selected from files. If loudness_cutoff is None
    or enabled is False, a random offset is used without loudness-based filtering.

    Args:
        enabled (bool): Whether to enable saliency detection. Defaults to True. If False, loads from
            offset=0 (deterministic). If True without loudness_cutoff, uses a random offset.
        num_tries (int): Maximum number of attempts to find a salient section of audio (default 8).
            Only used when loudness_cutoff is not None.
        loudness_cutoff (float): Minimum loudness cutoff in decibels for determining salient audio (default -40).
            If loudness_cutoff is None but SaliencyParams is enabled, uses a random offset without loudness filtering.
        search_function (Union[Callable, str]): The search function for determining the random offset. The default is
            ``SaliencyParams.search_uniform``. Another option is ``SaliencyParams.search_bias_early`` which gradually
            searches earlier in the file as more attempts are made.
    """

    enabled: bool = field(default=True)
    num_tries: int = 8
    loudness_cutoff: float = -40.0

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


# todo: configure when initializing an AudioTree instance?
_str_max_length = 256


@struct.dataclass
class AudioTree:
    """
    A `flax.struct.dataclass`_ for holding audio information including a waveform, sample rate, and metadata.

    The ``AudioTree`` class is inspired by Descript AudioTools's `AudioSignal`_.
        .. _AudioSignal: https://github.com/descriptinc/audiotools/blob/master/audiotools/core/audio_signal.py
        .. _flax.struct.dataclass: https://flax.readthedocs.io/en/latest/api_reference/flax.struct.html#flax.struct.dataclass

    Args:
        waveform (jnp.ndarray): Audio waveform data shaped ``(Samples)``, ``(Channels, Samples)``, or ``(Batch, Channels, Samples)``
        sample_rate (int): Sample rate of ``waveform``, such as 44100 Hz.
        loudness (jnp.ndarray, optional): Loudness of the audio waveform in LUFs. You may not need to set this when initializing. Instead,
            use ``replace_loudness()`` to create a new AudioTree with ``loudness`` calculated.
        pitch (jnp.ndarray, optional): The MIDI pitch where 60 is middle C. The shape is ``(Batch,)``.
        velocity (jnp.ndarray, optional): The MIDI velocity between 0 and 127. The shape is ``(Batch,)``.
        note_duration (jnp.ndarray, optional): A note duration in units of your choice.
            The value is not necessarily the same as the duration of the audio data. The shape is ``(Batch,)``.
        codes (jnp.ndarray, optional): The neural audio codec tokens for the audio.
        latents (jnp.ndarray, optional): The latent representations of the audio.
        metadata (dict): Any extra metadata can be placed here.
        filepaths (Union[str, Path, List[Union[str, Path]]] | None): List of filepaths for the batch of audio.

    Note:
        If new fields are added to this class, update ``_AUDIOTREE_FIELDS`` in ``audiotree/writer.py``.
    """

    waveform: np.ndarray
    sample_rate: int = struct.field(pytree_node=False)
    loudness: np.ndarray = None
    pitch: np.ndarray = None
    velocity: np.ndarray = None
    note_duration: np.ndarray = None
    codes: np.ndarray = None
    latents: np.ndarray = None
    metadata: dict = struct.field(pytree_node=True, default_factory=dict)

    @classmethod
    def create(
        cls,
        waveform: np.ndarray,
        sample_rate: int,
        loudness: np.ndarray = None,
        pitch: np.ndarray = None,
        velocity: np.ndarray = None,
        note_duration: np.ndarray = None,
        codes: np.ndarray = None,
        latents: np.ndarray = None,
        metadata: dict = None,
        filepaths: Union[str, Path, List[Union[str, Path]]] | None = None,
    ) -> Self:
        """Create an AudioTree with automatic audio dimensionality handling and filepath processing."""
        # Handle audio dimensionality - ensure it's (Batch, Channels, Samples)
        if waveform.ndim == 1:
            waveform = waveform[None, None, :]  # Add batch and channel dimension
        elif waveform.ndim == 2:
            waveform = waveform[None, :, :]  # Add batch dimension

        # Handle metadata and filepaths
        if metadata is None:
            metadata = {}
        else:
            metadata = metadata.copy()  # Don't modify the original dict
        
        if filepaths is not None:
            metadata["filepath"] = cls._encode_filepaths(filepaths)

        return cls(
            waveform=waveform,
            sample_rate=sample_rate,
            loudness=loudness,
            pitch=pitch,
            velocity=velocity,
            note_duration=note_duration,
            codes=codes,
            latents=latents,
            metadata=metadata,
        )

    def replace_loudness(self) -> Self:
        """Compute and set the loudness in LUFS for each item in the batch.

        Calculates the integrated loudness following ITU-R BS.1770-4 standard.
        Returns a new AudioTree with the ``loudness`` property populated.

        Returns:
            AudioTree with loudness values computed, shaped (batch_size,).

        Note:
            **Channel Limitations**: Supports up to 5 channels:

            - Mono (1 channel): Single channel
            - Stereo (2 channels): [Left, Right]
            - 5.0/5.1 Surround (5 channels): [Left, Right, Center, Left Surround, Right Surround]

            Will raise ValueError if audio has more than 5 channels.
        """
        if isinstance(self.waveform, np.ndarray):
            # integrated_loudness requires at least 400ms of audio
            min_samples = int(np.ceil(0.4 * self.sample_rate))
            waveform = self.waveform
            if waveform.shape[-1] < min_samples:
                pad_right = min_samples - waveform.shape[-1]
                waveform = np.pad(waveform, ((0, 0), (0, 0), (0, pad_right)))
            audio_transposed = np.transpose(waveform, (0, 2, 1)) # [B, T, C]
            loudness_values = []
            for audio_item in audio_transposed:
                lufs = loudness.integrated_loudness(audio_item, self.sample_rate)
                loudness_values.append(lufs)
            loudness_array = np.array(loudness_values, dtype=np.float32)
        else:
            loudness_array = jit_integrated_loudness(
                jnp.array(self.waveform), self.sample_rate, zeros=512
            )
        return self.replace(loudness=loudness_array)

    def normalize_loudness(self, target_lufs: float) -> Self:
        """Normalize audio to a target LUFS level.

        Computes the current loudness (if not already set), then scales the audio
        to achieve the target LUFS. The returned AudioTree has both updated
        ``waveform`` and ``loudness`` fields.

        Args:
            target_lufs: Target loudness in LUFS (e.g., -18.0 for broadcast standard).

        Returns:
            AudioTree with audio scaled to target LUFS and loudness updated.

        Example:
            >>> tree = AudioTree.load("audio.wav")
            >>> normalized = tree.normalize_loudness(-18.0)
            >>> print(normalized.loudness)  # Should be close to -18.0
        """
        # Ensure loudness is computed
        if self.loudness is None:
            tree = self.replace_loudness()
        else:
            tree = self

        numpy = np if isinstance(self.waveform, np.ndarray) else jnp
        linear_gain = numpy.power(10.0, (target_lufs - tree.loudness) / 20.0)
        # Cast to audio dtype to avoid float64 promotion
        linear_gain = linear_gain.astype(tree.waveform.dtype)
        # Expand gain for broadcasting: [B] -> [B, 1, 1] for [B, C, T] audio
        linear_gain = linear_gain[:, None, None]
        scaled_waveform = tree.waveform * linear_gain

        # Update loudness to target (shape [B])
        target_loudness = numpy.full(tree.loudness.shape, target_lufs, dtype=numpy.float32)
        return tree.replace(waveform=scaled_waveform, loudness=target_loudness)

    @staticmethod
    def _encode_string(s: str) -> np.ndarray:
        """Encode a single filepath *s* to an array of Unicode code points.

        The returned array is shaped ``(1, _str_max_length)`` so that multiple
        rows (filepaths) can be concatenated along *axis=0*.
        """
        s = str(s)
        encoded = [ord(char) for char in s[:_str_max_length]]
        encoded += [0] * (_str_max_length - len(encoded))
        return np.array([encoded], dtype=np.int32)  # [1, _str_max_length]

    @classmethod
    def _encode_filepaths(
        cls, paths: Union[str, Path, List[Union[str, Path]]]
    ) -> np.ndarray:
        """Vectorized helper to encode one or more *paths*.

        Args:
            paths: A single filepath or an iterable of filepaths.

        Returns
        -------
        np.ndarray
            An array shaped ``(N, _str_max_length)`` where *N* is the number of
            paths provided.
        """
        if isinstance(paths, (str, Path)):
            paths = [paths]

        encoded_rows = [cls._encode_string(p)[0] for p in paths]
        return np.stack(encoded_rows, axis=0)

    @staticmethod
    def _decode_string(encoded_array: np.ndarray) -> str:
        """Decode a single encoded filepath back to *str*."""
        decoded = "".join(chr(int(val)) for val in encoded_array if val != 0)
        return decoded

    @property
    def filepath(self) -> List[str]:
        """Return the decoded filepaths stored in ``metadata['filepath']``.

        An empty list is returned if the AudioTree does not contain any filepath
        metadata.
        """
        if "filepath" not in self.metadata:
            return []
        return [self._decode_string(data) for data in self.metadata["filepath"]]

    @property
    def source(self) -> List[str]:
        """Return the decoded source names stored in ``metadata['source']``.

        Source names indicate which data source group each item in the batch came from.
        For example, if an AudioDataSimpleSource was created with
        ``sources={"music": [...], "speech": [...]}``, this property might return
        ``["music", "music", "speech", "music"]`` for a batch of 4 items.

        An empty list is returned if the AudioTree does not contain any source
        metadata.
        """
        if "source" not in self.metadata:
            return []
        return [self._decode_string(data) for data in self.metadata["source"]]

    @property
    def samples(self) -> int:
        """Return the number of samples in the ``waveform`` (its last dimension)."""
        return self.waveform.shape[-1]

    @property
    def batch_size(self) -> int:
        """Return the size of the leading (batch) axis.

        Derived from ``waveform``, falling back to ``codes`` / ``latents`` for
        audio-less trees (e.g. token-only training examples).
        """
        for value in (self.waveform, self.codes, self.latents):
            if value is not None:
                return value.shape[0]
        raise ValueError(
            "AudioTree has no waveform, codes, or latents to infer a batch size from."
        )

    @property
    def num_channels(self) -> int:
        """Return the number of audio channels (``waveform.shape[-2]``)."""
        return self.waveform.shape[-2]

    def __len__(self) -> int:
        """Number of items in the batch (the leading axis).

        Together with ``__getitem__`` this makes an AudioTree iterable over
        its batch items, e.g. ``for item in tree: ...`` — each ``item`` is a
        batch-of-1 AudioTree.
        """
        return self.batch_size

    def __getitem__(self, key: Union[int, slice]) -> Self:
        """Index the batch axis, returning an AudioTree of the selected item(s).

        An integer key selects a single item but keeps the leading batch axis
        (a batch of 1); a slice selects a sub-batch. Every array field —
        including ``codes``, ``latents``, and the ``metadata`` arrays — is
        indexed along the same axis so the fields stay rank-aligned.
        """
        if isinstance(key, int):
            n = self.batch_size
            if key < -n or key >= n:
                # Required for the sequence-iteration protocol: `for item in
                # tree` calls __getitem__(0), (1), ... and stops only on
                # IndexError (it does NOT consult __len__).
                raise IndexError(
                    f"batch index {key} out of range for batch_size {n}"
                )
            # Use a length-1 slice rather than a scalar index so the batch
            # axis survives on every field.
            key = slice(key, key + 1 or None)

        def _is_string_list(x) -> bool:
            return (
                isinstance(x, list) and bool(x)
                and all(isinstance(s, str) for s in x)
            )

        def _index(x):
            if isinstance(x, (np.ndarray, jnp.ndarray)) or _is_string_list(x):
                return x[key]
            return x

        return tree_util.tree_map(_index, self, is_leaf=_is_string_list)

    @classmethod
    def from_file(
        cls,
        audio_path: Union[str, Path],
        sample_rate: int | None = None,
        offset: float = 0.0,
        duration: float | None = None,
        mono: bool = False,
        pad_mode: Literal["constant", "edge", "reflect", "symmetric", "wrap"] | None = "constant",
        filepaths: Union[str, Path, List[Union[str, Path]]] | None = None,
        source: str | None = None,
        metadata: Optional[Dict[str, Any]] = None,
        # AudioTree properties
        loudness: Optional[np.ndarray] = None,
        pitch: Optional[np.ndarray] = None,
        velocity: Optional[np.ndarray] = None,
        note_duration: Optional[np.ndarray] = None,
        codes: Optional[np.ndarray] = None,
        latents: Optional[np.ndarray] = None,
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
            pad_mode (Literal): If duration is not None, and duration is less than the length of the audio, then
                ``pad_mode`` controls how the audio is right-padded (numpy.pad modes). Options:
                "constant" (zeros, default), "edge" (repeat edge), "reflect" (mirror), "symmetric" (mirror with edge),
                "wrap" (circular/loop), or None (no padding).
            filepaths (Union[str, Path, List[str | Path]], optional): One or more filepaths to store in the returned
                ``AudioTree``'s metadata. If *None* (default) the provided ``audio_path`` will be used.
            source (str, optional): The source group name for this audio file (e.g., "music", "speech").
                This is stored in metadata and accessible via the ``source`` property.
            metadata (dict, optional): Additional metadata to include in the AudioTree. This metadata is merged with
                automatically generated fields (offset, note_duration, filepath).
            loudness (np.ndarray, optional): Loudness values to assign to the AudioTree.
            pitch (np.ndarray, optional): Pitch values to assign to the AudioTree.
            velocity (np.ndarray, optional): Velocity values to assign to the AudioTree.
            note_duration (np.ndarray, optional): Note note_duration values to assign to the AudioTree.
            codes (jnp.ndarray, optional): The neural audio codec tokens for the audio.
            latents (jnp.ndarray, optional): The latent representations of the audio.

        Returns:
            AudioTree: An instance of ``AudioTree``.
        """
        audio_path = Path(audio_path)

        target_length = None
        if duration is not None and sample_rate is not None:
            target_length = round(duration * sample_rate)

        data, sample_rate = librosa.load(
            str(audio_path), sr=sample_rate, offset=offset, duration=duration, mono=mono
        )

        if data.ndim == 1:
            data = data[None, None, :]  # Add batch and channel dimension
        elif data.ndim == 2:
            data = data[None, :, :]  # Add batch dimension

        if (
            target_length is not None
            and pad_mode is not None
            and data.shape[-1] < target_length
        ):
            pad_right = target_length - data.shape[-1]
            # Modes like "wrap", "reflect", "edge" require non-empty data.
            # Fall back to "constant" (zero-pad) when the time axis is empty.
            effective_pad_mode = (
                "constant" if data.shape[-1] == 0 else pad_mode
            )
            data = np.pad(
                data,
                pad_width=((0, 0), (0, 0), (0, pad_right)),
                mode=effective_pad_mode,
            )

        # Start with user-provided metadata or empty dict
        if metadata is None:
            combined_metadata = {}
        else:
            combined_metadata = metadata.copy()  # Don't modify the original

        # Add automatic metadata (these override user metadata to ensure correctness)
        combined_metadata["offset"] = np.array([offset])

        if filepaths is None:
            filepaths_to_store = [audio_path]
        else:
            # Normalize to list
            if isinstance(filepaths, (str, Path)):
                filepaths_to_store = [filepaths]
            else:
                filepaths_to_store = list(filepaths)

        combined_metadata["filepath"] = cls._encode_filepaths(filepaths_to_store)

        if source is not None:
            combined_metadata["source"] = cls._encode_filepaths([source])

        # Wrap scalar properties in arrays with batch dimension
        # This ensures consistency - all AudioTree properties should have batch dimension
        def wrap_if_scalar(val, dtype=None):
            if val is None:
                return None
            if np.isscalar(val):
                return np.array([val], dtype=dtype)
            elif isinstance(val, np.ndarray) and val.ndim == 0:
                return np.array([val.item()], dtype=dtype)
            else:
                return val

        return cls(
            waveform=data,
            sample_rate=sample_rate,
            metadata=combined_metadata,
            loudness=wrap_if_scalar(loudness, np.float32),
            pitch=wrap_if_scalar(pitch, np.float32),
            velocity=wrap_if_scalar(velocity, np.int16),
            note_duration=wrap_if_scalar(note_duration, np.float32),
            codes=wrap_if_scalar(codes),
            latents=wrap_if_scalar(latents),
        )

    @classmethod
    def from_manifest(
        cls,
        manifest_path: Union[str, Path],
        audio_dir: Optional[Union[str, Path]] = None,
        filter_fn: Optional[callable] = None,
    ) -> Self:
        """Create an AudioTree by loading all items from a manifest file.

        This loads all entries from a manifest file created by AudioWriter and
        creates a single AudioTree with all items in the batch dimension.

        Args:
            manifest_path: Path to the manifest file (NPZ format)
            audio_dir: Optional directory containing audio files. If None, uses manifest directory
            filter_fn: Optional function to filter entries. Should accept a dict entry and return bool.

        Returns:
            AudioTree with all manifest entries concatenated along batch dimension

        Example:
            >>> # Load all items from manifest
            >>> tree = AudioTree.from_manifest("output/manifest.npz")
            >>> tree.waveform.shape
            (100, 2, 44100)  # 100 items, stereo, 1 second each

            >>> # Load with filtering
            >>> tree = AudioTree.from_manifest(
            ...     "output/manifest.npz",
            ...     filter_fn=lambda entry: entry.get('loudness', -float('inf')) > -20
            ... )
        """
        manifest_path = Path(manifest_path)

        # Load manifest data
        manifest_data = np.load(manifest_path, allow_pickle=True)

        # Determine audio directory
        if audio_dir is None:
            audio_dir = manifest_path.parent
        else:
            audio_dir = Path(audio_dir)

        # Convert to list of entry dictionaries for filtering
        num_entries = len(manifest_data['index'])

        if filter_fn is not None:
            # Create a lazy dict-like object for filtering
            class LazyEntry:
                def __init__(self, data, idx):
                    self.data = data
                    self.idx = idx

                def get(self, key, default=None):
                    if key in self.data:
                        return self.data[key][self.idx]
                    return default

                def __getitem__(self, key):
                    return self.data[key][self.idx]

            # Build mask using filter function
            mask = np.array([filter_fn(LazyEntry(manifest_data, i)) for i in range(num_entries)])
            indices = np.where(mask)[0]

            if len(indices) == 0:
                raise ValueError(f"No entries match filter in manifest: {manifest_path}")
        else:
            indices = np.arange(num_entries)

        # Get metadata for reconstruction
        sample_rate = int(manifest_data['sample_rate'][indices[0]])
        channels = int(manifest_data['channels'][indices[0]])
        samples = int(manifest_data['samples'][indices[0]])
        files_written = manifest_data.get('files_written', np.ones(num_entries, dtype=bool))[indices[0]]

        # Check if audio files exist
        if files_written:
            # Load audio from files
            waveform = []
            for idx in indices:
                filename = manifest_data['filename'][idx]
                audio_path = audio_dir / filename

                if not audio_path.exists():
                    raise FileNotFoundError(f"Audio file not found: {audio_path}")

                data, _ = librosa.load(str(audio_path), sr=sample_rate, mono=False)

                if data.ndim == 1:
                    data = data[None, :]  # Add channel dimension
                elif data.ndim == 2:
                    pass  # Already (channels, samples)

                waveform.append(data)

            waveform = np.stack(waveform, axis=0)  # (batch, channels, samples)
        else:
            # No audio files - create zeros
            waveform = np.zeros((len(indices), channels, samples), dtype=np.float32)

        # Build metadata dictionary
        metadata = {}
        for key in manifest_data.keys():
            if key.startswith('metadata_'):
                # Extract metadata field
                metadata_key = key[9:]  # Remove 'metadata_' prefix
                metadata[metadata_key] = manifest_data[key][indices]

        # Build AudioTree kwargs
        tree_kwargs = {
            'sample_rate': sample_rate,
            'metadata': metadata,
        }

        # Add AudioTree fields from manifest
        from audiotree.writer import _AUDIOTREE_FIELDS
        for field_name in _AUDIOTREE_FIELDS:
            if field_name in manifest_data:
                tree_kwargs[field_name] = manifest_data[field_name][indices]

        return cls.create(waveform, **tree_kwargs)

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
                SaliencyParams.search_uniform, attempt=0, max_attempts=1
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
        if "offset" in kwargs:
            raise ValueError("``salient_excerpt`` cannot be used with kwarg ``offset``.")
        if "duration" not in kwargs:
            raise ValueError("``salient_excerpt`` must be used with kwarg ``duration``.")
        if (
            not saliency_params.enabled
            or saliency_params.loudness_cutoff is None
        ):
            excerpt = cls.excerpt(audio_path, rng=rng, **kwargs)
        else:
            # Get file info once before the loop to avoid repeated soundfile.info calls
            info = soundfile.info(audio_path)
            file_duration = info.duration

            duration = kwargs["duration"]
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
                # Inline the `excerpt` logic to reuse file_duration
                random_offset = search_function(rng, 0.0, duration, file_duration)
                new_excerpt = cls.from_file(
                    audio_path=audio_path, offset=random_offset, **kwargs
                )
                if new_excerpt.waveform.shape[-1] == 0:
                    logging.warning(
                        f"Empty audio loaded from {audio_path} at offset "
                        f"{random_offset:.2f}s (file_duration={file_duration:.2f}s)"
                    )
                new_excerpt = new_excerpt.replace_loudness()
                if current_try == 0 or new_excerpt.loudness > loudness:
                    excerpt = new_excerpt
                    loudness = new_excerpt.loudness
                current_try += 1
                if num_tries is not None and current_try >= num_tries:
                    break

        return excerpt

    def to_mono(self, strategy: Literal["average", "left", "right"] = "average") -> Self:
        """Reduce the ``waveform`` to mono.

        Args:
            strategy: ``"average"`` mixes all channels down (default);
                ``"left"`` / ``"right"`` select the corresponding channel of a
                stereo waveform.

        Returns:
            AudioTree: An instance of ``AudioTree``.
        """
        waveform = self.waveform
        B, C, T = waveform.shape
        if C == 1:
            return self
        if strategy == "average":
            waveform = waveform.mean(axis=-2, keepdims=True)
        elif strategy in ("left", "right") and C == 2:
            idx = 0 if strategy == "left" else 1
            waveform = waveform[:, idx:idx + 1, :]
        else:
            raise ValueError(
                f"Unsupported to_mono strategy {strategy!r} for {C} channels."
            )
        return self.replace(waveform=waveform, loudness=None)

    def to_stereo(self) -> Self:
        """Make the ``waveform`` stereo.

        Returns:
            AudioTree: An instance of ``AudioTree``.
        """
        waveform = self.waveform
        B, C, T = waveform.shape
        if C == 1:
            waveform = np.tile(waveform, (1, 2, 1))
            return self.replace(waveform=waveform)
        elif C == 2:
            return self
        else:
            raise ValueError(f"Cannot make AudioTree stereo if it has {C} channels.")

    def resample(
        self,
        sample_rate: int,
        zeros: int = 24,
        rolloff: float = 0.945,
        output_length: int = None,
        full: bool = False,
    ) -> Self:
        """
        Resample the AudioTree's ``waveform`` to a new sample rate. The algorithm is a JAX port of ``ResampleFrac``
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
        waveform = resample(
            self.waveform,
            self.sample_rate,
            sample_rate,
            zeros=zeros,
            rolloff=rolloff,
            output_length=output_length,
            full=full,
        )
        return self.replace(
            waveform=waveform, sample_rate=sample_rate, loudness=None
        )

    def split(self, n_splits: int) -> List[Self]:
        """Split batch dimension into a list of smaller AudioTree objects.

        Divides the batch dimension evenly into n_splits separate AudioTree objects,
        each containing a portion of the original batch.

        Args:
            n_splits: Number of AudioTree objects to create. The batch size must be
                evenly divisible by this value.

        Returns:
            List of AudioTree objects, each with batch_size = original_batch_size / n_splits.

        Example:
            >>> big_tree = AudioTree(np.zeros((12, 1, 44100)), 44100)
            >>> big_tree.waveform.shape
            (12, 1, 44100)
            >>> split_trees = big_tree.split(2)
            >>> len(split_trees)
            2
            >>> split_trees[0].waveform.shape
            (6, 1, 44100)  # Each tree has half the original batch size
        """
        total_batch_size = self.waveform.shape[0]
        assert total_batch_size % n_splits == 0, \
            f"Total batch size {total_batch_size} must be divisible by number of splits {n_splits}"

        split_batch_size = total_batch_size // n_splits

        return [
            tree_util.tree_map(lambda x: x[i * split_batch_size:(i + 1) * split_batch_size], self)
            for i in range(n_splits)
        ]

    def reshape_mini_batches(self, mini_batch_size: int) -> Self:
        """Reshape batch dimension into mini-batches by adding a new leading axis.

        Transforms audio data from shape (B, C, T) to (num_mini_batches, mini_batch_size, C, T),
        where B must be evenly divisible by mini_batch_size.

        Args:
            mini_batch_size: Number of samples per mini-batch. The total batch size must be
                evenly divisible by this value.

        Returns:
            AudioTree with an additional mini-batch dimension as the first axis.

        Example:
            >>> x = AudioTree(np.zeros((12, 1, 44100)), 44100)
            >>> x_batched = x.reshape_mini_batches(3)
            >>> x_batched.waveform.shape
            (4, 3, 1, 44100)  # 4 mini-batches of size 3
        """
        B, C, _ = self.waveform.shape

        # Calculate number of mini-batches (assuming B is evenly divisible)
        assert B % mini_batch_size == 0
        num_mini_batches = B // mini_batch_size

        # Reshape AudioTree to have leading mini-batch dimension
        # From (B, C, T) to (num_mini_batches, mini_batch_size, C, T)
        # Only reshape array-like objects since metadata can contain non-arrays
        reshaped_audio_tree = tree_util.tree_map(
            lambda x: x.reshape(num_mini_batches, mini_batch_size, *x.shape[1:]) if hasattr(x, "shape") else x,
            self,
        )
        return reshaped_audio_tree

    def flatten_mini_batches(self) -> Self:
        """Flatten mini-batches back into a single batch dimension.

        Undoes the operation performed by reshape_mini_batches(), transforming
        audio data from shape (num_mini_batches, mini_batch_size, C, T) back to
        (B, C, T).

        Returns:
            AudioTree with the mini-batch dimension flattened into the batch dimension.

        Example:
            >>> x = AudioTree(np.zeros((12, 1, 44100)), 44100)
            >>> x_batched = x.reshape_mini_batches(3)
            >>> x_batched.waveform.shape
            (4, 3, 1, 44100)  # 4 mini-batches of size 3
            >>> x_unbatched = x_batched.flatten_mini_batches()
            >>> x_unbatched.waveform.shape
            (12, 1, 44100)  # Back to original shape
        """
        # Assuming the waveform has shape (num_mini_batches, mini_batch_size, C, T)
        # We want to reshape to (num_mini_batches * mini_batch_size, C, T)

        # Get the current shape
        shape = self.waveform.shape

        # We expect at least 4 dimensions for mini-batched data
        assert len(shape) >= 4, (
            f"Expected at least 4 dimensions for mini-batched data, got {len(shape)}. "
            f"Shape: {shape}"
        )

        # Flatten the first two dimensions
        # From (num_mini_batches, mini_batch_size, C, T) to (B, C, T)
        # Only reshape array-like objects since metadata can contain non-arrays
        flattened_audio_tree = tree_util.tree_map(
            lambda x: x.reshape(-1, *x.shape[2:]) if hasattr(x, "shape") else x,
            self,
        )
        return flattened_audio_tree

    def filter(self, filter_fn: Callable) -> Self:
        B = self.waveform.shape[0]
        audio_trees = self.split(B)
        audio_trees = list(filter(filter_fn, audio_trees))

        numpy = np if isinstance(self.waveform, np.ndarray) else jnp

        if len(audio_trees) == 0:
            return tree_util.tree_map(
                lambda x: x[:0] if hasattr(x, 'shape') else x,
                self
            )

        audio_trees = tree_util.tree_map(
            lambda *xs: numpy.concatenate(xs, axis=0),
            *audio_trees
        )
        return audio_trees

    @staticmethod
    def batch_fn(items: Sequence[Any]) -> Any:
        """Batch function for use with grain's IterDataset.batch().

        Concatenates AudioTree objects along the batch axis (axis 0).
        Use this instead of grain's default batching, which would add
        an extra dimension since AudioTree already has shape (batch, channels, samples).

        Supports arbitrary nested structures containing AudioTrees. All arrays
        (including AudioTrees) are concatenated along axis 0, so data should have
        a leading batch dimension.

        Args:
            items: Sequence of AudioTree objects, or structures (dicts, lists, etc.)
                containing AudioTree objects.

        Returns:
            Batched structure with the same shape as the input items.

        Example:
            >>> from audiotree.sources import create_audio_dataset
            >>> ds = create_audio_dataset("/path/to/audio", duration=1.0)
            >>> iter_ds = ds.to_iter_dataset().batch(32, batch_fn=AudioTree.batch_fn)
            >>> for batch in iter_ds:
            ...     print(batch.waveform.shape)  # (32, channels, samples)
        """
        items = list(items)

        def batching_function(*args):
            first_arg = args[0]
            if isinstance(first_arg, AudioTree):
                return _batch_audiotrees(args)
            elif isinstance(first_arg, (np.ndarray, jnp.ndarray)):
                return np.concatenate(args, axis=0)
            else:
                return list(args)

        return tree_util.tree_map(
            batching_function,
            items[0],
            *items[1:],
            is_leaf=lambda x: isinstance(x, AudioTree),
        )


def _batch_audiotrees(audio_trees: Sequence[AudioTree]) -> AudioTree:
    """Batch a list of AudioTrees into a single AudioTree.

    Concatenates all array fields along the batch axis (axis 0) using NumPy.
    Requires all AudioTrees to have the same sample_rate and compatible shapes.

    Prefer using ``AudioTree.batch_fn`` instead, which handles mixed-type
    structures (dicts with AudioTrees, arrays, strings, etc.).

    Args:
        audio_trees: List of AudioTree objects to batch together.

    Returns:
        Single AudioTree with all items batched along axis 0.
    """
    if not audio_trees:
        raise ValueError("Cannot batch empty list of AudioTrees")

    return tree_util.tree_map(
        lambda *xs: np.concatenate(xs, axis=0),
        *audio_trees
    )
