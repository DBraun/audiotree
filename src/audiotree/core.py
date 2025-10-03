from dataclasses import field
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union
from typing_extensions import Self

from flax import struct
from jax import numpy as jnp, tree_util
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


_str_max_length = 256


@struct.dataclass
class AudioTree:
    """
    A `flax.struct.dataclass`_ for holding audio information including a waveform, sample rate, and metadata.

    The ``AudioTree`` class is inspired by Descript AudioTools's `AudioSignal`_.
        .. _AudioSignal: https://github.com/descriptinc/audiotools/blob/master/audiotools/core/audio_signal.py
        .. _flax.struct.dataclass: https://flax.readthedocs.io/en/latest/api_reference/flax.struct.html#flax.struct.dataclass

    Args:
        audio_data (jnp.ndarray): Audio waveform data shaped ``(Samples)``, ``(Channels, Samples)``, or ``(Batch, Channels, Samples)``
        sample_rate (int): Sample rate of ``audio_data``, such as 44100 Hz.
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
    """

    audio_data: np.ndarray
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
        audio_data: np.ndarray,
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
        if audio_data.ndim == 1:
            audio_data = audio_data[None, None, :]  # Add batch and channel dimension
        elif audio_data.ndim == 2:
            audio_data = audio_data[None, :, :]  # Add batch dimension

        # Handle metadata and filepaths
        if metadata is None:
            metadata = {}
        else:
            metadata = metadata.copy()  # Don't modify the original dict
        
        if filepaths is not None:
            metadata["filepath"] = cls._encode_filepaths(filepaths)

        return cls(
            audio_data=audio_data,
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
        loudness = jit_integrated_loudness(
            jnp.array(self.audio_data), self.sample_rate, zeros=512
        )
        return self.replace(loudness=loudness)

    @staticmethod
    def _encode_string(s: str) -> np.ndarray:
        """Encode a single filepath *s* to an array of ASCII codes.

        The returned array is shaped ``(1, _str_max_length)`` so that multiple
        rows (filepaths) can be concatenated along *axis=0*.
        """
        s = str(s)
        encoded = [ord(char) for char in s[:_str_max_length]]
        encoded += [0] * (_str_max_length - len(encoded))
        return np.array([encoded], dtype=np.int16)  # [1, _str_max_length]

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

    @classmethod
    def from_file(
        cls,
        audio_path: Union[str, Path],
        sample_rate: int | None = None,
        offset: float = 0.0,
        duration: float | None = None,
        mono: bool = False,
        pad_mode: Literal["constant"] | None = "constant",
        filepaths: Union[str, Path, List[Union[str, Path]]] | None = None,
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
                ``pad_mode`` controls how the audio is right-padded. The default is "constant" (zeros). A choice of
                ``None`` results in no padding. Another useful choice is "wrap" to loop the audio.
            filepaths (Union[str, Path, List[str | Path]], optional): One or more filepaths to store in the returned
                ``AudioTree``'s metadata. If *None* (default) the provided ``audio_path`` will be used.
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

        data, sr = librosa.load(
            str(audio_path), sr=sample_rate, offset=offset, duration=duration, mono=mono
        )
        assert sr == sample_rate
        if data.ndim == 1:
            data = data[None, None, :]  # Add batch and channel dimension
        elif data.ndim == 2:
            data = data[None, :, :]  # Add batch dimension

        if (
            duration is not None
            and pad_mode is not None
            and data.shape[-1] < round(duration * sample_rate)
        ):
            pad_right = round(duration * sample_rate) - data.shape[-1]
            data = np.pad(
                data, pad_width=((0, 0), (0, 0), (0, pad_right)), mode=pad_mode
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
            audio_data=data,
            sample_rate=sr,
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
            >>> tree.audio_data.shape
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
            audio_data = []
            for idx in indices:
                filename = manifest_data['filename'][idx]
                audio_path = audio_dir / filename

                if not audio_path.exists():
                    raise FileNotFoundError(f"Audio file not found: {audio_path}")

                data, sr = librosa.load(str(audio_path), sr=sample_rate, mono=False)
                assert sr == sample_rate

                if data.ndim == 1:
                    data = data[None, :]  # Add channel dimension
                elif data.ndim == 2:
                    pass  # Already (channels, samples)

                audio_data.append(data)

            audio_data = np.stack(audio_data, axis=0)  # (batch, channels, samples)
        else:
            # No audio files - create zeros
            audio_data = np.zeros((len(indices), channels, samples), dtype=np.float32)

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

        return cls.create(audio_data, **tree_kwargs)

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

        # todo: revisit whether casting to numpy here actually prevents any slowdown with grain.
        excerpt = excerpt.replace(
            audio_data=np.array(excerpt.audio_data), loudness=np.array(excerpt.loudness)
        )
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

    def mini_batch_list(self, n_splits: int) -> List[Self]:
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
            >>> big_tree.audio_data.shape
            (12, 1, 44100)
            >>> split_trees = big_tree.mini_batch_list(2)
            >>> len(split_trees)
            2
            >>> split_trees[0].audio_data.shape
            (6, 1, 44100)  # Each tree has half the original batch size
        """
        total_batch_size = self.audio_data.shape[0]
        assert total_batch_size % n_splits == 0, \
            f"Total batch size {total_batch_size} must be divisible by number of splits {n_splits}"

        split_batch_size = total_batch_size // n_splits

        return [
            tree_util.tree_map(lambda x: x[i * split_batch_size:(i + 1) * split_batch_size], self)
            for i in range(n_splits)
        ]

    def mini_batch(self, mini_batch_size: int) -> Self:
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
            >>> x_batched = x.mini_batch(3)
            >>> x_batched.audio_data.shape
            (4, 3, 1, 44100)  # 4 mini-batches of size 3
        """
        B, C, _ = self.audio_data.shape

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

    def unbatch(self) -> Self:
        """Flatten mini-batches back into a single batch dimension.

        Undoes the operation performed by mini_batch(), transforming audio data
        from shape (num_mini_batches, mini_batch_size, C, T) back to (B, C, T).

        Returns:
            AudioTree with the mini-batch dimension flattened into the batch dimension.

        Example:
            >>> x = AudioTree(np.zeros((12, 1, 44100)), 44100)
            >>> x_batched = x.mini_batch(3)
            >>> x_batched.audio_data.shape
            (4, 3, 1, 44100)  # 4 mini-batches of size 3
            >>> x_unbatched = x_batched.unbatch()
            >>> x_unbatched.audio_data.shape
            (12, 1, 44100)  # Back to original shape
        """
        # Assuming the audio_data has shape (num_mini_batches, mini_batch_size, C, T)
        # We want to reshape to (num_mini_batches * mini_batch_size, C, T)

        # Get the current shape
        shape = self.audio_data.shape

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
        B = self.audio_data.shape[0]
        audio_trees = self.mini_batch_list(B)
        audio_trees = list(filter(filter_fn, audio_trees))

        if len(audio_trees) == 0:
            numpy = np if isinstance(self.audio_data, np.ndarray) else jnp
            return tree_util.tree_map(
                lambda x: x[:0] if hasattr(x, 'shape') else x,
                self
            )

        numpy = np if isinstance(self.audio_data, np.ndarray) else jnp
        audio_trees = tree_util.tree_map(
            lambda *xs: numpy.concatenate(xs, axis=0),
            *audio_trees
        )
        return audio_trees
