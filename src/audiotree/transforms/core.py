from typing import Any, Callable, Dict

from einops import rearrange
from grain import python as grain
import jax
from jax import numpy as jnp
import numpy as np

from audiotree import AudioTree
from audiotree.transforms.base import BaseRandomTransform, BaseMapTransform
from audiotree.transforms.helpers import (
    _volume_norm_transform,
    _volume_change_transform,
    _rescale_audio_transform,
    _invert_phase_audio_transform,
    _swap_stereo_audio_transform,
    _corrupt_phase,
    _shift_phase,
)


class Identity(BaseMapTransform):
    """
    A transform that returns each item without any modifications.

    .. code-block:: python

        @staticmethod
        def get_default_config() -> Dict[str, Any]:
            return {}
    """

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        return {}

    @staticmethod
    def _apply_transform(audio_tree: AudioTree) -> AudioTree:
        return audio_tree


class VolumeChange(BaseRandomTransform):
    """Change the volume by a uniformly randomly selected decibel value.

    .. code-block:: python

        @staticmethod
        def get_default_config() -> Dict[str, Any]:
            return {
                "min_db": 0,
                "max_db": 0,
            }
    """

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        return {
            "min_db": 0,
            "max_db": 0,
        }

    @staticmethod
    def _apply_transform(
        audio_tree: AudioTree, rng: jax.Array, min_db: float, max_db: float
    ) -> AudioTree:
        audio_tree, gain_db = _volume_change_transform(audio_tree, rng, min_db, max_db)
        if audio_tree.loudness is not None:
            audio_tree = audio_tree.replace(loudness=(audio_tree.loudness + gain_db))
        return audio_tree


class VolumeNorm(BaseRandomTransform):
    """Normalize the volume to a randomly selected loudness value specified in LUFS.

    .. code-block:: python

        @staticmethod
        def get_default_config() -> Dict[str, Any]:
            return {
                "min_db": 0,
                "max_db": 0,
            }
    """

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        return {
            "min_db": 0,
            "max_db": 0,
        }

    @staticmethod
    def _pre_transform(audio_tree: AudioTree) -> AudioTree:
        audio_tree = audio_tree.replace_loudness()
        return audio_tree

    @staticmethod
    def _apply_transform(
        audio_tree: AudioTree, rng: jax.Array, min_db: float, max_db: float
    ) -> AudioTree:
        return _volume_norm_transform(audio_tree, rng, min_db, max_db)


class RescaleAudio(BaseMapTransform):
    """
    Rescale the audio so that the largest absolute value is 1.0. If none of the values are outside the range
    ``[-1., 1.]``, then no transformation is applied.

    .. code-block:: python

        @staticmethod
        def get_default_config() -> Dict[str, Any]:
            return {}
    """

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        return {}

    @staticmethod
    def _apply_transform(audio_tree: AudioTree) -> AudioTree:
        return _rescale_audio_transform(audio_tree).replace(loudness=None)


class InvertPhase(BaseMapTransform):
    """
    Invert the phase of all channels of audio.

    .. code-block:: python

        @staticmethod
        def get_default_config() -> Dict[str, Any]:
            return {}
    """

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        return {}

    @staticmethod
    def _apply_transform(audio_tree: AudioTree) -> AudioTree:
        return _invert_phase_audio_transform(audio_tree)


class SwapStereo(BaseMapTransform):
    """Swap the channels of stereo audio.

    .. code-block:: python

        @staticmethod
        def get_default_config() -> Dict[str, Any]:
            return {}
    """

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        return {}

    @staticmethod
    def _apply_transform(audio_tree: AudioTree) -> AudioTree:
        return _swap_stereo_audio_transform(audio_tree)


class CorruptPhase(BaseRandomTransform):
    """
    Perform a phase corruption on the audio. The phase shift range is in the range ``[-pi * amount, pi * amount]``, and
    it's independently selected for each frequency in the STFT.

    .. code-block:: python

        @staticmethod
        def get_default_config() -> Dict[str, Any]:
            return {
                "amount": 1,
                "hop_factor": 0.5,
                "frame_length": 2048,
                "window": "hann",
            }
    """

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        return {
            "amount": 1,
            "hop_factor": 0.5,
            "frame_length": 2048,
            "window": "hann",
        }

    @staticmethod
    def _apply_transform(
        audio_tree: AudioTree,
        rng: jax.Array,
        amount: float,
        hop_factor: float,
        frame_length: int,
        window: str,
    ) -> AudioTree:
        return _corrupt_phase(audio_tree, rng, amount, hop_factor, frame_length, window)


class ShiftPhase(BaseRandomTransform):
    """
    Perform a phase shift on the audio. The phase shift range is in the range ``[-pi * amount, pi * amount]``.

    .. code-block:: python

        @staticmethod
        def get_default_config() -> Dict[str, Any]:
            return {
                "amount": 1,
                "hop_factor": 0.5,
                "frame_length": 2048,
                "window": "hann",
            }
    """

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        return {
            "amount": 1,
        }

    @staticmethod
    def _apply_transform(
        audio_tree: AudioTree, rng: jax.Array, amount: float
    ) -> AudioTree:
        return _shift_phase(audio_tree, rng, amount)


class Choose(grain.RandomMapTransform):
    """
    With probability ``prob``, choose ``c`` transform(s) among ``transforms`` with optional probability weights
    ``weights``.
    """

    def __init__(self, *transforms, c: int = 1, weights=None, prob: float = 1):

        if weights is not None:
            assert len(weights) == len(transforms)

        assert c <= len(transforms)

        self.c = c
        self.weights = weights
        assert 0 <= prob <= 1
        self.prob = prob

        self.transforms = transforms

    def random_map(self, element, rng: np.random.Generator):

        # Reference:
        # https://github.com/google/grain/blob/cbad82fddd4c5bd94b87d93d3f29849e8e59a501/grain/_src/python/data_loader.py#L481

        if rng.random() >= self.prob:
            return element

        transforms = rng.choice(
            self.transforms, size=(self.c,), replace=False, p=self.weights
        )

        for transform in transforms:

            if isinstance(transform, grain.MapTransform):
                element = transform.map(element)
            elif isinstance(transform, grain.RandomMapTransform):
                element = transform.random_map(element, rng)
            elif hasattr(transform, "np_random_map"):  # TfRandomMapTransform
                element = transform.np_random_map(element, rng)
            else:
                # If a `seed` is provided we treat the Callable as RandomMapTransform
                element = transform(element, rng)

        return element


class NeuralAudioCodecEncodeTransform(grain.MapTransform):
    """
    Use a neural audio codec such as `Descript Audio Codec (DAC) <https://github.com/DBraun/DAC-JAX>`_ to encode audio
    into tokens.

    Args:
        encode_audio_fn (Callable): A jitted function that takes audio shaped ``(B, C, T)`` and returns tokens
            shaped ``((B C), S, K)``, where ``T`` is length in samples, ``S`` is encoded sequence length, and ``K`` is
            number of codebooks.
        n_codebooks (int): The number of codebooks in the codec.
    """

    def __init__(
        self,
        encode_audio_fn: Callable[[jnp.ndarray], jnp.ndarray],
        n_codebooks: int,
    ):
        self.encode_audio_fn = encode_audio_fn
        self.n_codebooks = n_codebooks

    def map(self, audio_signal: AudioTree):

        def append_codes(leaf):
            audio_data = leaf.audio_data
            B, C, T = audio_data.shape

            codes = self.encode_audio_fn(audio_data)

            codes = rearrange(
                codes, "(B C) S K -> B (K C) S", B=B, C=C, K=self.n_codebooks
            )

            leaf = leaf.replace(codes=codes)
            return leaf

        def is_leaf(x):
            return isinstance(x, AudioTree)

        audio_signal = jax.tree_util.tree_map(
            append_codes, audio_signal, is_leaf=is_leaf
        )

        return audio_signal


class ReduceBatchTransform(grain.MapTransform):

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate

    def map(self, audio_signal: AudioTree) -> AudioTree:

        def f(leaf):
            if isinstance(leaf, (np.ndarray, jnp.ndarray)):
                if leaf.ndim > 1:
                    shape = leaf.shape
                    shape = (shape[0] * shape[1],) + shape[2:]
                    return leaf.reshape(shape)
            return leaf

        audio_signal = jax.tree_util.tree_map(f, audio_signal)

        return audio_signal
