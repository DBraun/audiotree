from typing import Any, Callable, Dict, List, Union

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


class NeuralLatentEncodeTransform(BaseMapTransform):
    """Use a neural network to set the latents of the AudioTree.

    .. code-block:: python

        @staticmethod
        def get_default_config() -> Dict[str, Any]:
            return {}
    """

    def __init__(
        self,
        encoder_fn: Callable[[AudioTree], jnp.ndarray],
        config: Dict[str, Dict[str, Any]] = None,
        scope: Dict[str, Dict[str, Any]] = None,
        output_key: Union[str, Callable[[List[str]], str]] = None,
    ):
        """
        Initialize the base transform with a configuration, a flag for seed splitting, a probability, a scope, and an
        output key.

        Args:
            encoder_fn (Callable[[AudioTree], jnp.ndarray]): A function that takes an audio_tree and returns a latent
                sequence.
            config (Dict[str, Dict[str, Any]]): Configuration dictionary for the transform
            scope (Dict[str, Dict[str, Any]]): Dictionary indicating which modalities to apply the transform to
            output_key (Union[str, Callable[[List[str]], str]], optional): Key under which to store the transformed
                value. By default, the values will be transformed in-place.
        """
        self.encoder_fn = encoder_fn
        super().__init__(config, scope, output_key)

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        return {}

    def _apply_transform(self, audio_tree: AudioTree) -> AudioTree:
        if audio_tree.latents is None:
            latents = self.encoder_fn(audio_tree)
            audio_tree = audio_tree.replace(latents=latents)
        return audio_tree


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


class NeuralAudioCodecEncodeTransform(BaseMapTransform):
    """
    Use a neural audio codec such as `Descript Audio Codec (DAC) or EnCodec <https://github.com/DBraun/DAC-JAX>`_ to
    encode audio into tokens.

    .. code-block:: python

        @staticmethod
        def get_default_config() -> Dict[str, Any]:
            return {}
    """

    def __init__(
        self,
        encoder_fn: Callable[[AudioTree], jnp.ndarray],
        num_codebooks: int,
        config: Dict[str, Dict[str, Any]] = None,
        scope: Dict[str, Dict[str, Any]] = None,
        output_key: Union[str, Callable[[List[str]], str]] = None,
    ):
        """
        Initialize the base transform with a configuration, a flag for seed splitting, a probability, a scope, and an
        output key.

        Args:
            encoder_fn (Callable[[AudioTree], jnp.ndarray]): A function that takes audio shaped ``(B, C, T)`` and
                returns tokens shaped ``((B C), K, S)``, where ``T`` is length in samples, ``S`` is encoded sequence
                length, and ``K`` is number of codebooks.
            num_codebooks (int): The number of codebooks in the codec.
            config (Dict[str, Dict[str, Any]]): Configuration dictionary for the transform
            scope (Dict[str, Dict[str, Any]]): Dictionary indicating which modalities to apply the transform to
            output_key (Union[str, Callable[[List[str]], str]], optional): Key under which to store the transformed
                value. By default, the values will be transformed in-place.
        """
        self.encoder_fn = encoder_fn
        self.num_codebooks = num_codebooks
        super().__init__(config, scope, output_key)

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        return {}

    def _apply_transform(self, audio_tree: AudioTree) -> AudioTree:
        if audio_tree.codes is None:
            B, C, T = audio_tree.audio_data.shape
            codes = self.encoder_fn(audio_tree)
            codes = rearrange(
                codes,
                "(B C) K S -> B (K C) S",
                B=B,
                C=C,
            )
            audio_tree = audio_tree.replace(codes=codes)
        return audio_tree


class ReduceBatchTransform(grain.MapTransform):

    def __init__(self):
        pass

    def map(self, audio_tree: AudioTree) -> AudioTree:

        def f(leaf):
            if isinstance(leaf, (np.ndarray, jnp.ndarray)):
                if leaf.ndim > 1:
                    shape = leaf.shape
                    shape = (shape[0] * shape[1],) + shape[2:]
                    return leaf.reshape(shape)
            return leaf

        audio_tree = jax.tree.map(f, audio_tree)

        return audio_tree
