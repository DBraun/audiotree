from functools import partial
from typing import Any, Optional, Tuple, Union

import chex
import jax
from einops import rearrange
from jax import numpy as jnp
from jax import random
from jax.tree_util import DictKey

from audiotree import AudioTree

KeyLeafPairs = list[tuple[list[DictKey], Any]]


def stft(x: jnp.ndarray, frame_length=2048, hop_factor=0.25, window="hann"):

    batch_size, num_channels, audio_length = x.shape

    frame_step = int(frame_length * hop_factor)

    x = rearrange(x, "b c t -> (b c) t")

    # This probably uses less memory than the aux method, but it's definitely slower than aux.
    _, _, stft_data = jax.scipy.signal.stft(
        x,
        window=window,
        nperseg=frame_length,
        noverlap=(frame_length - frame_step),
    )
    stft_data = rearrange(stft_data, "(b c) nf nt -> b c nf nt", b=batch_size)

    return stft_data


def istft(
    stft_matrix: chex.Array,
    noverlap: int,
    window: Optional[Union[str, float, Tuple[str, float]]] = "hann",
    length: Optional[int] = None,
) -> chex.Array:
    _, reconstructed_signal = jax.scipy.signal.istft(
        stft_matrix,
        noverlap=noverlap,
        window=window,
    )

    # Trim or pad the output signal to the desired length
    if length is not None:
        if length > reconstructed_signal.shape[-1]:
            # Pad the signal if it is shorter than the desired length
            pad_width = length - reconstructed_signal.shape[-1]
            reconstructed_signal = jnp.pad(
                reconstructed_signal, ((0, 0), (0, 0), (0, pad_width)), mode="constant"
            )
        else:
            # Trim the signal if it is longer than the desired length
            reconstructed_signal = reconstructed_signal[..., :length]

    return reconstructed_signal


# def _where_with_p(rng, modified: jnp.ndarray, original: jnp.ndarray, p: float):
#     B = original.shape[0]
#
#     shape = (B,) + (1,) * (original.ndim - 1)
#     use_modified = random.uniform(rng, shape=shape, minval=0) < p
#
#     return jnp.where(use_modified, modified, original)


def _db2linear(decibels):
    return jnp.pow(10.0, decibels / 20.0)


@jax.jit
def _volume_norm_transform(
    audio_tree: AudioTree, key: jax.Array, min_db: float, max_db: float
) -> AudioTree:

    audio_data = audio_tree.audio_data

    B = audio_data.shape[0]

    key, subkey = random.split(key)
    target_db = random.uniform(subkey, shape=(B,), minval=min_db, maxval=max_db)
    gain_db = target_db - audio_tree.loudness

    audio_data = audio_data * _db2linear(gain_db)[:, None, None]

    audio_tree = audio_tree.replace(audio_data=audio_data, loudness=target_db)

    return audio_tree


@jax.jit
def _volume_change_transform(
    audio_tree: AudioTree, key: jax.Array, min_db: float, max_db: float
) -> (tuple)[AudioTree, jnp.ndarray]:

    audio_data = audio_tree.audio_data

    B = audio_data.shape[0]

    key, subkey = random.split(key)
    gain_db = random.uniform(subkey, shape=(B,), minval=min_db, maxval=max_db)

    audio_data = audio_data * _db2linear(gain_db)[:, None, None]

    audio_tree = audio_tree.replace(audio_data=audio_data)

    return audio_tree, gain_db


@jax.jit
def _rescale_audio_transform(audio_tree: AudioTree) -> AudioTree:
    """Rescales audio to the range [-1, 1] only if the original audio exceeds those bounds. Useful if transforms have
    caused the audio to clip. It won't change the relative balance of multichannel audio.
    """
    audio_data = audio_tree.audio_data
    maxes = jnp.max(jnp.absolute(audio_data), axis=[-2, -1])
    maxes = jnp.expand_dims(maxes, [-2, -1])
    maxes = jnp.maximum(maxes, jnp.ones_like(maxes))
    audio_data = audio_data / maxes

    return audio_tree.replace(audio_data=audio_data)


@jax.jit
def _invert_phase_audio_transform(audio_tree: AudioTree) -> AudioTree:
    audio_data = audio_tree.audio_data
    audio_data = -audio_data
    return audio_tree.replace(audio_data=audio_data)


@jax.jit
def _swap_stereo_audio_transform(audio_tree: AudioTree) -> AudioTree:
    audio_data = audio_tree.audio_data
    audio_data = jnp.flip(audio_data, axis=1)
    return audio_tree.replace(audio_data=audio_data)


# todo: potential slow-down with re-jitting due to changed static args
@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def _corrupt_phase(
    audio_tree: AudioTree,
    rng: jax.Array,
    amount: float,
    hop_factor: float = 0.5,
    frame_length: float = 2048,
    window: str = "hann",
) -> AudioTree:
    audio_data = audio_tree.audio_data
    B, C, length = audio_data.shape

    frame_step = int(frame_length * hop_factor)
    noverlap = frame_length - frame_step

    stft_fun = partial(
        stft, frame_length=frame_length, hop_factor=hop_factor, window=window
    )
    istft_fun = partial(istft, noverlap=noverlap, window=window, length=length)

    stft_data = stft_fun(audio_data)

    amt = random.uniform(
        rng, shape=stft_data.shape[:-1], minval=-jnp.pi * amount, maxval=jnp.pi * amount
    )

    stft_data = stft_data * jnp.expand_dims(jnp.exp(1j * amt), axis=-1)
    audio_data = istft_fun(stft_data)

    return audio_tree.replace(audio_data=audio_data)


# todo: potential slow-down with re-jitting due to changed static args
@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def _shift_phase(
    audio_tree: AudioTree,
    key: jax.Array,
    amount: float,
    hop_factor: float = 0.5,
    frame_length: float = 2048,
    window: str = "hann",
) -> AudioTree:
    audio_data = audio_tree.audio_data
    B, C, length = audio_data.shape

    frame_step = int(frame_length * hop_factor)
    noverlap = frame_length - frame_step

    stft_fun = partial(
        stft, frame_length=frame_length, hop_factor=hop_factor, window=window
    )
    istft_fun = partial(istft, noverlap=noverlap, window=window, length=length)

    stft_data = stft_fun(audio_data)

    key, subkey = random.split(key)
    amt = random.uniform(
        subkey,
        shape=stft_data.shape[:-2],
        minval=-jnp.pi * amount,
        maxval=jnp.pi * amount,
    )

    stft_data = stft_data * jnp.expand_dims(jnp.exp(1j * amt), axis=(-2, -1))
    audio_data = istft_fun(stft_data)

    return audio_tree.replace(audio_data=audio_data)
