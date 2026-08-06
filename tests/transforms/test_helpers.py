from itertools import product

from jax import numpy as jnp
from jax import random
import librosax
import numpy as np
import pytest

from audiotree import AudioTree
from audiotree.transforms.helpers import _swap_stereo_jax, _swap_stereo_np


@pytest.mark.parametrize(
    "hop_factor,n_frames",
    list(
        product(
            [0.25, 0.5],
            [128, 129],  # Use frame counts instead of arbitrary lengths
        )
    ),
)
def test_istft_invariance(hop_factor: float, n_frames: int):
    """Test that librosax STFT->ISTFT round-trip preserves the input.

    Note: Signal length is chosen to be a multiple of hop_length to avoid
    boundary effects. This matches how librosax is tested.
    """
    frame_length = 2048
    hop_length = int(frame_length * hop_factor)
    # Length chosen to work well with the hop_length
    length = hop_length * n_frames
    window = "hann"

    waveform = random.uniform(
        random.key(0), shape=(1, 1, length), minval=-0.5, maxval=0.5
    )

    stft_data = librosax.stft(
        waveform, n_fft=frame_length, hop_length=hop_length, window=window, center=True
    )

    recons = librosax.istft(
        stft_data,
        n_fft=frame_length,
        hop_length=hop_length,
        window=window,
        center=True,
        length=length,
    )

    assert jnp.allclose(recons, waveform, atol=1e-5, rtol=1e-5)


def _channel_ramp(channels: int) -> np.ndarray:
    """A (1, channels, 8) waveform whose channel ``c`` is filled with ``c``."""
    values = np.arange(channels, dtype=np.float32)
    return np.tile(values[None, :, None], (1, 1, 8))


@pytest.mark.parametrize("channels", [1, 2])
def test_swap_stereo_backends_agree(channels: int):
    """Both backends swap stereo and leave mono alone, identically.

    Mono is a no-op because the only permutation of one channel is the
    identity; it is not an error, and it must not be an error on one backend
    only.
    """
    waveform = _channel_ramp(channels)
    expected = waveform[:, ::-1]

    np_out = _swap_stereo_np(AudioTree(waveform=waveform, sample_rate=16000))
    jax_out = _swap_stereo_jax(
        AudioTree(waveform=jnp.asarray(waveform), sample_rate=16000)
    )

    np.testing.assert_array_equal(np.asarray(np_out.waveform), expected)
    np.testing.assert_array_equal(np.asarray(jax_out.waveform), expected)


def test_swap_stereo_rejects_more_than_two_channels_on_both_backends():
    """Four channels is undefined for a stereo swap, and raises on both backends.

    It used to reverse the channel order, so channel 0 of a 5.1 mix came back
    holding channel 5.
    """
    waveform = _channel_ramp(4)

    with pytest.raises(ValueError, match="4 channels"):
        _swap_stereo_np(AudioTree(waveform=waveform, sample_rate=16000))

    with pytest.raises(ValueError, match="4 channels"):
        _swap_stereo_jax(AudioTree(waveform=jnp.asarray(waveform), sample_rate=16000))
