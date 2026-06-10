from itertools import product

from jax import numpy as jnp
from jax import random
import librosax
import pytest


@pytest.mark.parametrize(
    "hop_factor,n_frames",
    product(
        [0.25, 0.5],
        [128, 129],  # Use frame counts instead of arbitrary lengths
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

    waveform = random.uniform(random.key(0), shape=(1, 1, length), minval=-0.5, maxval=0.5)

    stft_data = librosax.stft(
        waveform, n_fft=frame_length, hop_length=hop_length, window=window, center=True
    )

    recons = librosax.istft(
        stft_data, n_fft=frame_length, hop_length=hop_length, window=window, center=True, length=length
    )

    assert jnp.allclose(recons, waveform, atol=1e-5, rtol=1e-5)
