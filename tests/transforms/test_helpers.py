from itertools import product

from jax import numpy as jnp
from jax import random
import librosax
import numpy as np
import pytest

from audiotree import AudioTree
from audiotree.transforms.helpers import (
    _corrupt_phase_jax,
    _corrupt_phase_np,
    _shift_phase_jax,
    _shift_phase_np,
    _swap_stereo_jax,
    _swap_stereo_np,
)


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


def _dual_mono(length: int, seed: int = 0) -> np.ndarray:
    """A ``(1, 2, length)`` waveform whose two channels are identical."""
    channel = np.random.default_rng(seed).standard_normal((1, 1, length))
    return np.tile(channel * 0.1, (1, 2, 1)).astype(np.float32)


@pytest.mark.parametrize("seed", range(8))
def test_shift_phase_keeps_a_stereo_image_coherent(seed: int):
    """A phase shift is one rotation per item, not one per channel.

    ``_shift_phase_jax`` used to draw an angle per (batch, channel), which
    rotated the two halves of a stereo image by different amounts: on a pair of
    identical channels the two outputs came apart by up to 0.86 (peak
    amplitude 0.45) and were negatively correlated in half of the draws.
    ``_shift_phase_np`` always drew one angle per item, so the two backends
    disagreed on what the transform even means.
    """
    waveform = _dual_mono(44100, seed=seed)
    sample_rate = 44100

    jax_out = _shift_phase_jax(
        AudioTree(waveform=jnp.asarray(waveform), sample_rate=sample_rate),
        random.key(seed),
        amount=1.0,
    ).waveform
    np_out = _shift_phase_np(
        AudioTree(waveform=waveform, sample_rate=sample_rate),
        np.random.default_rng(seed),
        amount=1.0,
    ).waveform

    for name, out in (("jax", np.asarray(jax_out)), ("numpy", np.asarray(np_out))):
        left, right = out[0, 0], out[0, 1]
        assert np.abs(left - right).max() < 1e-5, f"{name} decorrelated the channels"


@pytest.mark.parametrize("hop_factor", [0.25, 0.5])
@pytest.mark.parametrize(
    "transform_jax,transform_np",
    [(_shift_phase_jax, _shift_phase_np), (_corrupt_phase_jax, _corrupt_phase_np)],
)
def test_phase_transforms_reconstruct_the_tail(
    hop_factor: float, transform_jax, transform_np
):
    """With ``amount=0`` the STFT round trip must return the whole signal.

    ``librosax.istft`` reconstructs only ``(n_frames - 1) * hop_length``
    samples and zero-fills the rest of the requested ``length``, so the JAX
    backend used to silence the last ``length % hop_length`` samples -- 68
    samples of this 1 s @ 44.1 kHz clip at either hop below, an audible drop
    out. The helpers now pad up to a whole number of hops before the STFT and
    trim afterwards.
    """
    length = 44100  # 44100 % 1024 == 44100 % 512 == 68
    hop_length = int(2048 * hop_factor)
    assert length % hop_length != 0, "this test is only meaningful for a ragged tail"

    waveform = _dual_mono(length, seed=1)
    sample_rate = 44100

    jax_out = np.asarray(
        transform_jax(
            AudioTree(waveform=jnp.asarray(waveform), sample_rate=sample_rate),
            random.key(0),
            amount=0.0,
            hop_factor=hop_factor,
        ).waveform
    )
    np_out = np.asarray(
        transform_np(
            AudioTree(waveform=waveform, sample_rate=sample_rate),
            np.random.default_rng(0),
            amount=0.0,
            hop_factor=hop_factor,
        ).waveform
    )

    tail = slice(-2 * hop_length, None)
    assert np.abs(jax_out[..., tail]).max() > 0.0
    np.testing.assert_allclose(jax_out, waveform, atol=1e-5)
    np.testing.assert_allclose(np_out, waveform, atol=1e-5)
