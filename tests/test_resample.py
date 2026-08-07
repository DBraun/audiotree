import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.signal import butter, resample_poly, sosfiltfilt

from audiotree import AudioTree
from audiotree.resample import resample

# The Julius kernel (a cos^2-windowed sinc, `zeros=24`, cutoff at
# `0.945 * min(sr) / 2`) is not the same filter as scipy's Kaiser-windowed
# `resample_poly` default, so the two only agree to within their shared passband
# ripple + stopband leakage. Measured on the signals below (band-limited to
# ~0.35 of the lower Nyquist, edges trimmed) the worst deviation is ~1.1e-3 on a
# unit-scaled waveform; 3e-3 leaves headroom without being loose enough to hide a
# real defect -- e.g. transposing the polyphase de-interleave takes the error to
# ~2.0.
_SCIPY_ATOL = 3e-3
# Both filters ring at the boundaries, where they disagree by ~2e-2. Compare the
# steady-state interior only.
_EDGE_SEC = 0.005

_RATE_PAIRS = [
    (44_100, 48_000),  # up, awkward ratio (147/160)
    (48_000, 44_100),  # down, awkward ratio
    (44_100, 22_050),  # down, simple ratio
    (22_050, 44_100),  # up, simple ratio
    (16_000, 8_000),
    (8_000, 16_000),
]


def _band_limited_tones(sample_rate: int, duration: float, seed: int) -> np.ndarray:
    """A multi-tone signal well inside the passband of both resamplers."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(sample_rate * duration)) / sample_rate
    y = sum(
        np.sin(2 * np.pi * f * t + rng.uniform(0.0, 2 * np.pi))
        for f in (55.0, 220.0, 1000.0, 3000.0)
    )
    return (y / 4.0).astype(np.float64)


def _band_limited_noise(
    sample_rate: int, duration: float, cutoff_hz: float, seed: int
) -> np.ndarray:
    """Broadband noise low-passed below the resampler's transition band."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(sample_rate * duration))
    sos = butter(8, cutoff_hz / (sample_rate / 2), btype="low", output="sos")
    x = sosfiltfilt(sos, x)
    return (x / np.abs(x).max()).astype(np.float64)


def _assert_matches_resample_poly(x: np.ndarray, old_sr: int, new_sr: int):
    """Compare `resample` against `scipy.signal.resample_poly` on `[B, C, T]` input."""
    got = np.asarray(
        resample(jnp.asarray(x, dtype=jnp.float32), old_sr, new_sr, full=True)
    )
    ref = resample_poly(x, new_sr, old_sr, axis=-1)
    assert got.shape[:2] == x.shape[:2]

    n = min(got.shape[-1], ref.shape[-1])
    edge = int(_EDGE_SEC * new_sr)
    err = np.abs(got[..., edge : n - edge] - ref[..., edge : n - edge])
    assert err.max() < _SCIPY_ATOL, f"{old_sr} -> {new_sr}: max abs error {err.max()}"


@pytest.mark.parametrize("old_sr,new_sr", _RATE_PAIRS)
def test_resample_matches_scipy_mono(old_sr: int, new_sr: int):
    x = _band_limited_tones(old_sr, 0.5, seed=0)[None, None]
    _assert_matches_resample_poly(x, old_sr, new_sr)


@pytest.mark.parametrize("old_sr,new_sr", _RATE_PAIRS)
def test_resample_matches_scipy_stereo_batched(old_sr: int, new_sr: int):
    """Each (batch, channel) is resampled independently and correctly."""
    # Every (batch, channel) carries a different signal, so a mix-up between the
    # batch, channel or polyphase axes shows up as a gross error.
    x = np.stack(
        [
            np.stack(
                [_band_limited_tones(old_sr, 0.3, seed=2 * b + c) for c in range(2)]
            )
            for b in range(2)
        ]
    )
    assert x.shape[:2] == (2, 2)
    _assert_matches_resample_poly(x, old_sr, new_sr)


@pytest.mark.parametrize("old_sr,new_sr", [(44_100, 48_000), (48_000, 16_000)])
def test_resample_matches_scipy_broadband(old_sr: int, new_sr: int):
    """Broadband (not a pure tone) content, where the anti-aliasing filter matters."""
    x = _band_limited_noise(old_sr, 0.5, 0.35 * min(old_sr, new_sr), seed=7)[None, None]
    _assert_matches_resample_poly(x, old_sr, new_sr)


def test_resample_under_jit_matches_eager():
    """Tracing the resample does not change its numerics."""
    old_sr, new_sr = 44_100, 48_000
    x = jnp.asarray(_band_limited_tones(old_sr, 0.2, seed=3)[None, None], jnp.float32)
    jitted = jax.jit(resample, static_argnames=("old_sr", "new_sr"))
    np.testing.assert_allclose(
        np.asarray(jitted(x, old_sr=old_sr, new_sr=new_sr)),
        np.asarray(resample(x, old_sr=old_sr, new_sr=new_sr)),
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "dtype", [jnp.float32, jnp.float16, jnp.bfloat16], ids=["f32", "f16", "bf16"]
)
def test_resample_preserves_dtype(dtype):
    """The sinc kernel follows the input dtype, not JAX's global default."""
    x = jnp.zeros((1, 1, 4_410), dtype=dtype)
    y = resample(x, 44_100, 48_000)
    assert y.dtype == dtype


def test_resample_float32_under_x64():
    """With x64 enabled the kernel must still be float32 for a float32 input.

    Guards the regression where the kernel took `jnp`'s global default dtype, so
    `JAX_ENABLE_X64=1` made the convolution's operands disagree.
    """
    with jax.enable_x64(True):
        x = jnp.zeros((1, 1, 4_410), dtype=jnp.float32)
        assert resample(x, 44_100, 48_000).dtype == jnp.float32


def test_resample_rejects_non_3d():
    with pytest.raises(ValueError, match=r"\[B, C, T\].*\(1, 4410\)"):
        resample(jnp.zeros((1, 4_410)), 44_100, 48_000)


def test_resample_rejects_non_integer_rates():
    with pytest.raises(ValueError, match="should be integers"):
        resample(jnp.zeros((1, 1, 4_410)), 44_100.0, 48_000)


def test_resample_rejects_non_positive_rates():
    with pytest.raises(ValueError, match="should be positive"):
        resample(jnp.zeros((1, 1, 4_410)), 44_100, 0)


@pytest.mark.parametrize("new_sr", [44_100, 48_000], ids=["same_rate", "other_rate"])
def test_resample_rejects_oversized_output_length(new_sr: int):
    """`output_length` is validated at the equal-rate short circuit too."""
    x = jnp.zeros((1, 1, 4_410))
    with pytest.raises(ValueError, match="output_length must be between"):
        resample(x, 44_100, new_sr, output_length=99_999)


@pytest.mark.parametrize("new_sr", [44_100, 48_000], ids=["same_rate", "other_rate"])
def test_resample_rejects_negative_output_length(new_sr: int):
    x = jnp.zeros((1, 1, 4_410))
    with pytest.raises(ValueError, match="output_length must be between"):
        resample(x, 44_100, new_sr, output_length=-5)


@pytest.mark.parametrize("new_sr", [44_100, 48_000], ids=["same_rate", "other_rate"])
def test_resample_rejects_full_with_output_length(new_sr: int):
    x = jnp.zeros((1, 1, 4_410))
    with pytest.raises(ValueError, match="cannot pass both"):
        resample(x, 44_100, new_sr, output_length=100, full=True)


def test_resample_same_rate_honors_output_length():
    """An equal-rate resample must trim like every other rate, not pass through."""
    x = jnp.asarray(_band_limited_tones(44_100, 0.1, seed=1)[None, None], jnp.float32)
    out = resample(x, 44_100, 44_100, output_length=1_000)
    assert out.shape == (1, 1, 1_000)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(x[..., :1_000]))


def test_resample_same_rate_is_identity_by_default():
    x = jnp.asarray(_band_limited_tones(44_100, 0.1, seed=1)[None, None], jnp.float32)
    np.testing.assert_array_equal(
        np.asarray(resample(x, 44_100, 44_100)), np.asarray(x)
    )


def test_resample_output_lengths():
    """Default is the floored length; `full=True` is the ceiled one."""
    x = jnp.zeros((1, 1, 1_001))
    # 1001 * 160 / 147 = 1089.52...
    assert resample(x, 44_100, 48_000).shape[-1] == 1_089
    assert resample(x, 44_100, 48_000, full=True).shape[-1] == 1_090
    assert resample(x, 44_100, 48_000, output_length=1_090).shape[-1] == 1_090


def test_audiotree_resample_numpy_stays_numpy():
    """The NumPy backend resamples with librosa and does not become a JAX array."""
    tree = AudioTree.create(np.zeros((1, 1, 44_100), dtype=np.float32), 44_100)
    out = tree.resample(22_050)
    assert isinstance(out.waveform, np.ndarray)
    assert out.sample_rate == 22_050
    assert out.waveform.shape == (1, 1, 22_050)
    assert out.waveform.dtype == np.float32


def test_audiotree_resample_jax_stays_jax():
    """The JAX backend resamples with the Julius port and stays a JAX array."""
    tree = AudioTree.create(jnp.zeros((1, 1, 44_100)), 44_100)
    out = tree.resample(22_050)
    assert isinstance(out.waveform, jax.Array)
    assert out.sample_rate == 22_050
    assert out.waveform.shape == (1, 1, 22_050)


def test_audiotree_resample_numpy_output_length():
    """The NumPy backend honors output_length (trim/pad to a fixed size)."""
    tree = AudioTree.create(np.zeros((1, 1, 44_100), dtype=np.float32), 44_100)
    out = tree.resample(22_050, output_length=20_000)
    assert out.waveform.shape == (1, 1, 20_000)


def test_audiotree_resample_numpy_length_matches_jax():
    """NumPy and JAX resample agree on output length (soxr can differ by a sample)."""
    np_tree = AudioTree.create(np.zeros((1, 1, 44_100), dtype=np.float32), 44_100)
    jax_tree = AudioTree.create(jnp.zeros((1, 1, 44_100)), 44_100)
    # 44.1k -> 48k is the case where soxr would otherwise return 48001.
    assert (
        np_tree.resample(48_000).samples == jax_tree.resample(48_000).samples == 48_000
    )
