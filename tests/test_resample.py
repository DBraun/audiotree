import jax
import jax.numpy as jnp
import librosa
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


# --- AudioTree.resample(engine=...) -----------------------------------------
#
# ``engine`` picks the algorithm; ``None`` follows the waveform's array library.
# The two engines are different filters, so nothing below asserts that they
# agree numerically -- only that each engine is the one that ran, and that one
# engine gives the same answer however it is invoked.

# Same XLA convolution, eager versus traced, on unit-scaled float32 audio.
# XLA may fuse or reorder the traced graph, so allow float32 rounding noise
# (~1e-7 per op, accumulated over a ~1.3k-tap dot product) rather than
# demanding bitwise equality. Measured differences are 0 on CPU.
_EAGER_JIT_ATOL = 1e-6


def _tones_tree(xp=np, *, sample_rate=44_100, batch=2, channels=2, seconds=0.1):
    x = np.stack(
        [
            np.stack(
                [
                    _band_limited_tones(sample_rate, seconds, seed=10 * b + c)
                    for c in range(channels)
                ]
            )
            for b in range(batch)
        ]
    ).astype(np.float32)
    return AudioTree.create(xp.asarray(x), sample_rate)


def test_default_engine_numpy_is_soxr():
    """``engine=None`` on a NumPy waveform is exactly the librosa/soxr result."""
    tree = _tones_tree()
    got = tree.resample(48_000)
    want = librosa.util.fix_length(
        librosa.resample(tree.waveform, orig_sr=44_100, target_sr=48_000, axis=-1),
        size=tree.samples * 48_000 // 44_100,
        axis=-1,
    )
    np.testing.assert_array_equal(got.waveform, want)
    np.testing.assert_array_equal(
        got.waveform, tree.resample(48_000, engine="soxr").waveform
    )


def test_default_engine_jax_is_jax_kernel():
    """``engine=None`` on a JAX waveform is exactly the Julius-port kernel."""
    tree = _tones_tree(jnp)
    got = tree.resample(48_000)
    np.testing.assert_array_equal(
        np.asarray(got.waveform), np.asarray(resample(tree.waveform, 44_100, 48_000))
    )
    np.testing.assert_array_equal(
        np.asarray(got.waveform),
        np.asarray(tree.resample(48_000, engine="jax").waveform),
    )


def test_engine_jax_on_numpy_stays_numpy_and_matches_jax_tree():
    """The JAX algorithm runs on a NumPy tree and hands back NumPy."""
    np_tree = _tones_tree()
    out = np_tree.resample(48_000, engine="jax")
    assert isinstance(out.waveform, np.ndarray)
    assert out.waveform.dtype == np.float32
    assert out.sample_rate == 48_000
    assert out.waveform.shape == (2, 2, np_tree.samples * 48_000 // 44_100)
    np.testing.assert_array_equal(
        out.waveform,
        np.asarray(_tones_tree(jnp).resample(48_000).waveform),
    )


def test_engine_soxr_on_concrete_jax_stays_jax_and_matches_numpy_tree():
    """soxr runs on an eager JAX tree (via the host) and hands back JAX."""
    jax_tree = _tones_tree(jnp)
    out = jax_tree.resample(48_000, engine="soxr")
    assert isinstance(out.waveform, jax.Array)
    assert out.waveform.dtype == jnp.float32
    np.testing.assert_array_equal(
        np.asarray(out.waveform), _tones_tree().resample(48_000).waveform
    )


@pytest.mark.parametrize("engine", [None, "jax"], ids=["default", "jax"])
def test_engine_jax_eager_matches_jit(engine):
    """``engine="jax"`` gives the same algorithm eagerly and under ``jax.jit``.

    ``jax.jit`` turns the NumPy tree into traced JAX arrays, so inside it the
    default engine is the JAX one. Passing ``engine="jax"`` eagerly on the NumPy
    tree reproduces that preprocessing outside the jitted function.
    """
    np_tree = _tones_tree()
    eager = np_tree.resample(48_000, engine="jax")
    jitted = jax.jit(lambda t: t.resample(48_000, engine=engine))(np_tree)
    assert jitted.sample_rate == eager.sample_rate == 48_000
    assert jitted.waveform.shape == eager.waveform.shape
    np.testing.assert_allclose(
        np.asarray(jitted.waveform), eager.waveform, atol=_EAGER_JIT_ATOL
    )


def test_engine_jax_options_apply_on_numpy():
    """zeros/rolloff/full/output_length reach the JAX kernel from a NumPy tree."""
    np_tree = _tones_tree()
    jax_tree = _tones_tree(jnp)
    for kwargs in (
        dict(zeros=8),
        dict(rolloff=0.8),
        dict(full=True),
        dict(output_length=1_000),
    ):
        got = np_tree.resample(48_000, engine="jax", **kwargs)
        want = jax_tree.resample(48_000, **kwargs)
        np.testing.assert_array_equal(got.waveform, np.asarray(want.waveform))
    # The knobs are not silently dropped: a shorter filter changes the output.
    assert not np.array_equal(
        np_tree.resample(48_000, engine="jax", zeros=8).waveform,
        np_tree.resample(48_000, engine="jax").waveform,
    )
    # 4410 * 160 / 147 = 4800 exactly, so pick a length where full differs.
    odd = AudioTree.create(np.zeros((1, 1, 1_001), np.float32), 44_100)
    assert odd.resample(48_000, engine="jax").samples == 1_089
    assert odd.resample(48_000, engine="jax", full=True).samples == 1_090


@pytest.mark.parametrize("engine", ["soxr", "jax"])
def test_engine_preserves_mini_batch_and_channel_axes(engine):
    """Rank-4 (mini-batched) trees and channel counts survive either engine."""
    tree = _tones_tree(batch=4, channels=3).reshape_mini_batches(2)
    out = tree.resample(22_050, engine=engine)
    assert out.waveform.shape == (2, 2, 3, tree.samples // 2)
    assert out.sample_rate == 22_050


def test_engine_soxr_under_jit_raises():
    tree = _tones_tree()
    with pytest.raises(ValueError, match="engine='soxr'.*traced"):
        jax.jit(lambda t: t.resample(48_000, engine="soxr"))(tree)


def test_engine_soxr_under_vmap_raises():
    x = jnp.asarray(_tones_tree().waveform)
    with pytest.raises(ValueError, match="engine='soxr'.*traced"):
        jax.vmap(lambda w: AudioTree.create(w, 44_100).resample(48_000, engine="soxr"))(
            x
        )


def test_engine_soxr_raises_under_jit_even_at_the_same_rate():
    """The error does not depend on whether the resample happened to be a no-op."""
    tree = _tones_tree()
    with pytest.raises(ValueError, match="traced"):
        jax.jit(lambda t: t.resample(44_100, engine="soxr"))(tree)


def test_jax_transform_with_soxr_engine_raises_under_jit():
    from audiotree.transforms import jax as jax_transforms

    transform = jax_transforms.resample(sample_rate=48_000, engine="soxr")
    with pytest.raises(ValueError, match="traced"):
        jax.jit(transform.map)(_tones_tree(jnp))


@pytest.mark.parametrize(
    "kwargs",
    [dict(zeros=8), dict(rolloff=0.9), dict(full=True)],
    ids=["zeros", "rolloff", "full"],
)
@pytest.mark.parametrize("engine", [None, "soxr"], ids=["default", "soxr"])
@pytest.mark.parametrize("sample_rate", [48_000, 44_100], ids=["resample", "no-op"])
def test_jax_only_options_rejected_with_soxr(kwargs, engine, sample_rate):
    """JAX-only knobs are an error when soxr runs, rather than ignored."""
    (name,) = kwargs
    with pytest.raises(ValueError, match=f"{name} only apply to the JAX"):
        _tones_tree().resample(sample_rate, engine=engine, **kwargs)
    # Explicit soxr on a JAX tree is rejected the same way.
    if engine == "soxr":
        with pytest.raises(ValueError, match=f"{name} only apply to the JAX"):
            _tones_tree(jnp).resample(sample_rate, engine=engine, **kwargs)


def test_explicit_defaults_of_jax_options_still_rejected_with_soxr():
    """Passing the default value explicitly is still an explicit request."""
    with pytest.raises(ValueError, match="zeros, rolloff only apply"):
        _tones_tree().resample(48_000, zeros=24, rolloff=0.945)


def test_full_false_is_accepted_with_soxr():
    """``full=False`` is soxr's own (floored) behavior, so it is not a conflict."""
    assert _tones_tree().resample(48_000, engine="soxr", full=False).samples == 4_800


@pytest.mark.parametrize("engine", ["numpy", "librosa", "auto", ""])
def test_unknown_engine_rejected(engine):
    with pytest.raises(ValueError, match="engine must be None, 'soxr', or 'jax'"):
        _tones_tree().resample(48_000, engine=engine)
    with pytest.raises(ValueError, match="engine must be None"):
        _tones_tree().resample(44_100, engine=engine)


@pytest.mark.parametrize("output_length", [-1, 1_091])
def test_soxr_rejects_out_of_range_output_length(output_length):
    """Both engines share the documented ``[0, ceil(T * new / old)]`` range."""
    tree = AudioTree.create(np.zeros((1, 1, 1_001), np.float32), 44_100)
    with pytest.raises(ValueError, match="between 0 and 1090"):
        tree.resample(48_000, engine="soxr", output_length=output_length)
    with pytest.raises(ValueError, match="between 0 and 1090"):
        tree.resample(48_000, engine="jax", output_length=output_length)
    assert tree.resample(48_000, engine="soxr", output_length=1_090).samples == 1_090


@pytest.mark.parametrize("dtype", [np.float16, np.float64])
def test_soxr_preserves_non_float32_dtypes(dtype):
    """soxr computes in float32/float64 only; other dtypes round-trip."""
    tree = AudioTree.create(_tones_tree().waveform.astype(dtype), 44_100)
    assert tree.resample(48_000).waveform.dtype == dtype


def test_transform_passes_engine():
    from audiotree import transforms

    tree = _tones_tree()
    got = transforms.resample(sample_rate=48_000, engine="jax").map(tree)
    np.testing.assert_array_equal(
        got.waveform, tree.resample(48_000, engine="jax").waveform
    )
