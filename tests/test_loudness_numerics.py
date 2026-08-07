"""Numerical tests for the ITU-R BS.1770 meter in :mod:`audiotree.loudness`.

``lufs`` is a persisted manifest column and the input to ``lufs_cutoff`` corpus
filtering, so a silent change to the meter silently re-selects everyone's
training corpus. These tests pin the parts of the meter that the rest of the
suite cannot see, because every other loudness test uses a 440 Hz tone -- a
frequency where the 1.5 kHz K-weighting shelf is inert and where 512 FIR taps
happen to be enough.
"""

import tracemalloc

import jaxloudnorm as jln
import loudness as loudness_cpp
import numpy as np
import pytest
from jax import numpy as jnp

from audiotree.loudness import (
    _FILTER_CLASS,
    _K_FILTER_STAGES,
    _jit_integrated_loudness,
    _jit_windowed_loudness,
    _kweighting_biquad,
    _numpy_windowed_lufs,
    _window_mean_square_jax,
    _window_mean_square_numpy,
    _window_samples,
    _windowed_num_windows,
    fir_taps,
    pad_to_gating_block,
)

# The two K-weighting biquads at 48 kHz, exactly as printed in ITU-R BS.1770-4
# (Tables 1 and 2): a high-frequency shelf followed by the RLB high-pass.
_BS1770_COEFFICIENTS_48K = (
    (
        [1.53512485958697, -2.69169618940638, 1.19839281085285],
        [1.0, -1.69065929318241, 0.73248077421585],
    ),
    (
        [1.0, -2.0, 1.0],
        [1.0, -1.99004745483398, 0.99007225036621],
    ),
)

# NumPy filters with exact IIR biquads while JAX convolves with a truncated FIR
# approximation of the same filters. With the tap count scaled to the sample rate
# (see ``audiotree.loudness.fir_taps``) the residual is ~0.005 LU on program
# material and ~0.02 LU on the pathological low-frequency signals below; 0.05 LU
# leaves room for that without admitting the ~0.7-8 LU errors that a fixed
# 512-tap FIR produces.
_BACKEND_ATOL = 0.05

# Red noise is the pathological case for the FIR approximation (see
# ``test_numpy_and_jax_agree_on_infrasonic_content``); the residual there is
# ~0.03 LU at 44.1/48 kHz and ~0.08 LU at 96 kHz, against the 5-9 LU that a fixed
# 512-tap FIR produces.
_PATHOLOGICAL_ATOL = 0.15

# ``lufs`` (gated, integrated) and ``lufs_windows`` (ungated, per window) are
# different statistics, but on a signal that is exactly periodic in the window
# length the gates exclude nothing and the blocks all carry equal power, so the
# two must agree once the windows are recombined in the power domain. Measured
# residual is ~2e-4 LU; 0.01 LU still catches the two paths running *different*
# K-weighting filters, which costs ~0.05 LU.
_WINDOW_CONSISTENCY_ATOL = 0.01

_ABSOLUTE_OFFSET = -0.691

# Window/hop pairs the streaming reductions have to tile the same way the old
# gather did. Every duration below leaves a trailing partial window, which must be
# dropped rather than zero-padded.
_WINDOW_HOP_CASES = [
    (48_000, 0.4, 0.4),  # windows tile the signal, no overlap
    (48_000, 0.4, 0.1),  # 4x overlap -- the EBU-style setting
    (48_000, 0.5, 0.3),  # non-integer window/hop ratio (5/3)
    (48_000, 1.0, 0.35),  # non-integer window/hop ratio (20/7)
    (44_100, 0.4, 0.15),  # spans that are not round sample counts
    (16_000, 0.75, 0.2),
]

# Per-window LUFS of ``_window_probe(48_000, 1.75)``, recorded from the gather
# implementation that ``_window_mean_square_{numpy,jax}`` replaced. The two
# backends differ by ~7e-4 LU here (exact IIR vs. FIR approximation), so each gets
# its own column; a change to the *reduction* would move a column by far more than
# the ``_REFERENCE_ATOL`` below.
_WINDOW_LUFS_REFERENCE = {
    ("numpy", 0.4, 0.4): (-6.224682, -5.484969, -6.910188, -11.125896),
    ("jax", 0.4, 0.4): (-6.224178, -5.484275, -6.909417, -11.125133),
    ("numpy", 0.5, 0.3): (-6.063997, -5.524408, -6.124948, -8.125831, -11.736936),
    ("jax", 0.5, 0.3): (-6.063460, -5.523734, -6.124200, -8.125055, -11.736178),
    ("numpy", 1.0, 0.35): (-5.920049, -6.479918, -8.468568),
    ("jax", 1.0, 0.35): (-5.919413, -6.479195, -8.467807),
}

# The streaming reductions are not bit-identical to the gather -- they sum the same
# samples in a different order. Measured drift is ~1e-13 LU (NumPy, float64) and
# ~1e-5 LU (JAX, float32); 1e-3 LU absorbs that and any platform-to-platform wobble
# in the FFT convolution behind the JAX K-weighting, while still being 100x tighter
# than the smallest step between reference windows.
_REFERENCE_ATOL = 1e-3

# Peak allocation as a multiple of the input waveform, measured on a 12 s stereo
# excerpt at the 3 s / 100 ms setting. The gather cost 72x (NumPy, whose filtering
# is float64) and 24x (JAX); the streaming reductions cost 4.0x and 3.5x, which is
# the filtered signal and its square. 8x fails the old code by a wide margin
# without being so tight that an extra temporary trips it.
_MAX_ALLOCATION_RATIO = 8.0


def _colored_noise(sample_rate: int, duration: float, exponent: float, seed: int):
    """Noise with a ``1 / f**exponent`` amplitude spectrum, peak-normalized to 0.5.

    ``exponent=0.5`` is pink (a decent stand-in for program material), ``1.0`` is
    red -- almost all of its energy sits in the band where the 38 Hz K-weighting
    high-pass does its work.
    """
    rng = np.random.default_rng(seed)
    n = int(sample_rate * duration)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    freqs[0] = freqs[1]  # avoid dividing by DC
    y = np.fft.irfft(spectrum / freqs**exponent, n)
    return (0.5 * y / np.abs(y).max()).astype(np.float32)


def _tone(sample_rate: int, duration: float, freq: float, amplitude: float = 1.0):
    t = np.arange(int(sample_rate * duration)) / sample_rate
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _stationary_harmonics(sample_rate: int, duration: float, seed: int = 0):
    """A broadband signal that is exactly periodic in 0.2 s, hence in every window.

    Every 400 ms analysis window and every 400 ms gating block then carries the
    same power, which takes block alignment out of the comparison between the
    integrated and the windowed loudness.
    """
    rng = np.random.default_rng(seed)
    n = int(sample_rate * duration)
    t = np.arange(n) / sample_rate
    y = np.zeros(n)
    for harmonic in range(6, 2000, 7):
        freq = 5.0 * harmonic  # 5 Hz fundamental -> 0.2 s period
        if freq > 0.45 * sample_rate:
            break
        y += np.sin(2 * np.pi * freq * t + rng.uniform(0.0, 2 * np.pi)) / np.sqrt(freq)
    return (0.4 * y / np.abs(y).max()).astype(np.float32)


def _window_probe(sample_rate: int, duration: float) -> np.ndarray:
    """A deterministic ``(1, 2, samples)`` stereo probe whose level drifts slowly.

    Two amplitude-modulated tones, one either side of the K-weighting shelf, at
    modulation rates that are not commensurate with any window length below. Every
    window therefore has a *different* loudness, so a reduction that misaligns the
    windows by even one hop cannot pass by accident. Purely analytic, so the
    reference values it pins are reproducible without an asset file.
    """
    n = int(sample_rate * duration)
    t = np.arange(n) / sample_rate
    left = np.sin(2 * np.pi * 220.0 * t) * (0.4 + 0.3 * np.sin(2 * np.pi * 0.37 * t))
    right = np.sin(2 * np.pi * 3_500.0 * t + 1.1) * (
        0.2 + 0.15 * np.cos(2 * np.pi * 0.23 * t)
    )
    return np.stack([left, right]).astype(np.float32)[None]


def _gathered_window_mean_square(sq, window_span, hop_span, num_windows, xp):
    """The reduction this module used to run: materialize every window, then mean.

    Kept as the reference that :func:`_window_mean_square_numpy` and
    :func:`_window_mean_square_jax` have to reproduce -- they exist only because
    this one allocates ``window_span / hop_span`` copies of the signal.
    """
    idx = (xp.arange(num_windows) * hop_span)[:, None] + xp.arange(window_span)[None, :]
    return sq[..., idx].mean(axis=-1)


def _numpy_integrated(waveform: np.ndarray, sample_rate: int) -> float:
    """Integrated LUFS of one ``(1, channels, samples)`` item via the CPU meter."""
    return float(
        loudness_cpp.integrated_loudness(
            np.ascontiguousarray(waveform[0].T), sample_rate
        )
    )


def _jax_integrated(waveform: np.ndarray, sample_rate: int) -> float:
    return float(_jit_integrated_loudness(jnp.asarray(waveform), sample_rate)[0])


def _combine_windows(windows) -> float:
    """Recombine ungated per-window LUFS into a single LUFS, in the power domain."""
    power = 10.0 ** ((np.asarray(windows, dtype=np.float64) - _ABSOLUTE_OFFSET) / 10.0)
    return _ABSOLUTE_OFFSET + 10.0 * np.log10(power.mean())


@pytest.mark.parametrize("sample_rate", [44_100, 48_000, 96_000])
def test_numpy_and_jax_integrated_agree_on_program_material(sample_rate: int):
    """The exact-IIR CPU meter and the FIR-approximated JAX meter agree."""
    waveform = np.stack(
        [_colored_noise(sample_rate, 2.0, 0.5, seed=s) for s in (1, 2)]
    )[None]
    assert waveform.shape[:2] == (1, 2)
    assert (
        abs(
            _jax_integrated(waveform, sample_rate)
            - _numpy_integrated(waveform, sample_rate)
        )
        < _BACKEND_ATOL
    )


@pytest.mark.parametrize("sample_rate", [44_100, 48_000, 96_000])
def test_numpy_and_jax_windows_agree_on_program_material(sample_rate: int):
    waveform = np.stack(
        [_colored_noise(sample_rate, 2.0, 0.5, seed=s) for s in (3, 4)]
    )[None]
    jax_windows = np.asarray(
        _jit_windowed_loudness(jnp.asarray(waveform), sample_rate, 0.4, 0.4)
    )
    numpy_windows = _numpy_windowed_lufs(waveform, sample_rate, 0.4, 0.4)
    assert jax_windows.shape == numpy_windows.shape
    np.testing.assert_allclose(jax_windows, numpy_windows, atol=_BACKEND_ATOL)


@pytest.mark.parametrize("sample_rate", [44_100, 48_000, 96_000])
def test_numpy_and_jax_agree_on_infrasonic_content(sample_rate: int):
    """Red noise: nearly all energy below the 38 Hz high-pass corner.

    This is where a FIR approximation that is too short to realize the high-pass
    diverges without bound -- 512 taps puts the two backends 5-9 LU apart here.
    """
    waveform = _colored_noise(sample_rate, 2.0, 1.0, seed=5)[None, None]
    assert (
        abs(
            _jax_integrated(waveform, sample_rate)
            - _numpy_integrated(waveform, sample_rate)
        )
        < _PATHOLOGICAL_ATOL
    )


@pytest.mark.parametrize("sample_rate", [44_100, 48_000, 96_000])
def test_numpy_and_jax_agree_on_a_45hz_tone(sample_rate: int):
    """A tone right in the 38 Hz high-pass transition band."""
    waveform = _tone(sample_rate, 2.0, 45.0, 0.5)[None, None]
    assert (
        abs(
            _jax_integrated(waveform, sample_rate)
            - _numpy_integrated(waveform, sample_rate)
        )
        < _BACKEND_ATOL
    )


def test_fir_taps_scale_with_sample_rate():
    """The FIR approximation must span a fixed duration, not a fixed tap count."""
    assert fir_taps(96_000) == 2 * fir_taps(48_000) == 4 * fir_taps(24_000)
    for sample_rate in (8_000, 16_000, 22_050, 44_100, 48_000, 96_000):
        # At least 50 ms: shorter than that and the 38 Hz high-pass is unrealizable.
        assert fir_taps(sample_rate) / sample_rate >= 0.05
    with pytest.raises(ValueError, match="sample_rate must be positive"):
        fir_taps(0)


@pytest.mark.parametrize("stage_index", [0, 1])
def test_numpy_kweighting_matches_the_bs1770_coefficients(stage_index: int):
    """``lufs_windows``' NumPy filter is the standard's, not an approximation of it.

    ``lufs`` and ``lufs_windows`` used to be computed with *different* K-weighting
    inside a single NumPy call: the windowed path ran pyloudnorm's legacy
    RBJ-cookbook approximation while the integrated path (the ``loudness`` C++
    library) implements the coefficients printed in BS.1770-4. That is a ~0.05 LU
    inconsistency between two fields of the same ``AudioTree``.
    """
    gain_db, q, fc, filter_type = _K_FILTER_STAGES[stage_index]
    b, a = _kweighting_biquad(gain_db, q, fc, 48_000, filter_type)
    expected_b, expected_a = _BS1770_COEFFICIENTS_48K[stage_index]
    np.testing.assert_allclose(b, expected_b, atol=1e-9)
    np.testing.assert_allclose(a, expected_a, atol=1e-9)


@pytest.mark.parametrize("stage_index", [0, 1])
def test_jax_meter_uses_the_same_kweighting(stage_index: int):
    """The jaxloudnorm meter behind both JAX paths runs the standard's filter too."""
    stage = jln.Meter(48_000, filter_class=_FILTER_CLASS)._filters[stage_index]
    expected_b, expected_a = _BS1770_COEFFICIENTS_48K[stage_index]
    np.testing.assert_allclose(np.asarray(stage.b), expected_b, atol=1e-6)
    np.testing.assert_allclose(np.asarray(stage.a), expected_a, atol=1e-6)


@pytest.mark.parametrize("sample_rate", [44_100, 48_000, 96_000])
def test_integrated_lufs_matches_recombined_windows(sample_rate: int):
    """``lufs`` and ``lufs_windows`` describe the same signal.

    On a signal periodic in the window length the BS.1770 gates exclude nothing
    and every block carries equal power, so the gated integrated loudness equals
    the power-mean of the ungated windows. They diverge only if the two paths
    weight or block the audio differently.
    """
    waveform = _stationary_harmonics(sample_rate, 2.0)[None, None]
    integrated = _jax_integrated(waveform, sample_rate)
    windows = np.asarray(
        _jit_windowed_loudness(jnp.asarray(waveform), sample_rate, 0.4, 0.4)
    )[0]
    assert abs(integrated - _combine_windows(windows)) < _WINDOW_CONSISTENCY_ATOL


def test_high_frequency_shelf_is_applied():
    """Pin the 1.5 kHz K-weighting shelf, which a 440 Hz tone cannot see.

    Equal-amplitude tones above and below the shelf must differ by the shelf's
    gain (~+3.3 LU as realized by the BS.1770 biquad); zeroing the shelf gain
    would make them equal.
    """
    sample_rate = 48_000
    low = _jax_integrated(_tone(sample_rate, 2.0, 1_000.0)[None, None], sample_rate)
    high = _jax_integrated(_tone(sample_rate, 2.0, 5_000.0)[None, None], sample_rate)
    assert 3.0 < high - low < 3.7


@pytest.mark.parametrize("duration", [0.05, 0.1, 0.2, 0.399, 0.4, 1.0])
def test_sub_gating_block_excerpt_measures_its_own_loudness(duration: float):
    """A short excerpt must not be diluted by the padding that makes it measurable.

    A full-scale 1 kHz sine is -3.01 LUFS at any length. Zero-padding it out to
    one 400 ms gating block without compensating used to report
    ``-3.01 + 10 * log10(duration / 0.4)`` instead -- 3.01 LU low at 200 ms.
    """
    sample_rate = 44_100
    waveform = _tone(sample_rate, duration, 1_000.0)[None, None]
    assert abs(_jax_integrated(waveform, sample_rate) - (-3.01)) < 0.05


def test_sub_gating_block_silence_is_still_neginf():
    """Level compensation must not turn a silent short excerpt into NaN."""
    sample_rate = 44_100
    silence = np.zeros((1, 1, sample_rate // 10), dtype=np.float32)
    assert _jax_integrated(silence, sample_rate) == -np.inf


def test_pad_to_gating_block_preserves_mean_square():
    """The padded block has the same mean square as the excerpt it came from."""
    sample_rate = 44_100
    waveform = _tone(sample_rate, 0.1, 1_000.0, amplitude=0.3)[None, None]
    padded = pad_to_gating_block(waveform, sample_rate, xp=np)
    assert padded.shape[-1] == int(np.ceil(0.4 * sample_rate))
    np.testing.assert_allclose((padded**2).mean(), (waveform**2).mean(), rtol=1e-5)


def test_pad_to_gating_block_passes_through_long_waveforms():
    sample_rate = 44_100
    waveform = _tone(sample_rate, 1.0, 1_000.0)[None, None]
    assert pad_to_gating_block(waveform, sample_rate, xp=np) is waveform


def test_pad_to_gating_block_rejects_empty():
    with pytest.raises(ValueError, match="zero-length"):
        pad_to_gating_block(np.zeros((1, 1, 0), dtype=np.float32), 44_100, xp=np)


def test_windowed_num_windows_rejects_a_zero_sample_hop():
    """A hop of 0 samples raises ValueError, not a bare ZeroDivisionError."""
    with pytest.raises(ValueError, match="hop_span"):
        _windowed_num_windows(1_000, 100, 0)


@pytest.mark.parametrize(("sample_rate", "window_sec", "hop_sec"), _WINDOW_HOP_CASES)
def test_streaming_window_reduction_matches_the_gather(
    sample_rate: int, window_sec: float, hop_sec: float
):
    """The sliding/``reduce_window`` reductions tile and reduce exactly as the gather did.

    Both backends, at overlapping and non-overlapping hops, at window/hop ratios
    that are not integers, and with a trailing partial window that must be dropped
    rather than padded.
    """
    waveform = _window_probe(sample_rate, 2.53)
    window_span = _window_samples(window_sec, sample_rate)
    hop_span = _window_samples(hop_sec, sample_rate)
    num_windows = _windowed_num_windows(waveform.shape[-1], window_span, hop_span)
    # Every case must exercise the dropped tail, else it proves nothing about it.
    assert 0 < (num_windows - 1) * hop_span + window_span < waveform.shape[-1]

    sq64 = waveform.astype(np.float64) ** 2
    expected = _gathered_window_mean_square(
        sq64, window_span, hop_span, num_windows, np
    )
    actual = _window_mean_square_numpy(sq64, window_span, hop_span)
    assert actual.shape == expected.shape == (1, 2, num_windows)
    np.testing.assert_allclose(actual, expected, rtol=1e-12)

    sq32 = jnp.asarray(waveform) ** 2
    jax_actual = np.asarray(_window_mean_square_jax(sq32, window_span, hop_span))
    assert jax_actual.shape == expected.shape
    np.testing.assert_allclose(jax_actual, expected, rtol=1e-5)


@pytest.mark.parametrize(("backend", "window_sec", "hop_sec"), _WINDOW_LUFS_REFERENCE)
def test_windowed_lufs_matches_saved_reference(
    backend: str, window_sec: float, hop_sec: float
):
    """Per-window LUFS is unchanged from the gather implementation, to 1e-3 LU.

    ``lufs_windows`` is a persisted manifest column, so the streaming reduction
    that replaced the gather has to be a pure memory optimization: same windows,
    same values.
    """
    sample_rate = 48_000
    waveform = _window_probe(sample_rate, 1.75)
    if backend == "numpy":
        windows = _numpy_windowed_lufs(waveform, sample_rate, window_sec, hop_sec)
    else:
        windows = np.asarray(
            _jit_windowed_loudness(
                jnp.asarray(waveform), sample_rate, window_sec, hop_sec
            )
        )
    expected = _WINDOW_LUFS_REFERENCE[backend, window_sec, hop_sec]
    assert windows.shape == (1, len(expected))
    np.testing.assert_allclose(windows[0], expected, atol=_REFERENCE_ATOL)


def test_numpy_windowed_lufs_does_not_materialize_the_windows():
    """The NumPy reduction runs over a strided view, so cost is O(signal), not O(gather).

    An 84.7 MB batch at the standard EBU setting used to peak at 5.8 GB RSS,
    because indexing with a ``(num_windows, window_span)`` index array copies the
    signal once per window of overlap -- 30x at 3 s / 100 ms, and 60x once the
    float64 filtering is counted.
    """
    sample_rate = 48_000
    waveform = _window_probe(sample_rate, 12.0)
    tracemalloc.start()
    try:
        _numpy_windowed_lufs(waveform, sample_rate, 3.0, 0.1)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < _MAX_ALLOCATION_RATIO * waveform.nbytes


def test_jax_windowed_lufs_does_not_materialize_the_windows():
    """Same for the JAX reduction, read off XLA's own buffer assignment."""
    sample_rate = 48_000
    waveform = _window_probe(sample_rate, 12.0)
    compiled = _jit_windowed_loudness.lower(
        jnp.asarray(waveform), sample_rate, 3.0, 0.1
    ).compile()
    analysis = compiled.memory_analysis()
    if analysis is None:  # not every XLA backend reports buffer assignment
        pytest.skip("backend does not expose a memory analysis")
    assert analysis.temp_size_in_bytes < _MAX_ALLOCATION_RATIO * waveform.nbytes
