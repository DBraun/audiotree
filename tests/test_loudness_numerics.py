"""Numerical tests for the ITU-R BS.1770 meter in :mod:`audiotree.loudness`.

``lufs`` is a persisted manifest column and the input to ``lufs_cutoff`` corpus
filtering, so a silent change to the meter silently re-selects everyone's
training corpus. These tests pin the parts of the meter that the rest of the
suite cannot see, because every other loudness test uses a 440 Hz tone -- a
frequency where the 1.5 kHz K-weighting shelf is inert and where 512 FIR taps
happen to be enough.
"""

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
