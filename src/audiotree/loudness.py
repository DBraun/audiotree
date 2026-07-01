import math

import jax
import jaxloudnorm as jln
import numpy as np
from jax import numpy as jnp
from scipy.signal import lfilter

# ITU-R BS.1770 K-weighting: two RBJ-cookbook biquads (a high-frequency shelf
# then a high-pass), as ``(gain_db, Q, fc_hz, filter_type)``. These match the
# ``K-weighting`` filter class in :mod:`jaxloudnorm`, so the NumPy (exact IIR) and
# JAX (FIR-approximated) paths weight audio the same way.
_K_FILTER_STAGES = (
    (4.0, 1.0 / math.sqrt(2.0), 1500.0, "high_shelf"),
    (0.0, 0.5, 38.0, "high_pass"),
)
# BS.1770 per-channel weights for [L, R, C, Ls, Rs] (surround channels count more).
_CHANNEL_GAINS = (1.0, 1.0, 1.0, 1.41, 1.41)
# BS.1770 absolute loudness offset in the LUFS formula.
_ABSOLUTE_OFFSET = -0.691


def window_samples(window_duration_sec: float, sample_rate: int) -> int:
    """Number of samples spanned by a loudness window (or hop) of the given duration.

    Both the NumPy and JAX per-window loudness paths derive their window/hop spans
    from this, so they tile a waveform the same way.
    """
    return int(round(window_duration_sec * sample_rate))


def windowed_num_windows(samples: int, window_span: int, hop_span: int) -> int:
    """Number of whole windows that fit, stepping ``hop_span`` samples at a time.

    The trailing partial window is dropped; ``0`` when the signal is shorter than
    one window.
    """
    if samples < window_span:
        return 0
    return (samples - window_span) // hop_span + 1


def _rbj_biquad(
    gain_db: float, q: float, fc: float, sample_rate: int, filter_type: str
) -> tuple[np.ndarray, np.ndarray]:
    """RBJ-cookbook biquad coefficients ``(b, a)`` for a K-weighting stage (NumPy).

    Mirrors :meth:`jaxloudnorm.IIRfilter.generate_coefficients` for the two filter
    shapes the K-weighting uses, so the CPU IIR filter matches the JAX meter's.
    """
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * (fc / sample_rate)
    alpha = np.sin(w0) / (2.0 * q)
    cw = np.cos(w0)
    if filter_type == "high_shelf":
        b0 = A * ((A + 1) + (A - 1) * cw + 2 * np.sqrt(A) * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cw)
        b2 = A * ((A + 1) + (A - 1) * cw - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) - (A - 1) * cw + 2 * np.sqrt(A) * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cw)
        a2 = (A + 1) - (A - 1) * cw - 2 * np.sqrt(A) * alpha
    elif filter_type == "high_pass":
        b0 = (1 + cw) / 2
        b1 = -(1 + cw)
        b2 = (1 + cw) / 2
        a0 = 1 + alpha
        a1 = -2 * cw
        a2 = 1 - alpha
    else:
        raise RuntimeError(f"Unsupported K-weighting filter stage: {filter_type!r}")
    return np.array([b0, b1, b2]) / a0, np.array([a0, a1, a2]) / a0


def _windowed_lufs_from_kweighted(filtered, window_span, hop_span, num_windows, xp):
    """Ungated per-window LUFS from an already K-weighted ``(batch, channels, samples)`` signal.

    Each window's loudness is the K-weighted mean square of its samples expressed
    in LUFS (no gating), so windows are directly comparable. Windows are gathered
    (not accumulated), which keeps quiet windows exact next to loud ones. Fully
    silent windows map to ``-inf``. Returns ``(batch, num_windows)``.
    """
    sq = filtered * filtered
    starts = xp.arange(num_windows) * hop_span
    idx = (
        starts[:, None] + xp.arange(window_span)[None, :]
    )  # (num_windows, window_span)
    mean_square = sq[..., idx].mean(axis=-1)  # (batch, channels, num_windows)
    channels = filtered.shape[1]
    gains = xp.asarray(_CHANNEL_GAINS[:channels], dtype=mean_square.dtype)
    power = (gains[None, :, None] * mean_square).sum(axis=1)  # (batch, num_windows)
    return _ABSOLUTE_OFFSET + 10.0 * xp.log10(power)


@jax.jit(static_argnames=("sample_rate", "zeros"))
def jit_integrated_loudness(
    data: jnp.ndarray,
    sample_rate: int,
    zeros: int = 512,
):
    """Integrated loudness (LUFS) per item of a ``(batch, channels, samples)`` batch.

    Uses the ITU-R BS.1770 gating-block length (0.4s / 400ms). Items shorter than
    one block are right-padded with silence so the measurement stays valid.
    """
    block_size = 0.4  # BS.1770 gating block
    min_samples = math.ceil(block_size * sample_rate)

    original_length = data.shape[-1]
    if original_length < min_samples:
        data = jnp.pad(
            data,
            pad_width=(
                (0, 0),
                (0, 0),
                (0, min_samples - original_length),
            ),
        )

    meter = jln.Meter(sample_rate, block_size=block_size, use_fir=True, zeros=zeros)
    # jaxloudnorm >= 0.3.1 returns -inf LUFS for digital silence (the mathematical
    # limit of zero gated power), so no NaN guard is needed here.
    return jax.vmap(meter.integrated_loudness)(data)


@jax.jit(
    static_argnames=("sample_rate", "window_duration_sec", "hop_duration_sec", "zeros")
)
def jit_windowed_loudness(
    data: jnp.ndarray,
    sample_rate: int,
    window_duration_sec: float,
    hop_duration_sec: float,
    zeros: int = 512,
):
    """Ungated per-window loudness (LUFS) for a batch of waveforms on GPU.

    K-weights the whole signal once with :mod:`jaxloudnorm`'s filters (as a
    hardware loudness meter would run continuously), then reports the ungated
    K-weighted loudness of each ``window_duration_sec`` window, stepping
    ``hop_duration_sec`` between window starts. With ``hop == window`` the windows
    tile the audio without overlap; a smaller hop overlaps them. The trailing
    partial window is dropped and silent windows are ``-inf``. Every value is
    ungated, so windows are directly comparable (matching the upstream
    ``loudness.loudness_per_window``). The whole computation is vectorized, so it
    stays on the accelerator.

    The caller ensures at least one whole window fits; the empty case is handled
    upstream.

    Returns:
        A ``(batch, num_windows)`` array of per-window LUFS.
    """
    meter = jln.Meter(sample_rate, use_fir=True, zeros=zeros)

    def _k_weight(item):  # item: (channels, samples); jaxloudnorm filters are 2-D
        for stage in meter._filters:
            item = stage.apply_filter(item, axis=-1)
        return item

    filtered = jax.vmap(_k_weight)(data)  # (batch, channels, samples)

    window_span = window_samples(window_duration_sec, sample_rate)
    hop_span = window_samples(hop_duration_sec, sample_rate)
    num_windows = windowed_num_windows(data.shape[-1], window_span, hop_span)
    return _windowed_lufs_from_kweighted(
        filtered, window_span, hop_span, num_windows, jnp
    )


def numpy_windowed_lufs(
    waveform: np.ndarray,
    sample_rate: int,
    window_duration_sec: float,
    hop_duration_sec: float,
) -> np.ndarray:
    """Ungated per-window loudness (LUFS) for a ``(batch, channels, samples)`` NumPy batch.

    The CPU counterpart of :func:`jit_windowed_loudness`: K-weights the whole
    signal with exact IIR biquads (``scipy.signal.lfilter``), then reports the
    ungated K-weighted loudness of each window (stepping ``hop_duration_sec``).
    JAX-free so it is safe inside grain workers. Returns ``(batch, num_windows)``;
    fully silent windows are ``-inf``.
    """
    filtered = waveform.astype(np.float64)
    for gain_db, q, fc, filter_type in _K_FILTER_STAGES:
        b, a = _rbj_biquad(gain_db, q, fc, sample_rate, filter_type)
        filtered = lfilter(b, a, filtered, axis=-1)

    window_span = window_samples(window_duration_sec, sample_rate)
    hop_span = window_samples(hop_duration_sec, sample_rate)
    num_windows = windowed_num_windows(waveform.shape[-1], window_span, hop_span)
    with np.errstate(divide="ignore"):  # a fully silent window is -inf by definition
        lufs = _windowed_lufs_from_kweighted(
            filtered, window_span, hop_span, num_windows, np
        )
    return lufs.astype(np.float32)
