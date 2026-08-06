"""ITU-R BS.1770 loudness, with matching NumPy and JAX implementations.

The K-weighting coefficients, per-channel gains and gating constants follow
pyloudnorm (https://github.com/csteinmetz1/pyloudnorm, Copyright (c) 2018
Christian Steinmetz) and its JAX port jaxloudnorm, both MIT licensed; the full
notice is bundled at LICENSES/pyloudnorm-MIT.txt.
"""

import math
from typing import Optional

import jax
import jaxloudnorm as jln
import numpy as np
from jax import numpy as jnp
from scipy.signal import lfilter

# ITU-R BS.1770 K-weighting, as the pair of biquads Brecht DeMan fitted to the
# coefficients printed in the standard (pyloudnorm/jaxloudnorm call this the
# ``DeMan`` filter class), given as ``(gain_db, Q, fc_hz, filter_type)``.
#
# pyloudnorm *defaults* to a ``K-weighting`` class built from the RBJ cookbook
# instead; that is an approximation of these and measures ~0.04 LU differently on
# program material. The ``loudness`` C++ library behind the NumPy ``lufs`` path
# implements the standard's coefficients, so DeMan is what keeps all four paths
# -- NumPy/JAX x integrated/windowed -- weighting audio identically.
_FILTER_CLASS = "DeMan"
_K_FILTER_STAGES = (
    (3.99984385397, 0.7071752369554193, 1681.9744509555319, "high_shelf_DeMan"),
    (0.0, 0.5003270373253953, 38.13547087613982, "high_pass_DeMan"),
)
# BS.1770 per-channel weights for [L, R, C, Ls, Rs] (surround channels count more).
_CHANNEL_GAINS = (1.0, 1.0, 1.0, 1.41, 1.41)
# BS.1770 absolute loudness offset in the LUFS formula.
_ABSOLUTE_OFFSET = -0.691
# BS.1770 gating block length, in seconds. Loudness is undefined below one block.
_GATING_BLOCK_SEC = 0.4
# Duration the FIR approximation of the K-weighting IIR filters must span. The
# 38 Hz high-pass rings for tens of milliseconds, so the tap count has to track
# the sample rate: jaxloudnorm's default of 512 taps is 10.7 ms at 48 kHz and
# cannot realize the filter at all there (0.6-0.7 LU error on 38-60 Hz tones, and
# unbounded error on infrasonic content). Measured at 48 kHz the approximation
# has converged to within 0.02 LU by ~43 ms, so 50 ms leaves margin at any rate.
_FIR_IMPULSE_SEC = 0.05


def safe_gain_db(lufs, target_lufs, max_gain_db=None, *, xp=np):
    """Per-item dB gain that moves ``lufs`` to ``target_lufs``, guarding silence.

    An item whose measured loudness is not finite — digital silence, or anything
    below the BS.1770 absolute gate, both of which read ``-inf`` — has no defined
    gain to a target: ``target - (-inf)`` is ``+inf``, and ``0 * inf`` is ``NaN``.
    Such items get a gain of ``0.0`` (left untouched) rather than being turned
    into an all-``NaN`` waveform.

    The resulting loudness is always ``lufs + gain_db``: ``target_lufs`` for a
    normally-measured item, the (unchanged) non-finite value for a silent one,
    and ``lufs + max_gain_db`` when the gain is capped.

    Args:
        lufs: Measured integrated loudness, shaped ``(*batch,)``.
        target_lufs: Target loudness in LUFS — a scalar or a ``(*batch,)`` array.
        max_gain_db: Optional ceiling on the applied gain, so a very quiet (but
            still finite) item is not amplified without bound. ``None`` (the
            default) applies whatever gain the target implies.
        xp: The array module to compute with — :mod:`numpy` or ``jax.numpy``.

    Returns:
        The ``(*batch,)`` gain in dB, finite for every item.
    """
    gain_db = xp.where(xp.isfinite(lufs), target_lufs - lufs, 0.0)
    if max_gain_db is not None:
        gain_db = xp.minimum(gain_db, max_gain_db)
    return gain_db


def shift_lufs(lufs, gain_db, *, xp=np):
    """Apply a per-item dB gain to a measured loudness.

    Adding the gain rather than assigning the target is what keeps a skipped
    (non-finite) or capped item honest: ``-inf + 0.0`` stays ``-inf``.

    The result is normalized to ``float32``, the dtype :meth:`AudioTree.replace_lufs`
    produces. ``lufs`` is a derived measurement rather than user data, and a
    float64 leaf would be silently narrowed by JAX under the default x64-off
    config — which would make the NumPy and JAX backends disagree on a field
    that is supposed to round-trip through both.
    """
    return (lufs + gain_db).astype(xp.float32)


def shift_lufs_windows(lufs_windows, gain_db, *, xp=np):
    """Shift per-window LUFS by a per-item dB gain, or pass through ``None``.

    A constant gain shifts every window equally, so this keeps ``lufs_windows``
    aligned with ``lufs`` instead of leaving it stale. ``gain_db`` is ``(batch,)``
    and broadcasts over the window axis; it must be finite, which
    :func:`safe_gain_db` guarantees, so a silent item's ``-inf`` windows stay
    ``-inf`` rather than becoming ``NaN``.
    """
    if lufs_windows is None:
        return None
    return (lufs_windows + gain_db[..., None]).astype(xp.float32)


def _window_samples(lufs_window_sec: float, sample_rate: int) -> int:
    """Number of samples spanned by a loudness window (or hop) of the given duration.

    Both the NumPy and JAX per-window loudness paths derive their window/hop spans
    from this, so they tile a waveform the same way.
    """
    return int(round(lufs_window_sec * sample_rate))


def _windowed_num_windows(samples: int, window_span: int, hop_span: int) -> int:
    """Number of whole windows that fit, stepping ``hop_span`` samples at a time.

    The trailing partial window is dropped; ``0`` when the signal is shorter than
    one window.
    """
    if samples < window_span:
        return 0
    return (samples - window_span) // hop_span + 1


def fir_taps(sample_rate: int) -> int:
    """Tap count for the FIR approximation of the K-weighting IIR filters.

    The K-weighting filters are specified in Hz, so their impulse responses last
    a fixed *duration* — the tap count must therefore scale with the sample rate.
    Rounded up to a power of two, which is what the FFT convolution behind
    :func:`jaxloudnorm.lfilter.approximate_iir_as_fir` wants anyway.
    """
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}.")
    return 1 << math.ceil(math.log2(_FIR_IMPULSE_SEC * sample_rate))


def pad_to_gating_block(waveform, sample_rate: int, *, xp=np):
    """Right-pad an excerpt shorter than one BS.1770 gating block, level-compensated.

    BS.1770 has no answer below one 400 ms gating block, and every implementation
    here needs *some* answer because short excerpts are routine. Padding alone is
    not it: zero-padding a ``dur`` second excerpt out to 400 ms dilutes its mean
    square by ``dur / 0.4``, so the measured loudness under-reports the excerpt's
    own loudness by exactly ``10 * log10(dur / 0.4)`` — 3.01 dB for a 200 ms
    excerpt. Scaling the padded signal by ``sqrt(0.4 * sample_rate / samples)``
    restores the excerpt's own mean square, so the meter reports the loudness of
    the audio that is actually there. Scaling before the meter (rather than
    correcting its output afterwards) also means BS.1770's absolute and relative
    gates see the corrected level.

    Longer waveforms are returned unchanged.

    Args:
        waveform: A ``(..., samples)`` array.
        sample_rate: Sample rate of ``waveform`` in Hz.
        xp: The array module to compute with — :mod:`numpy` or ``jax.numpy``.

    Returns:
        The waveform, right-padded and level-compensated to at least one gating
        block.
    """
    samples = waveform.shape[-1]
    min_samples = math.ceil(_GATING_BLOCK_SEC * sample_rate)
    if samples >= min_samples:
        return waveform
    if samples == 0:
        raise ValueError("Cannot measure the loudness of a zero-length waveform.")
    pad_width = ((0, 0),) * (waveform.ndim - 1) + ((0, min_samples - samples),)
    padded = xp.pad(waveform, pad_width)
    return padded * xp.asarray(math.sqrt(min_samples / samples), dtype=padded.dtype)


def _kweighting_biquad(
    gain_db: float, q: float, fc: float, sample_rate: int, filter_type: str
) -> tuple[np.ndarray, np.ndarray]:
    """Biquad coefficients ``(b, a)`` for a K-weighting stage (NumPy).

    Mirrors :meth:`jaxloudnorm.IIRfilter.generate_coefficients` for the two
    ``DeMan`` filter shapes the K-weighting uses, so the CPU IIR filter matches
    the JAX meter's.
    """
    K = np.tan(np.pi * fc / sample_rate)
    if filter_type == "high_shelf_DeMan":
        Vh = 10.0 ** (gain_db / 20.0)
        Vb = Vh**0.499666774155
        a0 = 1.0 + K / q + K * K
        b = np.array(
            [
                Vh + Vb * K / q + K * K,
                2.0 * (K * K - Vh),
                Vh - Vb * K / q + K * K,
            ]
        )
    elif filter_type == "high_pass_DeMan":
        a0 = 1.0 + K / q + K * K
        b = np.array([1.0, -2.0, 1.0]) * a0
    else:
        raise RuntimeError(f"Unsupported K-weighting filter stage: {filter_type!r}")
    a = np.array([a0, 2.0 * (K * K - 1.0), 1.0 - K / q + K * K])
    return b / a0, a / a0


def _window_mean_square_numpy(
    sq: np.ndarray, window_span: int, hop_span: int
) -> np.ndarray:
    """Per-window mean of ``(..., samples)`` squared samples, without a gather.

    :func:`numpy.lib.stride_tricks.sliding_window_view` is a zero-copy strided
    *view*, and reducing over its window axis never realizes it — where indexing
    ``sq`` with a ``(num_windows, window_span)`` index array allocates one full
    copy of the signal per window of overlap (30x the signal at the standard EBU
    3 s / 100 ms setting).

    The window axis of the view is the original sample axis, so each window is
    still summed over contiguous memory in ascending order — the same pairwise
    summation, hence the same value to the last bit or two, as reducing a copy.
    """
    windows = np.lib.stride_tricks.sliding_window_view(sq, window_span, axis=-1)
    return windows[..., ::hop_span, :].mean(axis=-1)


def _window_mean_square_jax(
    sq: jnp.ndarray, window_span: int, hop_span: int
) -> jnp.ndarray:
    """Per-window mean of ``(..., samples)`` squared samples, without a gather.

    :func:`jax.lax.reduce_window` is XLA's strided sliding reduction: it walks the
    windows and accumulates, so its footprint is the ``(..., num_windows)`` output
    rather than the ``(..., num_windows, window_span)`` gather it replaces.

    A prefix-sum differenced at the window edges would also be O(n), but its error
    grows with the length of the whole signal (a float32 ``cumsum`` over a minutes-
    long waveform drifts at the tail, and JAX runs float32 by default), whereas
    each ``reduce_window`` accumulator only ever spans one window. Measured
    against a float64 reference the two are 1.5e-6 and 1.2e-6 relative on a 20 s
    signal; only ``reduce_window`` keeps that bound as the signal grows.
    """
    lead = (1,) * (sq.ndim - 1)
    total = jax.lax.reduce_window(
        sq,
        jnp.zeros((), sq.dtype),
        jax.lax.add,
        window_dimensions=lead + (window_span,),
        window_strides=lead + (hop_span,),
        padding="VALID",
    )
    return total / window_span


def _windowed_lufs_from_kweighted(filtered, window_span, hop_span, num_windows, xp):
    """Ungated per-window LUFS from an already K-weighted ``(batch, channels, samples)`` signal.

    Each window's loudness is the K-weighted mean square of its samples expressed
    in LUFS (no gating), so windows are directly comparable. Each window is summed
    on its own — no running accumulator carries error across the signal — which
    keeps quiet windows exact next to loud ones. Fully silent windows map to
    ``-inf``. Returns ``(batch, num_windows)``.
    """
    if num_windows == 0:
        return xp.zeros((filtered.shape[0], 0), dtype=filtered.dtype)
    sq = filtered * filtered
    window_mean_square = (
        _window_mean_square_numpy if xp is np else _window_mean_square_jax
    )
    mean_square = window_mean_square(sq, window_span, hop_span)  # (b, c, num_windows)
    channels = filtered.shape[1]
    gains = xp.asarray(_CHANNEL_GAINS[:channels], dtype=mean_square.dtype)
    power = (gains[None, :, None] * mean_square).sum(axis=1)  # (batch, num_windows)
    return _ABSOLUTE_OFFSET + 10.0 * xp.log10(power)


@jax.jit(static_argnames=("sample_rate", "zeros"))
def _jit_integrated_loudness(
    data: jnp.ndarray,
    sample_rate: int,
    zeros: Optional[int] = None,
):
    """Integrated loudness (LUFS) per item of a ``(batch, channels, samples)`` batch.

    Uses the ITU-R BS.1770 gating-block length (0.4s / 400ms). Items shorter than
    one block are padded and level-compensated by :func:`pad_to_gating_block`, so
    a short excerpt measures its own loudness rather than a diluted one.

    Args:
        data: A ``(batch, channels, samples)`` waveform.
        sample_rate: Sample rate of ``data`` in Hz.
        zeros: Tap count for the FIR approximation of the K-weighting filters.
            ``None`` (the default) uses :func:`fir_taps`, which scales with the
            sample rate; pass an explicit count only to study the approximation.
    """
    data = pad_to_gating_block(data, sample_rate, xp=jnp)

    meter = jln.Meter(
        sample_rate,
        filter_class=_FILTER_CLASS,
        block_size=_GATING_BLOCK_SEC,
        use_fir=True,
        zeros=fir_taps(sample_rate) if zeros is None else zeros,
    )
    # jaxloudnorm >= 0.3.1 returns -inf LUFS for digital silence (the mathematical
    # limit of zero gated power), so no NaN guard is needed here.
    return jax.vmap(meter.integrated_loudness)(data)


@jax.jit(static_argnames=("sample_rate", "lufs_window_sec", "lufs_hop_sec", "zeros"))
def _jit_windowed_loudness(
    data: jnp.ndarray,
    sample_rate: int,
    lufs_window_sec: float,
    lufs_hop_sec: float,
    zeros: Optional[int] = None,
):
    """Ungated per-window loudness (LUFS) for a batch of waveforms on GPU.

    K-weights the whole signal once with :mod:`jaxloudnorm`'s filters (as a
    hardware loudness meter would run continuously), then reports the ungated
    K-weighted loudness of each ``lufs_window_sec`` window, stepping
    ``lufs_hop_sec`` between window starts. With ``hop == window`` the windows
    tile the audio without overlap; a smaller hop overlaps them. The trailing
    partial window is dropped and silent windows are ``-inf``. Every value is
    ungated, so windows are directly comparable (matching the upstream
    ``loudness.loudness_per_window``). The whole computation is vectorized, so it
    stays on the accelerator.

    The caller ensures at least one whole window fits; the empty case is handled
    upstream.

    Args:
        data: A ``(batch, channels, samples)`` waveform.
        sample_rate: Sample rate of ``data`` in Hz.
        lufs_window_sec: Window length in seconds.
        lufs_hop_sec: Step between window starts, in seconds.
        zeros: Tap count for the FIR approximation of the K-weighting filters.
            ``None`` (the default) uses :func:`fir_taps`, which scales with the
            sample rate; pass an explicit count only to study the approximation.

    Returns:
        A ``(batch, num_windows)`` array of per-window LUFS.
    """
    meter = jln.Meter(
        sample_rate,
        filter_class=_FILTER_CLASS,
        use_fir=True,
        zeros=fir_taps(sample_rate) if zeros is None else zeros,
    )

    def _k_weight(item):  # item: (channels, samples); jaxloudnorm filters are 2-D
        for stage in meter._filters:
            item = stage.apply_filter(item, axis=-1)
        return item

    filtered = jax.vmap(_k_weight)(data)  # (batch, channels, samples)

    window_span = _window_samples(lufs_window_sec, sample_rate)
    hop_span = _window_samples(lufs_hop_sec, sample_rate)
    num_windows = _windowed_num_windows(data.shape[-1], window_span, hop_span)
    return _windowed_lufs_from_kweighted(
        filtered, window_span, hop_span, num_windows, jnp
    )


def _numpy_windowed_lufs(
    waveform: np.ndarray,
    sample_rate: int,
    lufs_window_sec: float,
    lufs_hop_sec: float,
) -> np.ndarray:
    """Ungated per-window loudness (LUFS) for a ``(batch, channels, samples)`` NumPy batch.

    The CPU counterpart of :func:`_jit_windowed_loudness`: K-weights the whole
    signal with exact IIR biquads (``scipy.signal.lfilter``), then reports the
    ungated K-weighted loudness of each window (stepping ``lufs_hop_sec``).
    JAX-free so it is safe inside grain workers. Returns ``(batch, num_windows)``;
    fully silent windows are ``-inf``.
    """
    filtered = waveform.astype(np.float64)
    for gain_db, q, fc, filter_type in _K_FILTER_STAGES:
        b, a = _kweighting_biquad(gain_db, q, fc, sample_rate, filter_type)
        filtered = lfilter(b, a, filtered, axis=-1)

    window_span = _window_samples(lufs_window_sec, sample_rate)
    hop_span = _window_samples(lufs_hop_sec, sample_rate)
    num_windows = _windowed_num_windows(waveform.shape[-1], window_span, hop_span)
    with np.errstate(divide="ignore"):  # a fully silent window is -inf by definition
        lufs = _windowed_lufs_from_kweighted(
            filtered, window_span, hop_span, num_windows, np
        )
    return lufs.astype(np.float32)
