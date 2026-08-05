"""ITU-R BS.1770 loudness, with matching NumPy and JAX implementations.

The K-weighting coefficients, per-channel gains and gating constants follow
pyloudnorm (https://github.com/csteinmetz1/pyloudnorm, Copyright (c) 2018
Christian Steinmetz) and its JAX port jaxloudnorm, both MIT licensed; the full
notice is bundled at LICENSES/pyloudnorm-MIT.txt.
"""

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
def _jit_integrated_loudness(
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


@jax.jit(static_argnames=("sample_rate", "lufs_window_sec", "lufs_hop_sec", "zeros"))
def _jit_windowed_loudness(
    data: jnp.ndarray,
    sample_rate: int,
    lufs_window_sec: float,
    lufs_hop_sec: float,
    zeros: int = 512,
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

    Returns:
        A ``(batch, num_windows)`` array of per-window LUFS.
    """
    meter = jln.Meter(sample_rate, use_fir=True, zeros=zeros)

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
        b, a = _rbj_biquad(gain_db, q, fc, sample_rate, filter_type)
        filtered = lfilter(b, a, filtered, axis=-1)

    window_span = _window_samples(lufs_window_sec, sample_rate)
    hop_span = _window_samples(lufs_hop_sec, sample_rate)
    num_windows = _windowed_num_windows(waveform.shape[-1], window_span, hop_span)
    with np.errstate(divide="ignore"):  # a fully silent window is -inf by definition
        lufs = _windowed_lufs_from_kweighted(
            filtered, window_span, hop_span, num_windows, np
        )
    return lufs.astype(np.float32)
