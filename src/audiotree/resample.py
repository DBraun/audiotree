# Adapted from julius (https://github.com/adefossez/julius).
# Copyright 2020 Alexandre Défossez. Licensed under the MIT license; the full
# notice is bundled at LICENSES/julius-MIT.txt.
"""
Differentiable JAX resampling adapted from PyTorch.
Implementation of Julius O. Smith algorithm for resampling.
See https://ccrma.stanford.edu/~jos/resample/ for details.
This implementation is specially optimized for when new_sr / old_sr is a fraction
with a small numerator and denominator when removing the gcd (e.g. new_sr = 700, old_sr = 500).

Very similar to [bmcfee/resampy](https://github.com/bmcfee/resampy) except this implementation
is optimized for the case mentioned before, while resampy is slower but more general.

"""

import math
import operator
from functools import lru_cache
from typing import Optional

import numpy as np
from einops import rearrange
from jax import lax
from jax import numpy as jnp


_MAX_KERNEL_BYTES = 64 * 1024**2


@lru_cache(maxsize=32)
def _sinc_kernels(
    old_sr: int, new_sr: int, zeros: int, rolloff: float
) -> tuple[np.ndarray, int]:
    """The polyphase sinc kernels for one rate ratio, and their half-width.

    Built once per ``(old_sr, new_sr, zeros, rolloff)`` — the rates are already
    reduced by their GCD — and cached, because the kernels depend on nothing else
    and rebuilding them dominates the cost of a short resample. Computed in NumPy
    float64: the windowed sinc is a fixed constant of the filter, so it is worth
    getting right regardless of the dtype the audio is carried in.

    Returns:
        ``(kernels, width)`` where ``kernels`` is ``(new_sr, 1, 2 * width + old_sr)``
        and ``width`` is the padding needed on each side of the input.
    """
    sr = min(new_sr, old_sr) * rolloff
    half_width = zeros * old_sr / sr if sr else math.inf
    if not math.isfinite(half_width) or (
        new_sr * (2 * math.ceil(half_width) + old_sr) * np.dtype(np.float64).itemsize
        > _MAX_KERNEL_BYTES
    ):
        raise ValueError(
            "The JAX resampling filter would exceed 64 MiB for the reduced "
            f"ratio {new_sr}/{old_sr}. Use engine='soxr' outside JAX tracing "
            "or choose a simpler rate ratio."
        )
    width = math.ceil(half_width)
    idx = np.arange(-width, width + old_sr, dtype=np.float64)

    t = (-np.arange(new_sr, dtype=np.float64)[:, None] / new_sr + idx / old_sr) * sr
    t = np.clip(t, -zeros, zeros) * np.pi
    window = np.cos(t / zeros / 2) ** 2
    kernels = np.sinc(t / np.pi) * window
    kernels /= kernels.sum(axis=-1, keepdims=True)

    kernels = kernels.reshape((new_sr, 1, -1))
    kernels.setflags(write=False)  # cached and shared; never mutate in place
    return kernels, width


def _resample_output_length(old_sr, new_sr, length, output_length, full) -> int:
    try:
        old_sr, new_sr = operator.index(old_sr), operator.index(new_sr)
    except TypeError as exc:
        raise ValueError("old_sr and new_sr should be integers") from exc
    if old_sr <= 0 or new_sr <= 0:
        raise ValueError(
            f"old_sr and new_sr should be positive, got {old_sr} and {new_sr}"
        )
    max_output_length = -(-new_sr * length // old_sr)
    if output_length is None:
        return max_output_length if full else new_sr * length // old_sr
    if full:
        raise ValueError("You cannot pass both full=True and output_length")
    try:
        output_length = operator.index(output_length)
    except TypeError as exc:
        raise ValueError("output_length must be an integer") from exc
    if not 0 <= output_length <= max_output_length:
        raise ValueError(f"output_length must be between 0 and {max_output_length}")
    return output_length


def _validate_resample_filter(zeros, rolloff) -> None:
    if not isinstance(zeros, (int, np.integer)) or zeros <= 0:
        raise ValueError(f"zeros must be a positive integer, got {zeros!r}")
    if not 0 < rolloff <= 1:
        raise ValueError(f"rolloff must be in (0, 1], got {rolloff!r}")


def _require_float_waveform(waveform) -> None:
    if not jnp.issubdtype(waveform.dtype, jnp.floating):
        raise ValueError("Resampling requires a real floating-point waveform.")


def resample(
    x: jnp.ndarray,
    old_sr: int,
    new_sr: int,
    zeros: int = 24,
    rolloff: float = 0.945,
    output_length: Optional[int] = None,
    full: bool = False,
) -> jnp.ndarray:
    """Resampling algorithm adapted from the pytorch library Julius:
    https://github.com/adefossez/julius/blob/main/julius/resample.py

    Args:
        x (jnp.ndarray): Floating-point input shaped `[B, C, T]`. The output keeps its dtype.
        old_sr (int): sample rate of the input signal x.
        new_sr (int): sample rate of the output.
        zeros (int): number of zero crossing to keep in the sinc filter.
        rolloff (float): use a lowpass filter that is `rolloff * min(old_sr, new_sr) / 2`,
            to ensure sufficient margin due to the imperfection of the FIR filter used.
            Lowering this value will reduce anti-aliasing, but will reduce some of the
            highest frequencies.
        output_length (Optional[int]): Desired length of the output's last axis.
            Must be between 0 and `ceil(new_sr * T / old_sr)`. When None (default),
            the floored length `floor(new_sr * T / old_sr)` is used (or the ceiled
            length when `full=True`). Cannot be combined with `full=True`.
        full (bool): If True (and `output_length` is None), return the longest
            possible output (`ceil(new_sr * T / old_sr)`) rather than the floored
            default. Useful when chaining resamplings: pass `full=True` to every
            intermediate step and give `output_length` only for the last one.

    Raises:
        ValueError: If the rates are not positive integers, `x` is not 3-D, or
            `output_length`/`full` are out of range or contradictory, or the
            requested filter is invalid or would exceed 64 MiB.

    Shape:

        - Input: `[B, C, T]`
        - Output: `[B, C, T']` with `T' = int(new_sr * T / old_sr)`

    .. caution::
        After dividing `old_sr` and `new_sr` by their GCD, both should be small
        for this implementation to be fast.
    """

    if x.ndim != 3:
        raise ValueError(
            f"resample expects a [B, C, T] array, got shape {tuple(x.shape)}"
        )

    output_length = _resample_output_length(
        old_sr, new_sr, x.shape[-1], output_length, full
    )
    _validate_resample_filter(zeros, rolloff)
    if new_sr == old_sr:
        return x[..., :output_length]
    _require_float_waveform(x)
    if x.size == 0 or output_length == 0:
        return jnp.zeros_like(x, shape=(*x.shape[:-1], output_length))
    return _resample_jax(x, old_sr, new_sr, zeros, rolloff, output_length)


def _resample_jax(x, old_sr, new_sr, zeros, rolloff, output_length):
    batch_size, c, _ = x.shape
    old_sr, new_sr = operator.index(old_sr), operator.index(new_sr)
    zeros = int(zeros)

    gcd = math.gcd(old_sr, new_sr)
    old_sr = old_sr // gcd
    new_sr = new_sr // gcd

    kernels, width = _sinc_kernels(old_sr, new_sr, zeros, rolloff)
    # The kernel follows the input's dtype, not JAX's global default -- otherwise
    # bfloat16/float16 inputs, and float32 inputs under `jax_enable_x64`, hand
    # `conv_general_dilated` two operands of different dtypes and it errors out.
    kernel = jnp.asarray(kernels, dtype=x.dtype)

    x = rearrange(x, "b (c one) t -> (b c) one t", b=batch_size, c=c, one=1)

    y = lax.conv_general_dilated(
        x,
        kernel,
        window_strides=(old_sr,),
        padding=((width, width + old_sr),),
        precision=lax.Precision.HIGHEST,
    )
    y = rearrange(y, "(b c) w t -> b c (t w)", b=batch_size, c=c)

    return y[..., :output_length]
