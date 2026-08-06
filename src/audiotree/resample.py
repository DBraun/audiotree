# Adapted from julius (https://github.com/adefossez/julius).
# Copyright 2020 Alexandre Défossez. Licensed under the MIT license; the full
# notice is bundled at LICENSES/julius-MIT.txt.
"""
Differentiable, Pytorch based resampling.
Implementation of Julius O. Smith algorithm for resampling.
See https://ccrma.stanford.edu/~jos/resample/ for details.
This implementation is specially optimized for when new_sr / old_sr is a fraction
with a small numerator and denominator when removing the gcd (e.g. new_sr = 700, old_sr = 500).

Very similar to [bmcfee/resampy](https://github.com/bmcfee/resampy) except this implementation
is optimized for the case mentioned before, while resampy is slower but more general.

"""

import math
from functools import lru_cache
from typing import Optional

import numpy as np
from einops import rearrange
from jax import lax
from jax import numpy as jnp


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
    width = math.ceil(zeros * old_sr / sr)
    idx = np.arange(-width, width + old_sr, dtype=np.float64)

    t = (-np.arange(new_sr, dtype=np.float64)[:, None] / new_sr + idx / old_sr) * sr
    t = np.clip(t, -zeros, zeros) * np.pi
    window = np.cos(t / zeros / 2) ** 2
    kernels = np.sinc(t / np.pi) * window
    kernels /= kernels.sum(axis=-1, keepdims=True)

    kernels = kernels.reshape((new_sr, 1, -1))
    kernels.setflags(write=False)  # cached and shared; never mutate in place
    return kernels, width


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
        x (jnp.ndarray): Input array shaped `[B, C, T]`. The output keeps its dtype.
        old_sr (int): sample rate of the input signal x.
        new_sr (int): sample rate of the output.
        zeros (int): number of zero crossing to keep in the sinc filter.
        rolloff (float): use a lowpass filter that is `rolloff * new_sr / 2`,
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
            `output_length`/`full` are out of range or contradictory. These are
            checked identically at every rate, including `old_sr == new_sr`.

    Shape:

        - Input: `[B, C, T]`
        - Output: `[B, C, T']` with `T' = int(new_sr * T / old_sr)`

    .. caution::
        After dividing `old_sr` and `new_sr` by their GCD, both should be small
        for this implementation to be fast.
    """

    if not isinstance(old_sr, int) or not isinstance(new_sr, int):
        raise ValueError("old_sr and new_sr should be integers")
    if old_sr <= 0 or new_sr <= 0:
        raise ValueError(
            f"old_sr and new_sr should be positive, got {old_sr} and {new_sr}"
        )
    if x.ndim != 3:
        raise ValueError(
            f"resample expects a [B, C, T] array, got shape {tuple(x.shape)}"
        )

    batch_size, c, length = x.shape

    # Integer arithmetic, on the unreduced rates, so the output length never
    # depends on how `new_sr * length / old_sr` happens to round.
    max_output_length = -(-new_sr * length // old_sr)
    default_output_length = new_sr * length // old_sr

    # Validated before the equal-rate short circuit below, so that a bad
    # `output_length` is an error at every rate rather than only at some of them.
    if output_length is None:
        applied_output_length = max_output_length if full else default_output_length
    elif full:
        raise ValueError("You cannot pass both full=True and output_length")
    elif output_length < 0 or output_length > max_output_length:
        raise ValueError(f"output_length must be between 0 and {max_output_length}")
    else:
        applied_output_length = output_length

    if new_sr == old_sr:
        return x[..., :applied_output_length]

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

    return y[..., :applied_output_length]
