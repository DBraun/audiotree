from functools import partial
import math

import jax
import jaxloudnorm as jln
from jax import numpy as jnp


@partial(jax.jit, static_argnames=("sample_rate", "zeros"))
def jit_integrated_loudness(data: jnp.ndarray, sample_rate: int, zeros: int):

    block_size = 0.4
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
    loudness = jax.vmap(meter.integrated_loudness)(data)

    loudness = jnp.where(
        jnp.isnan(loudness), jnp.full_like(loudness, -200), loudness
    )  # todo: -200 dB good default?

    return loudness
