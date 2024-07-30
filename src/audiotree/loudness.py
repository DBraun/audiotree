from functools import partial

from einops import rearrange
import jax
import jaxloudnorm as jln
from jax import numpy as jnp


@partial(jax.jit, static_argnames=("sample_rate", "zeros"))
def jit_integrated_loudness(data, sample_rate, zeros: int):

    block_size = 0.4

    original_length = data.shape[-1]
    signal_duration = original_length / sample_rate

    if signal_duration < block_size:
        data = jnp.pad(
            data,
            pad_width=(
                (0, 0),
                (0, 0),
                (0, int(block_size * sample_rate) - original_length),
            ),
        )

    data = rearrange(data, "b c t -> b t c")
    meter = jln.Meter(sample_rate, block_size=block_size, use_fir=True, zeros=zeros)
    loudness = jax.vmap(meter.integrated_loudness)(data)

    loudness = jnp.where(
        jnp.isnan(loudness), jnp.full_like(loudness, -200), loudness
    )  # todo: -200 dB good default?

    return loudness
