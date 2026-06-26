"""JAX-based transforms for GPU/JIT training pipelines.

Use these transforms inside @jax.jit functions or when working with JAX arrays on GPU.
These transforms use JAX operations and accept ``jax.random.key`` for random transforms.

For CPU-based grain data pipelines, use ``audiotree.transforms`` instead.

Example - Direct usage::

    from audiotree.transforms import jax as jax_transforms
    import jax

    # Create transform and apply with JAX key
    transform = jax_transforms.volume_norm(min_db=-20, max_db=-15)
    rng = jax.random.key(42)
    result = transform.random_map(audio_tree, rng)

Example - ArgBind-configured training pipeline::

    from audiotree.transforms import jax as jax_transforms
    import argbind

    # Bind all transforms for YAML configuration
    transforms_lib = argbind.bind_module(jax_transforms)

    @argbind.bind("train", "val")
    def augment_batch(rng, batch, transforms: list[str] = None):
        for transform_name in transforms or []:
            transform = getattr(transforms_lib, transform_name)()
            if hasattr(transform, "random_map"):
                rng, subkey = jax.random.split(rng)
                batch = transform.random_map(batch, subkey)
            elif hasattr(transform, "map"):
                batch = transform.map(batch)
        return batch

See also:
    - :mod:`audiotree.transforms` for CPU/grain transforms
"""

from .functional import (
    volume_norm,
    volume_change,
    invert_phase,
    trim,
    mono,
    stereo,
    identity,
    rescale_audio,
    peak_normalize,
    swap_stereo,
    corrupt_phase,
    shift_phase,
    roll,
    encode_with_codec,
    encode_latents,
)

__all__ = [
    "volume_norm",
    "volume_change",
    "invert_phase",
    "trim",
    "mono",
    "stereo",
    "identity",
    "rescale_audio",
    "peak_normalize",
    "swap_stereo",
    "corrupt_phase",
    "shift_phase",
    "roll",
    "encode_with_codec",
    "encode_latents",
]
