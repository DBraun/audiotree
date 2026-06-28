"""AudioTree transforms for data augmentation.

This module provides two sets of transforms:

1. **NumPy transforms** (this module, ``audiotree.transforms``):
   For CPU-based grain data pipelines. Uses NumPy operations and np.random.Generator.

2. **JAX transforms** (``audiotree.transforms.jax``):
   For GPU/JIT training pipelines. Uses JAX operations and jax.random.key.

Example - CPU (grain pipeline)::

    from audiotree.transforms import volume_norm, trim

    # Create transform and apply with grain
    transform = volume_norm(min_db=-20, max_db=-15)
    ds = ds.random_map(transform, seed=42)

Example - GPU (jitted training step)::

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
    - :mod:`audiotree.transforms.jax` for GPU transforms
"""

from .functional import identity
from .functional import mono
from .functional import stereo
from .functional import resample
from .functional import volume_change
from .functional import volume_norm
from .functional import rescale_audio
from .functional import peak_norm
from .functional import invert_phase
from .functional import swap_stereo
from .functional import corrupt_phase
from .functional import shift_phase
from .functional import roll
from .functional import choose
from .functional import encode_with_codec
from .functional import encode_latents
from .functional import trim

# Batch is special - it's a grain BatchOperation, not a regular transform
from .core import Batch

__all__ = [
    "identity",
    "mono",
    "stereo",
    "resample",
    "volume_change",
    "volume_norm",
    "rescale_audio",
    "peak_norm",
    "invert_phase",
    "swap_stereo",
    "corrupt_phase",
    "shift_phase",
    "roll",
    "choose",
    "encode_with_codec",
    "encode_latents",
    "trim",
    "Batch",
]
