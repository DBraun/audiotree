"""
Advanced example using argbind with scoped transforms and bind_module.

This example demonstrates:
1. Using argbind.bind_module() to bind an entire module at once
2. Scoping transforms for different use cases (train vs val)
3. JAX JIT compilation with argbind
4. Applying multiple transforms in sequence

Usage:
    # Run with config file
    python main2.py --args.load=config2.yml

    # Note: Argbind does not support JSON syntax for dict parameters from CLI.
    # Use YAML files for configuration instead.

Key Concepts:
    - bind_module: Binds all transforms in the module at once (vs binding individually)
    - Scopes: Different configs for "train" vs "val" (e.g., different augmentation strengths)
    - JIT: JAX JIT compilation works seamlessly with argbind scopes
"""

from typing import Dict

import argbind
import grain
import jax
from jax import random
import numpy as np

from audiotree import AudioTree
from audiotree.transforms import jax as jax_transforms


def filter_fn(fn):
    """Only bind transform functions (excludes non-callables).

    Args:
        fn: A function or class from the transforms module

    Returns:
        bool: True if the function should be bound with argbind
    """
    # Bind if it's a callable that's not a class
    return callable(fn) and not isinstance(fn, type)


# Bind the entire JAX transforms module with scopes for "train" and "val"
# See: https://github.com/DBraun/argbind/tree/main/examples/bind_module
transforms_lib = argbind.bind_module(
    jax_transforms, "train", "val", filter_fn=filter_fn
)


@argbind.bind("train", "val")
def augment_batch(
    rng: jax.Array,
    batch: Dict[str, AudioTree],
    transforms: list[str] = None,
) -> Dict[str, AudioTree]:
    """Apply a sequence of transforms to a batch.

    This function is bound to both "train" and "val" scopes, allowing
    different transform configurations for each.

    Args:
        rng: JAX random key
        batch: Dict containing AudioTree under "src" key
        transforms: List of transform function names to apply in order

    Returns:
        Dict containing transformed AudioTree
    """
    transforms = transforms or []

    for transform_name in transforms:
        # Get the bound transform by name
        transform = getattr(transforms_lib, transform_name)()

        # Apply transform based on its type
        if isinstance(transform, grain.transforms.RandomMap):
            rng, subkey = random.split(rng)
            batch = transform.random_map(batch, subkey)
        elif isinstance(transform, grain.transforms.Map):
            batch = transform.map(batch)
        elif hasattr(transform, "np_random_map"):
            rng, subkey = random.split(rng)
            batch = transform.np_random_map(batch, subkey)
        else:
            raise ValueError(f"Unknown operation type: {type(transform)}")

    return batch


def main(audio_tree: AudioTree):
    """Apply augmentations to audio within current argbind scope.

    Args:
        audio_tree: Input audio to augment

    Returns:
        Augmented audio
    """
    rng = random.key(0)
    batch = {"src": audio_tree}
    batch = augment_batch(rng, batch)
    audio_tree = batch["src"]

    return audio_tree


if __name__ == "__main__":
    # Parse command-line arguments
    args = argbind.parse_args()

    # Create JIT-compiled versions for train and val
    # Note: JAX JIT compilation works with argbind scopes
    @jax.jit
    def train(audio_tree):
        """Training augmentation pipeline with train scope."""
        with argbind.scope(args, "train"):
            return main(audio_tree)

    @jax.jit
    def val(audio_tree):
        """Validation augmentation pipeline with val scope."""
        with argbind.scope(args, "val"):
            return main(audio_tree)

    # Create test audio
    B = 4
    T = 44100

    # Test training augmentation
    print("Training...")
    audio_tree = AudioTree(
        np.random.uniform(-1, 1, size=(B, 1, T * 4)), sample_rate=44100
    )
    audio_tree = audio_tree.replace_loudness()
    print("Before:", audio_tree.loudness)
    print("length:", audio_tree.waveform.shape[-1])
    audio_tree = train(audio_tree)
    print("After:", audio_tree.loudness)
    print("length:", audio_tree.waveform.shape[-1])

    # Test validation augmentation
    print("Validation...")
    audio_tree = AudioTree(
        np.random.uniform(-1, 1, size=(B, 1, T * 4)), sample_rate=44100
    )
    audio_tree = audio_tree.replace_loudness()
    print("Before:", audio_tree.loudness)
    print("length:", audio_tree.waveform.shape[-1])
    audio_tree = val(audio_tree)
    print("After:", audio_tree.loudness)
    print("length:", audio_tree.waveform.shape[-1])
