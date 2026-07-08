"""
Basic example of using argbind with AudioTree transforms.

This example demonstrates:
1. Binding transform functions with argbind
2. Configuring transforms via YAML files or command-line arguments
3. Applying random and deterministic transforms to audio

Usage:
    # Run with default config file
    python main.py --args.load=config.yml

    # Save arguments used during a run
    python main.py --args.load=config.yml --args.save=run_config.yml

    # Note: Argbind does not support JSON syntax for dict parameters from CLI.
    # Use YAML files for configuration instead.
"""

import argbind
import numpy as np

from audiotree import AudioTree
from audiotree import transforms

# Bind transform functions to argbind
# This allows their parameters to be set via CLI or YAML
volume_norm = argbind.bind(transforms.volume_norm)
trim = argbind.bind(transforms.trim)


def main():
    """Demonstrate basic AudioTree transform usage with argbind configuration."""
    # Create a batch of random audio (4 samples, 1 channel, 4 seconds at 44.1kHz)
    B = 4
    T = 44100
    audio_tree = AudioTree(
        np.random.uniform(-1, 1, size=(B, 1, T * 4)), sample_rate=44100
    )

    # Compute loudness (required for volume_norm transform)
    audio_tree = audio_tree.replace_loudness()
    loudness_before = audio_tree.loudness
    print("Before:", loudness_before)

    # Create RNG for deterministic randomness
    rng = np.random.default_rng(42)

    # Apply volume_norm transform (normalizes loudness to a random value in [min_db, max_db])
    # The config values (min_db, max_db) are set via argbind from config.yml or CLI
    audio_tree = volume_norm().random_map(audio_tree, rng)
    loudness_after = audio_tree.loudness
    print("After:", loudness_after)

    # Apply volume_norm again to show it can produce different results with same RNG
    audio_tree = volume_norm().random_map(audio_tree, rng)
    loudness_after = audio_tree.loudness
    print("After again:", loudness_after)

    # Apply trim transform (deterministic - trims to specified length)
    print("length:", audio_tree.waveform.shape[-1])
    audio_tree = trim().map(audio_tree)
    print("after length:", audio_tree.waveform.shape[-1])


if __name__ == "__main__":
    # Parse command-line arguments
    args = argbind.parse_args()

    # Apply arguments within scope and run main
    with argbind.scope(args):
        main()
