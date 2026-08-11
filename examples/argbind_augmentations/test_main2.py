"""Tests for main2.py - advanced argbind usage with scoped JAX transforms."""

import subprocess
import sys

import argbind
from audiotree.transforms import jax as jax_transforms


def test_main2_with_config():
    """Test that main2.py runs successfully with config2.yml."""
    result = subprocess.run(
        [
            sys.executable,
            "examples/argbind_augmentations/main2.py",
            "--args.load=examples/argbind_augmentations/config2.yml",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
    )

    assert result.returncode == 0, f"Script failed with stderr: {result.stderr}"
    assert "Training..." in result.stdout
    assert "Validation..." in result.stdout
    assert "Before:" in result.stdout
    assert "After:" in result.stdout


def test_main2_different_train_val_configs():
    """Test that train and val scopes can have different configurations."""
    result = subprocess.run(
        [
            sys.executable,
            "examples/argbind_augmentations/main2.py",
            "--args.load=examples/argbind_augmentations/config2.yml",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
    )

    assert result.returncode == 0, f"Script failed with stderr: {result.stderr}"

    lines = result.stdout.split("\n")

    train_after_idx = None
    val_after_idx = None

    for i, line in enumerate(lines):
        if "Training..." in line:
            for j in range(i, min(i + 10, len(lines))):
                if "After:" in lines[j]:
                    train_after_idx = j
                    break
        if "Validation..." in line:
            for j in range(i, min(i + 10, len(lines))):
                if "After:" in lines[j]:
                    val_after_idx = j
                    break

    assert train_after_idx is not None, "Could not find training 'After' line"
    assert val_after_idx is not None, "Could not find validation 'After' line"


def test_bind_module_with_filter():
    """Test that bind_module correctly filters transforms."""

    def filter_fn(fn):
        """Only bind transform functions (excludes classes)."""
        return callable(fn) and not isinstance(fn, type)

    transforms_bound = argbind.bind_module(
        jax_transforms, "test_train", "test_val", filter_fn=filter_fn
    )

    assert hasattr(transforms_bound, "volume_norm")
    assert hasattr(transforms_bound, "trim")
    assert hasattr(transforms_bound, "volume_change")


def test_scoped_augmentation():
    """Test that bind_module correctly binds JAX transforms for different scopes.

    This test runs in a subprocess to avoid argbind state pollution from other tests.
    When argbind binds functions with the same name (e.g., volume_norm from both
    NumPy and JAX modules), the registry can get confused.
    """
    # Run in subprocess for isolation from other argbind bindings
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import argbind
from audiotree import AudioTree
from audiotree.transforms import jax as jax_transforms
from jax import random
import numpy as np

def filter_fn(fn):
    return callable(fn) and not isinstance(fn, type)

transforms_bound = argbind.bind_module(
    jax_transforms, "test_train", "test_val", filter_fn=filter_fn
)

B = 2
T = 44100

audio_tree = AudioTree(np.random.uniform(-1, 1, size=(B, 1, T)), sample_rate=44100)
audio_tree = audio_tree.replace_lufs()

train_args = {
    "test_train/volume_norm.min_db": -25,
    "test_train/volume_norm.max_db": -15,
}

val_args = {
    "test_val/volume_norm.min_db": -20,
    "test_val/volume_norm.max_db": -20,
}

rng = random.key(0)

with argbind.scope(train_args, "test_train"):
    transform = transforms_bound.volume_norm()
    train_audio = transform.random_map(audio_tree, rng)

with argbind.scope(val_args, "test_val"):
    transform = transforms_bound.volume_norm()
    val_audio = transform.random_map(audio_tree, rng)

assert train_audio.lufs is not None, "train_audio.lufs is None"
assert val_audio.lufs is not None, "val_audio.lufs is None"

assert np.all(train_audio.lufs >= -25 - 1), f"train loudness too low: {train_audio.lufs}"
assert np.all(train_audio.lufs <= -15 + 1), f"train loudness too high: {train_audio.lufs}"

assert np.all(val_audio.lufs >= -20 - 1), f"val loudness too low: {val_audio.lufs}"
assert np.all(val_audio.lufs <= -20 + 1), f"val loudness too high: {val_audio.lufs}"

print("SUCCESS")
""",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
    )

    assert result.returncode == 0, f"Test failed with stderr: {result.stderr}"
    assert "SUCCESS" in result.stdout
