"""Tests for main.py - basic argbind usage with transforms."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

from audiotree import AudioTree, transforms
import argbind


def test_main_with_config():
    """Test that main.py runs successfully with config.yml."""
    cwd = Path(__file__).parent.parent.parent
    result = subprocess.run(
        [
            sys.executable,
            "examples/argbind_augmentations/main.py",
            "--args.load=examples/argbind_augmentations/config.yml",
        ],
        capture_output=True,
        text=True,
        cwd=str(cwd),
    )

    assert result.returncode == 0, f"Script failed with stderr: {result.stderr}"
    assert "Before:" in result.stdout
    assert "After:" in result.stdout
    assert "length:" in result.stdout
    assert "after length:" in result.stdout


def test_main_with_custom_yaml():
    """Test that main.py works with a custom YAML config."""
    cwd = Path(__file__).parent.parent.parent

    # Create a temporary config file with custom values
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        config = {
            "volume_norm.min_db": -30,
            "volume_norm.max_db": -10,
            "trim.length": 2.0,
        }
        yaml.dump(config, f)
        config_path = f.name

    try:
        result = subprocess.run(
            [
                sys.executable,
                "examples/argbind_augmentations/main.py",
                f"--args.load={config_path}",
            ],
            capture_output=True,
            text=True,
            cwd=str(cwd),
        )

        assert result.returncode == 0, f"Script failed with stderr: {result.stderr}"
        assert "Before:" in result.stdout
        assert "After:" in result.stdout
        assert "after length: 88200" in result.stdout  # 2.0 seconds at 44100 Hz
    finally:
        os.unlink(config_path)


def test_volume_norm_binding():
    """Test that volume_norm can be bound with argbind."""
    import jax

    VolumeNorm = argbind.bind(transforms.volume_norm)

    B = 4
    T = 44100
    audio_tree = AudioTree(np.random.uniform(-1, 1, size=(B, 1, T)), sample_rate=44100)
    audio_tree = audio_tree.replace_loudness()
    loudness_before = audio_tree.loudness

    rng = jax.random.key(42)

    args = {
        "volume_norm.min_db": -20,
        "volume_norm.max_db": -15,
    }

    with argbind.scope(args):
        transform = VolumeNorm()
        audio_tree_normalized = transform.random_map(audio_tree, rng)

    assert audio_tree_normalized.loudness is not None
    loudness_after = audio_tree_normalized.loudness
    assert np.all(loudness_after >= -20 - 1)
    assert np.all(loudness_after <= -15 + 1)
    assert not np.array_equal(loudness_before, loudness_after)


def test_trim_binding():
    """Test that trim can be bound with argbind."""
    Trim = argbind.bind(transforms.trim)

    B = 4
    T = 44100
    audio_tree = AudioTree(np.random.uniform(-1, 1, size=(B, 1, T * 4)), sample_rate=44100)

    args = {
        "trim.length": 1.0,
    }

    with argbind.scope(args):
        transform = Trim()
        audio_tree_trimmed = transform.map(audio_tree)

    expected_length = int(1.0 * 44100)
    assert audio_tree_trimmed.audio_data.shape[-1] == expected_length
