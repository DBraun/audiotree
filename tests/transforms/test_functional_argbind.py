"""Test argbind integration with functional transforms (CLI and YAML)."""

import subprocess
import sys
import tempfile
from pathlib import Path

import argbind
import jax
import numpy as np
import yaml

from audiotree import AudioTree
from audiotree.transforms.functional import volume_norm, trim


class TestArgBindCLI:
    """Test that CLI parameters work correctly."""

    def test_trim_from_cli(self):
        """Test that --trim.length=4.0 works from command line."""
        # Create a test script
        script = '''
import argbind
import jax
import numpy as np
from audiotree import AudioTree
from audiotree.transforms.functional import trim

Trim = argbind.bind(trim)

args = argbind.parse_args()
with argbind.scope(args):
    transform = Trim()
    audio_tree = AudioTree(
        np.random.randn(1, 1, 44100 * 5).astype(np.float32) * 0.1,
        sample_rate=44100,
    )
    result = transform.map(audio_tree)
    print(f"Length: {result.audio_data.shape[-1]}")
    expected = int(4.0 * 44100)
    assert result.audio_data.shape[-1] == expected, f"Expected {expected}, got {result.audio_data.shape[-1]}"
    print("SUCCESS")
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script)
            script_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, script_path, '--trim.length=4.0'],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0, f"Script failed: {result.stderr}"
            assert "SUCCESS" in result.stdout
        finally:
            Path(script_path).unlink()

    def test_volume_norm_from_cli(self):
        """Test that --volume_norm.min_db=-30 works from command line."""
        script = '''
import argbind
import numpy as np
from audiotree import AudioTree
from audiotree.transforms.functional import volume_norm

VolumeNorm = argbind.bind(volume_norm)

args = argbind.parse_args()
with argbind.scope(args):
    transform = VolumeNorm()
    audio_tree = AudioTree(
        np.random.randn(1, 1, 44100).astype(np.float32) * 0.1,
        sample_rate=44100,
    )
    audio_tree = audio_tree.replace_loudness()

    rng = np.random.default_rng(42)
    result = transform.random_map(audio_tree, rng)

    # With min_db=max_db=-30, loudness should be exactly -30
    print(f"Loudness: {float(result.loudness[0])}")
    assert abs(float(result.loudness[0]) - (-30.0)) < 1.0
    print("SUCCESS")
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script)
            script_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, script_path, '--volume_norm.min_db=-30', '--volume_norm.max_db=-30'],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0, f"Script failed: {result.stderr}"
            assert "SUCCESS" in result.stdout
        finally:
            Path(script_path).unlink()

    def test_prob_from_cli(self):
        """Test that --volume_norm.prob=0.5 works from command line."""
        script = '''
import argbind
import numpy as np
from audiotree import AudioTree
from audiotree.transforms.functional import volume_norm

VolumeNorm = argbind.bind(volume_norm)

args = argbind.parse_args()
with argbind.scope(args):
    transform = VolumeNorm()
    audio_tree = AudioTree(
        np.random.randn(1, 1, 44100).astype(np.float32) * 0.1,
        sample_rate=44100,
    )
    audio_tree = audio_tree.replace_loudness()

    # Apply multiple times - some should be unchanged with prob=0.5
    original_loudness = float(audio_tree.loudness[0])
    different_count = 0

    for i in range(20):
        rng = np.random.default_rng(i)
        result = transform.random_map(audio_tree, rng)
        if abs(float(result.loudness[0]) - original_loudness) > 0.1:
            different_count += 1

    # With prob=0.5, roughly half should be different
    print(f"Different: {different_count}/20")
    assert 5 <= different_count <= 15, f"Expected ~10/20 different, got {different_count}"
    print("SUCCESS")
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script)
            script_path = f.name

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    script_path,
                    '--volume_norm.min_db=-30',
                    '--volume_norm.max_db=-30',
                    '--volume_norm.prob=0.5',
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0, f"Script failed: {result.stderr}"
            assert "SUCCESS" in result.stdout
        finally:
            Path(script_path).unlink()


class TestArgBindYAML:
    """Test that YAML configuration works correctly."""

    def test_trim_from_yaml(self):
        """Test that trim.length: 4 works from YAML file."""
        config = {
            "trim.length": 4.0,
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name

        script = f'''
import argbind
import numpy as np
from audiotree import AudioTree
from audiotree.transforms.functional import trim

Trim = argbind.bind(trim)

args = argbind.parse_args()
with argbind.scope(args):
    transform = Trim()
    audio_tree = AudioTree(
        np.random.randn(1, 1, 44100 * 5).astype(np.float32) * 0.1,
        sample_rate=44100,
    )
    result = transform.map(audio_tree)
    expected = int(4.0 * 44100)
    assert result.audio_data.shape[-1] == expected, f"Expected {{expected}}, got {{result.audio_data.shape[-1]}}"
    print("SUCCESS")
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script)
            script_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, script_path, f'--args.load={config_path}'],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0, f"Script failed: {result.stderr}"
            assert "SUCCESS" in result.stdout
        finally:
            Path(script_path).unlink()
            Path(config_path).unlink()

    def test_volume_norm_from_yaml(self):
        """Test volume_norm parameters from YAML."""
        config = {
            "volume_norm.min_db": -25,
            "volume_norm.max_db": -15,
            "volume_norm.prob": 0.9,
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name

        script = f'''
import argbind
import numpy as np
from audiotree import AudioTree
from audiotree.transforms.functional import volume_norm

VolumeNorm = argbind.bind(volume_norm)

args = argbind.parse_args()
with argbind.scope(args):
    transform = VolumeNorm()
    audio_tree = AudioTree(
        np.random.randn(1, 1, 44100).astype(np.float32) * 0.1,
        sample_rate=44100,
    )
    audio_tree = audio_tree.replace_loudness()

    rng = np.random.default_rng(42)
    result = transform.random_map(audio_tree, rng)

    loudness = float(result.loudness[0])
    print(f"Loudness: {{loudness}}")
    assert -25 - 1 <= loudness <= -15 + 1, f"Loudness {{loudness}} not in range [-26, -14]"
    print("SUCCESS")
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script)
            script_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, script_path, f'--args.load={config_path}'],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0, f"Script failed: {result.stderr}"
            assert "SUCCESS" in result.stdout
        finally:
            Path(script_path).unlink()
            Path(config_path).unlink()

    def test_scoped_yaml(self):
        """Test scoped configurations from YAML."""
        config = {
            "train/volume_norm.min_db": -30,
            "train/volume_norm.max_db": -10,
            "val/volume_norm.min_db": -20,
            "val/volume_norm.max_db": -20,
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name

        script = f'''
import argbind
import numpy as np
from audiotree import AudioTree
from audiotree.transforms.functional import volume_norm

VolumeNorm = argbind.bind(volume_norm, "train", "val")

args = argbind.parse_args()

audio_tree = AudioTree(
    np.random.randn(1, 1, 44100).astype(np.float32) * 0.1,
    sample_rate=44100,
)
audio_tree = audio_tree.replace_loudness()
rng = np.random.default_rng(42)

# Train scope
with argbind.scope(args, "train"):
    transform = VolumeNorm()
    train_result = transform.random_map(audio_tree, rng)
    train_loudness = float(train_result.loudness[0])
    print(f"Train loudness: {{train_loudness}}")

# Val scope - need a new rng to get different results
rng2 = np.random.default_rng(42)
with argbind.scope(args, "val"):
    transform = VolumeNorm()
    val_result = transform.random_map(audio_tree, rng2)
    val_loudness = float(val_result.loudness[0])
    print(f"Val loudness: {{val_loudness}}")

# They should be different because val has min_db=max_db=-20 (exact)
# while train has min_db=-30, max_db=-10 (range)
assert abs(train_loudness - val_loudness) > 1.0, "Train and val should use different configs"
print("SUCCESS")
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script)
            script_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, script_path, f'--args.load={config_path}'],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0, f"Script failed: {result.stderr}"
            assert "SUCCESS" in result.stdout
        finally:
            Path(script_path).unlink()
            Path(config_path).unlink()
