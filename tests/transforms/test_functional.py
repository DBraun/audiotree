"""Tests for function-based transforms with decorators."""

import tempfile
from pathlib import Path

import argbind
import jax
import numpy as np
import soundfile as sf

from audiotree import AudioTree
from audiotree.sources import create_audio_dataset
from audiotree.transforms.functional import (
    volume_norm,
    volume_change,
    trim,
    invert_phase,
    mono,
    stereo,
)


def _create_test_audio_file(tmpdir, filename, duration=3.0, sample_rate=44100):
    """Helper to create a test audio file."""
    audio_path = Path(tmpdir) / filename
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    sf.write(str(audio_path), audio, sample_rate)
    return str(audio_path)


class TestRandomTransformDecorator:
    """Test @random_transform decorator."""

    def test_volume_norm_basic(self):
        """Test volume_norm function transform."""
        audio_tree = AudioTree(
            np.random.randn(4, 1, 44100).astype(np.float32) * 0.1,
            sample_rate=44100,
        )
        audio_tree = audio_tree.replace_loudness()

        # Create transform with flat parameters
        transform = volume_norm(min_db=-20, max_db=-15)

        # Apply transform
        rng = np.random.default_rng(42)
        result = transform.random_map(audio_tree, rng)

        # Verify loudness is in range
        assert result.loudness is not None
        assert np.all(result.loudness >= -20 - 1)
        assert np.all(result.loudness <= -15 + 1)

    def test_volume_norm_with_prob(self):
        """Test volume_norm with probability."""
        audio_tree = AudioTree(
            np.random.randn(4, 1, 44100).astype(np.float32) * 0.1,
            sample_rate=44100,
        )
        audio_tree = audio_tree.replace_loudness()

        # Create transform with prob
        transform = volume_norm(min_db=-20, max_db=-15, prob=0.5)

        # Apply multiple times - some should be unchanged
        results = []
        for i in range(10):
            rng = np.random.default_rng(i)
            result = transform.random_map(audio_tree, rng)
            results.append(result)

        # Check that some were transformed and some weren't
        # (This is probabilistic but with 10 samples should be reliable)
        loudness_values = [float(r.loudness[0]) for r in results]
        assert len(set(loudness_values)) > 1  # Should have different values

    def test_volume_change_basic(self):
        """Test volume_change function transform."""
        audio_tree = AudioTree(
            np.random.randn(4, 1, 44100).astype(np.float32) * 0.1,
            sample_rate=44100,
        )

        # Create transform
        transform = volume_change(min_db=-6, max_db=6)

        # Apply transform
        rng = np.random.default_rng(42)
        result = transform.random_map(audio_tree, rng)

        # Verify audio was changed
        assert not np.array_equal(result.waveform, audio_tree.waveform)

    def test_invert_phase_basic(self):
        """Test invert_phase function transform."""
        audio_tree = AudioTree(
            np.random.randn(4, 1, 44100).astype(np.float32) * 0.1,
            sample_rate=44100,
        )

        # Create transform
        transform = invert_phase()

        # Apply transform
        rng = np.random.default_rng(42)
        result = transform.random_map(audio_tree, rng)

        # Verify phase was inverted
        np.testing.assert_array_almost_equal(
            result.waveform,
            -audio_tree.waveform,
        )


class TestMapTransformDecorator:
    """Test @map_transform decorator."""

    def test_trim_basic(self):
        """Test trim function transform."""
        audio_tree = AudioTree(
            np.random.randn(4, 1, 44100 * 5).astype(np.float32) * 0.1,
            sample_rate=44100,
        )

        # Create transform
        transform = trim(length=3.0)

        # Apply transform
        result = transform.map(audio_tree)

        # Verify length
        expected_length = int(3.0 * 44100)
        assert result.waveform.shape[-1] == expected_length

    def test_trim_shorter_audio(self):
        """Test trim with audio shorter than target pads audio."""
        audio_tree = AudioTree(
            np.random.randn(4, 1, 44100 * 2).astype(np.float32) * 0.1,
            sample_rate=44100,
        )

        # Create transform with longer target (default mode="wrap")
        transform = trim(length=5.0)

        # Apply transform
        result = transform.map(audio_tree)

        # Audio should be padded to target length
        assert result.waveform.shape[-1] == int(5.0 * 44100)

    def test_mono_conversion(self):
        """Test mono function transform."""
        audio_tree = AudioTree(
            np.random.randn(4, 2, 44100).astype(np.float32) * 0.1,
            sample_rate=44100,
        )

        # Create transform
        transform = mono()

        # Apply transform
        result = transform.map(audio_tree)

        # Verify mono
        assert result.waveform.shape[1] == 1

    def test_stereo_conversion(self):
        """Test stereo function transform."""
        audio_tree = AudioTree(
            np.random.randn(4, 1, 44100).astype(np.float32) * 0.1,
            sample_rate=44100,
        )

        # Create transform
        transform = stereo()

        # Apply transform
        result = transform.map(audio_tree)

        # Verify stereo
        assert result.waveform.shape[1] == 2


class TestDatasetChaining:
    """Test transforms work with dataset chaining."""

    def test_chain_with_dataset(self):
        """Test chaining function transforms with dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_test_audio_file(tmpdir, "test.wav", duration=5.0)

            # Create dataset
            ds = create_audio_dataset(
                sources=tmpdir,
                shuffle=False,
                repeat=False,
                sample_rate=44100,
                duration=5.0,
            ).slice(slice(0, 5))

            # Chain transforms
            ds = ds.random_map(volume_norm(min_db=-20, max_db=-15), seed=42)
            ds = ds.map(trim(length=3.0))

            # Load item
            item = ds[0]

            # Verify transforms were applied. trim runs last and changes the
            # audio length, so it invalidates the loudness set by volume_norm.
            assert item.loudness is None
            assert item.waveform.shape[-1] == int(3.0 * 44100)

    def test_multiple_random_transforms(self):
        """Test chaining multiple random transforms."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_test_audio_file(tmpdir, "test.wav", duration=3.0)

            # Create dataset
            ds = create_audio_dataset(
                sources=tmpdir,
                shuffle=False,
                repeat=False,
                sample_rate=44100,
                duration=3.0,
            ).slice(slice(0, 5))

            # Chain multiple random transforms
            ds = ds.random_map(volume_norm(min_db=-20, max_db=-15), seed=42)
            ds = ds.random_map(volume_change(min_db=-6, max_db=6), seed=43)
            ds = ds.random_map(invert_phase(prob=0.5), seed=44)

            # Load item
            item = ds[0]

            # Verify all transforms were applied
            assert item.loudness is not None


class TestArgBindIntegration:
    """Test integration with argbind."""

    def test_bind_volume_norm(self):
        """Test binding volume_norm with argbind."""
        volume_norm_bound = argbind.bind(volume_norm)

        # Create audio
        audio_tree = AudioTree(
            np.random.randn(4, 1, 44100).astype(np.float32) * 0.1,
            sample_rate=44100,
        )
        audio_tree = audio_tree.replace_loudness()

        # Set args
        args = {
            "volume_norm.min_db": -30,
            "volume_norm.max_db": -10,
            "volume_norm.prob": 0.9,
        }

        # Apply with scope
        with argbind.scope(args):
            transform = volume_norm_bound()
            rng = np.random.default_rng(42)
            result = transform.random_map(audio_tree, rng)

        # Verify parameters were used
        assert result.loudness is not None

    def test_bind_trim(self):
        """Test binding trim with argbind."""
        trim_bound = argbind.bind(trim)

        # Create audio
        audio_tree = AudioTree(
            np.random.randn(4, 1, 44100 * 5).astype(np.float32) * 0.1,
            sample_rate=44100,
        )

        # Set args
        args = {
            "trim.length": 2.0,
        }

        # Apply with scope
        with argbind.scope(args):
            transform = trim_bound()
            result = transform.map(audio_tree)

        # Verify length
        expected_length = int(2.0 * 44100)
        assert result.waveform.shape[-1] == expected_length

    def test_bind_with_dataset(self):
        """Test argbind with dataset chaining."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_test_audio_file(tmpdir, "test.wav", duration=5.0)

            # Bind transforms
            volume_norm_bound = argbind.bind(volume_norm)
            trim_bound = argbind.bind(trim)

            # Create dataset
            ds = create_audio_dataset(
                sources=tmpdir,
                shuffle=False,
                repeat=False,
                sample_rate=44100,
                duration=5.0,
            ).slice(slice(0, 5))

            # Set args
            args = {
                "volume_norm.min_db": -25,
                "volume_norm.max_db": -15,
                "trim.length": 3.0,
            }

            # Chain with scope
            with argbind.scope(args):
                ds = ds.random_map(volume_norm_bound(), seed=42)
                ds = ds.map(trim_bound())

            # Load item
            item = ds[0]

            # Verify transforms applied with correct parameters. trim runs last
            # and changes the audio length, invalidating volume_norm's loudness.
            assert item.loudness is None
            assert item.waveform.shape[-1] == int(3.0 * 44100)


class TestDefaultParameters:
    """Test that default parameters work correctly."""

    def test_volume_norm_defaults(self):
        """Test volume_norm with default parameters."""
        audio_tree = AudioTree(
            np.random.randn(4, 1, 44100).astype(np.float32) * 0.1,
            sample_rate=44100,
        )
        audio_tree = audio_tree.replace_loudness()

        # Create transform with defaults
        transform = volume_norm()

        # Apply transform
        rng = np.random.default_rng(42)
        result = transform.random_map(audio_tree, rng)

        # Should work but not change much (min=max=0)
        assert result.loudness is not None

    def test_trim_default(self):
        """Test trim with default parameters."""
        audio_tree = AudioTree(
            np.random.randn(4, 1, 44100 * 3).astype(np.float32) * 0.1,
            sample_rate=44100,
        )

        # Create transform with defaults (length=1.0)
        transform = trim()

        # Apply transform
        result = transform.map(audio_tree)

        # Verify default length was used
        expected_length = int(1.0 * 44100)
        assert result.waveform.shape[-1] == expected_length
