"""Test scope functionality with Dict[str, AudioTree] inputs."""

import numpy as np

from audiotree import AudioTree
from audiotree.transforms.functional import (
    volume_norm,
    volume_change,
    trim,
    invert_phase,
)


class TestScopeWithDict:
    """Test that scope works correctly with Dict[str, AudioTree]."""

    def test_scope_single_key(self):
        """Test applying transform only to 'src' key in dict."""
        # Create batch with multiple AudioTrees
        audio1 = AudioTree(
            np.random.randn(2, 1, 44100).astype(np.float32) * 0.1,
            sample_rate=44100,
        )
        audio2 = AudioTree(
            np.random.randn(2, 1, 44100).astype(np.float32) * 0.1,
            sample_rate=44100,
        )

        audio1 = audio1.replace_lufs()
        audio2 = audio2.replace_lufs()

        batch = {"src": audio1, "target": audio2}

        # Create transform with scope - only transform 'src'
        transform = volume_norm(
            min_db=-20,
            max_db=-15,
            scope={"src": {"scope": True}},
        )

        # Apply to batch
        rng = np.random.default_rng(42)
        result_batch = transform.random_map(batch, rng)

        # Verify only src was transformed
        assert not np.array_equal(result_batch["src"].lufs, audio1.lufs)
        assert np.array_equal(result_batch["target"].lufs, audio2.lufs)

    def test_scope_all_keys(self):
        """Test applying transform to all keys (no scope = transform everything)."""
        # Create batch
        audio1 = AudioTree(
            np.random.randn(2, 1, 44100).astype(np.float32) * 0.1,
            sample_rate=44100,
        )
        audio2 = AudioTree(
            np.random.randn(2, 1, 44100).astype(np.float32) * 0.1,
            sample_rate=44100,
        )

        audio1 = audio1.replace_lufs()
        audio2 = audio2.replace_lufs()

        batch = {"src": audio1, "target": audio2}

        # Create transform with no scope - should transform both
        transform = volume_norm(
            min_db=-20,
            max_db=-15,
        )

        # Apply to batch
        rng = np.random.default_rng(42)
        result_batch = transform.random_map(batch, rng)

        # Verify both were transformed
        assert not np.array_equal(result_batch["src"].lufs, audio1.lufs)
        assert not np.array_equal(result_batch["target"].lufs, audio2.lufs)

    def test_scope_multiple_keys(self):
        """Test applying transform to multiple specific keys."""
        # Create batch with 3 keys
        audio1 = AudioTree(
            np.random.randn(2, 1, 44100).astype(np.float32) * 0.1,
            sample_rate=44100,
        )
        audio2 = AudioTree(
            np.random.randn(2, 1, 44100).astype(np.float32) * 0.1,
            sample_rate=44100,
        )
        audio3 = AudioTree(
            np.random.randn(2, 1, 44100).astype(np.float32) * 0.1,
            sample_rate=44100,
        )

        audio1 = audio1.replace_lufs()
        audio2 = audio2.replace_lufs()
        audio3 = audio3.replace_lufs()

        batch = {"dry": audio1, "wet": audio2, "reference": audio3}

        # Create transform - only transform 'dry' and 'wet'
        transform = volume_norm(
            min_db=-20,
            max_db=-15,
            scope={
                "dry": {"scope": True},
                "wet": {"scope": True},
            },
        )

        # Apply to batch
        rng = np.random.default_rng(42)
        result_batch = transform.random_map(batch, rng)

        # Verify dry and wet transformed, reference not
        assert not np.array_equal(result_batch["dry"].lufs, audio1.lufs)
        assert not np.array_equal(result_batch["wet"].lufs, audio2.lufs)
        assert np.array_equal(result_batch["reference"].lufs, audio3.lufs)

    def test_scope_with_nested_dict(self):
        """Test scope with nested dictionary structure."""
        # Create nested batch
        audio1 = AudioTree(
            np.random.randn(2, 1, 44100).astype(np.float32) * 0.1,
            sample_rate=44100,
        )
        audio2 = AudioTree(
            np.random.randn(2, 1, 44100).astype(np.float32) * 0.1,
            sample_rate=44100,
        )
        audio3 = AudioTree(
            np.random.randn(2, 1, 44100).astype(np.float32) * 0.1,
            sample_rate=44100,
        )

        audio1 = audio1.replace_lufs()
        audio2 = audio2.replace_lufs()
        audio3 = audio3.replace_lufs()

        batch = {
            "input": {"dry": audio1, "wet": audio2},
            "target": audio3,
        }

        # Create transform - only transform input.dry
        transform = volume_norm(
            min_db=-20,
            max_db=-15,
            scope={"input": {"dry": {"scope": True}}},
        )

        # Apply to batch
        rng = np.random.default_rng(42)
        result_batch = transform.random_map(batch, rng)

        # Verify only input.dry transformed
        assert not np.array_equal(result_batch["input"]["dry"].lufs, audio1.lufs)
        assert np.array_equal(result_batch["input"]["wet"].lufs, audio2.lufs)
        assert np.array_equal(result_batch["target"].lufs, audio3.lufs)


class TestScopeWithOutputKey:
    """Test scope combined with output_key."""

    def test_scope_with_output_key(self):
        """Test applying transform with scope and output_key."""
        # Create batch
        audio1 = AudioTree(
            np.random.randn(2, 1, 44100).astype(np.float32) * 0.1,
            sample_rate=44100,
        )
        audio2 = AudioTree(
            np.random.randn(2, 1, 44100).astype(np.float32) * 0.1,
            sample_rate=44100,
        )

        audio1 = audio1.replace_lufs()
        audio2 = audio2.replace_lufs()

        batch = {"src": audio1, "target": audio2}

        # Transform only 'src' and output to 'src_modified'
        transform = volume_norm(
            min_db=-20,
            max_db=-15,
            scope={"src": {"scope": True}},
            output_key="modified",
        )

        # Apply to batch
        rng = np.random.default_rng(42)
        result_batch = transform.random_map(batch, rng)

        # Should have: src (original), target (original), modified (new)
        assert "src" in result_batch
        assert "target" in result_batch
        assert "modified" in result_batch

        # Original should be unchanged
        assert np.array_equal(result_batch["src"].lufs, audio1.lufs)
        assert np.array_equal(result_batch["target"].lufs, audio2.lufs)

        # Modified should be transformed
        assert not np.array_equal(result_batch["modified"].lufs, audio1.lufs)


class TestScopeWithMapTransforms:
    """Test scope with deterministic map transforms."""

    def test_trim_with_scope(self):
        """Test trim transform with scope."""
        # Create batch
        audio1 = AudioTree(
            np.random.randn(2, 1, 44100 * 5).astype(np.float32) * 0.1,
            sample_rate=44100,
        )
        audio2 = AudioTree(
            np.random.randn(2, 1, 44100 * 5).astype(np.float32) * 0.1,
            sample_rate=44100,
        )

        batch = {"dry": audio1, "wet": audio2}

        # Trim only 'dry'
        transform = trim(
            length=3.0,
            scope={"dry": {"scope": True}},
        )

        # Apply to batch
        result_batch = transform.map(batch)

        # Verify only dry was trimmed
        assert result_batch["dry"].waveform.shape[-1] == int(3.0 * 44100)
        assert result_batch["wet"].waveform.shape[-1] == int(5.0 * 44100)


class TestCommonTrainingPipelinePattern:
    """Test the common training pipeline pattern with Dict[str, AudioTree]."""

    def test_dry_wet_processing(self):
        """Test processing dry and wet signals differently."""
        # Create dry and wet audio
        dry_audio = AudioTree(
            np.random.randn(4, 1, 44100 * 3).astype(np.float32) * 0.1,
            sample_rate=44100,
        )
        wet_audio = AudioTree(
            np.random.randn(4, 1, 44100 * 3).astype(np.float32) * 0.1,
            sample_rate=44100,
        )

        dry_audio = dry_audio.replace_lufs()
        wet_audio = wet_audio.replace_lufs()

        batch = {"dry": dry_audio, "wet": wet_audio}

        # Process dry and wet differently
        # 1. Normalize both
        transform1 = volume_norm(min_db=-20, max_db=-15)
        batch = transform1.random_map(batch, np.random.default_rng(42))

        # 2. Add volume change only to wet
        transform2 = volume_change(
            min_db=-6,
            max_db=6,
            scope={"wet": {"scope": True}},
        )
        batch = transform2.random_map(batch, np.random.default_rng(43))

        # 3. Invert phase on dry only (50% prob)
        transform3 = invert_phase(
            prob=0.5,
            scope={"dry": {"scope": True}},
        )
        batch = transform3.random_map(batch, np.random.default_rng(44))

        # Verify structure preserved
        assert "dry" in batch
        assert "wet" in batch
        assert isinstance(batch["dry"], AudioTree)
        assert isinstance(batch["wet"], AudioTree)

    def test_input_target_pattern(self):
        """Test common input/target pattern."""
        # Create input and target
        input_audio = AudioTree(
            np.random.randn(4, 1, 44100 * 3).astype(np.float32) * 0.1,
            sample_rate=44100,
        )
        target_audio = AudioTree(
            np.random.randn(4, 1, 44100 * 3).astype(np.float32) * 0.1,
            sample_rate=44100,
        )

        input_audio = input_audio.replace_lufs()
        target_audio = target_audio.replace_lufs()

        batch = {"input": input_audio, "target": target_audio}

        # Apply shared transforms to both
        transform1 = volume_norm(min_db=-20, max_db=-15)
        batch = transform1.random_map(batch, np.random.default_rng(42))

        # Apply additional augmentation only to input
        transform2 = volume_change(
            min_db=-12,
            max_db=12,
            prob=0.9,
            scope={"input": {"scope": True}},
        )
        batch = transform2.random_map(batch, np.random.default_rng(43))

        # Both should be normalized
        assert batch["input"].lufs is not None
        assert batch["target"].lufs is not None

    def test_multi_key_batch(self):
        """Test batch with multiple audio keys (dry, wet, reference)."""
        # Create multiple audio sources
        dry = AudioTree(
            np.random.randn(4, 1, 44100 * 2).astype(np.float32) * 0.1,
            sample_rate=44100,
        )
        wet = AudioTree(
            np.random.randn(4, 1, 44100 * 2).astype(np.float32) * 0.1,
            sample_rate=44100,
        )
        reference = AudioTree(
            np.random.randn(4, 1, 44100 * 2).astype(np.float32) * 0.1,
            sample_rate=44100,
        )

        dry = dry.replace_lufs()
        wet = wet.replace_lufs()
        reference = reference.replace_lufs()

        batch = {"dry": dry, "wet": wet, "reference": reference}

        original_ref_loudness = reference.lufs.copy()

        # Normalize dry and wet, but not reference
        transform = volume_norm(
            min_db=-20,
            max_db=-15,
            scope={
                "dry": {"scope": True},
                "wet": {"scope": True},
            },
        )

        rng = np.random.default_rng(42)
        result = transform.random_map(batch, rng)

        # Verify dry and wet were transformed
        assert not np.array_equal(result["dry"].lufs, dry.lufs)
        assert not np.array_equal(result["wet"].lufs, wet.lufs)

        # Verify reference was not transformed
        assert np.array_equal(result["reference"].lufs, original_ref_loudness)
