"""Tests for MemmapDataSource class."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from audiotree import MemmapWriter, FieldSpec
from audiotree.sources import MemmapDataSource


def _create_test_dataset(tmpdir, num_samples=10, include_strings=False):
    """Helper to create a test memmap dataset."""
    output_dir = Path(tmpdir)

    field_specs = [
        FieldSpec("audio", np.float32, (2, 100)),
        FieldSpec("params", np.float32, (5,)),
        FieldSpec("label", np.int32, ()),
    ]

    audio_data = np.random.randn(num_samples, 2, 100).astype(np.float32)
    params_data = np.random.randn(num_samples, 5).astype(np.float32)
    labels = np.arange(num_samples, dtype=np.int32)

    strings = None
    if include_strings:
        strings = {"code": [f"code_{i}" for i in range(num_samples)]}

    with MemmapWriter(output_dir, field_specs, expected_samples=num_samples) as writer:
        writer.write_batch(
            {"audio": audio_data, "params": params_data, "label": labels},
            strings=strings,
        )

    return output_dir, audio_data, params_data, labels


def test_basic_loading():
    """Test basic loading from memmap files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir, audio_data, params_data, labels = _create_test_dataset(tmpdir)

        source = MemmapDataSource(output_dir / "manifest.json")
        assert len(source) == 10

        sample = source[0]
        assert "audio" in sample
        assert "params" in sample
        assert "label" in sample

        # Check shapes (batch dimension added)
        assert sample["audio"].shape == (1, 2, 100)
        assert sample["params"].shape == (1, 5)
        assert sample["label"].shape == (1,)


def test_data_correctness():
    """Test that loaded data matches written data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir, audio_data, params_data, labels = _create_test_dataset(tmpdir)

        source = MemmapDataSource(output_dir / "manifest.json")

        for i in range(10):
            sample = source[i]
            np.testing.assert_array_almost_equal(
                sample["audio"][0], audio_data[i], decimal=5
            )
            np.testing.assert_array_almost_equal(
                sample["params"][0], params_data[i], decimal=5
            )
            assert sample["label"][0] == labels[i]


def test_from_directory():
    """Test convenience constructor from_directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir, _, _, _ = _create_test_dataset(tmpdir)

        source = MemmapDataSource.from_directory(output_dir)
        assert len(source) == 10


def test_num_records_limit():
    """Test limiting number of records."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir, _, _, _ = _create_test_dataset(tmpdir, num_samples=10)

        source = MemmapDataSource(output_dir / "manifest.json", num_records=5)
        assert len(source) == 5


def test_field_filtering():
    """Test loading only specific fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir, _, _, _ = _create_test_dataset(tmpdir)

        source = MemmapDataSource(
            output_dir / "manifest.json", fields=["audio", "label"]
        )

        sample = source[0]
        assert "audio" in sample
        assert "label" in sample
        assert "params" not in sample


def test_unknown_field_error():
    """Test that requesting unknown fields raises error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir, _, _, _ = _create_test_dataset(tmpdir)

        with pytest.raises(ValueError, match="Unknown fields"):
            MemmapDataSource(output_dir / "manifest.json", fields=["nonexistent"])


def test_string_fields():
    """Test loading string fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir, _, _, _ = _create_test_dataset(tmpdir, include_strings=True)

        source = MemmapDataSource(output_dir / "manifest.json")

        for i in range(10):
            sample = source[i]
            assert sample["code"] == f"code_{i}"


def test_get_slice():
    """Test efficient slice loading."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir, audio_data, params_data, labels = _create_test_dataset(tmpdir)

        source = MemmapDataSource(output_dir / "manifest.json")

        batch = source.get_slice(2, 5)
        assert batch["audio"].shape == (3, 2, 100)
        assert batch["params"].shape == (3, 5)
        assert batch["label"].shape == (3,)

        np.testing.assert_array_almost_equal(batch["audio"], audio_data[2:5], decimal=5)


def test_get_slice_with_fields():
    """Test slice loading with field filtering."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir, _, _, _ = _create_test_dataset(tmpdir)

        source = MemmapDataSource(output_dir / "manifest.json")

        batch = source.get_slice(0, 3, fields=["label"])
        assert "label" in batch
        assert "audio" not in batch


def test_index_bounds():
    """Test that out-of-bounds access raises IndexError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir, _, _, _ = _create_test_dataset(tmpdir, num_samples=5)

        source = MemmapDataSource(output_dir / "manifest.json")

        with pytest.raises(IndexError):
            _ = source[10]

        with pytest.raises(IndexError):
            _ = source[-10]


def test_transform_fn():
    """Test custom transform function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir, _, _, _ = _create_test_dataset(tmpdir)

        def double_audio(sample):
            sample["audio"] = sample["audio"] * 2
            return sample

        source = MemmapDataSource(
            output_dir / "manifest.json", transform_fn=double_audio
        )

        original_source = MemmapDataSource(output_dir / "manifest.json")

        sample = source[0]
        original_sample = original_source[0]

        np.testing.assert_array_almost_equal(
            sample["audio"], original_sample["audio"] * 2, decimal=5
        )


def test_get_metadata():
    """Test getting manifest metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        field_specs = [FieldSpec("x", np.float32, ())]
        metadata = {"sample_rate": 44100, "custom_key": "custom_value"}

        with MemmapWriter(
            output_dir, field_specs, expected_samples=1, metadata=metadata
        ) as writer:
            writer.write_sample({"x": np.array(1.0, dtype=np.float32)})

        source = MemmapDataSource(output_dir / "manifest.json")
        meta = source.get_metadata()

        assert meta["sample_rate"] == 44100
        assert meta["custom_key"] == "custom_value"
        assert "fields" not in meta  # Should be excluded


def test_get_field_info():
    """Test getting field information."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir, _, _, _ = _create_test_dataset(tmpdir)

        source = MemmapDataSource(output_dir / "manifest.json")

        info = source.get_field_info("audio")
        assert info["dtype"] == np.float32
        assert info["shape"] == (10, 2, 100)


def test_fields_property():
    """Test fields property."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir, _, _, _ = _create_test_dataset(tmpdir)

        source = MemmapDataSource(output_dir / "manifest.json")

        assert set(source.fields) == {"audio", "params", "label"}


def test_string_fields_property():
    """Test string_fields property."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir, _, _, _ = _create_test_dataset(tmpdir, include_strings=True)

        source = MemmapDataSource(output_dir / "manifest.json")

        assert source.string_fields == ["code"]


def test_grain_integration():
    """Test that MemmapDataSource works as a Grain RandomAccessDataSource."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir, _, _, _ = _create_test_dataset(tmpdir)

        source = MemmapDataSource(output_dir / "manifest.json")

        # Should support len()
        assert len(source) == 10

        # Should support random access
        sample = source[5]
        assert sample is not None

        # Should support iteration
        items = [source[i] for i in range(3)]
        assert len(items) == 3


def test_missing_manifest():
    """Test error when manifest doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError):
            MemmapDataSource(Path(tmpdir) / "nonexistent" / "manifest.json")


def test_missing_manifest_from_directory():
    """Test error when manifest doesn't exist via from_directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError, match="Manifest not found"):
            MemmapDataSource.from_directory(tmpdir)


# === Split tests ===


def test_split_train():
    """Test train split."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir, _, _, _ = _create_test_dataset(tmpdir, num_samples=100)

        train_source = MemmapDataSource(
            output_dir / "manifest.json",
            split="train",
            split_ratios=(0.8, 0.1, 0.1),
        )

        # 80% of 100 = 80 samples
        assert len(train_source) == 80


def test_split_val():
    """Test val split."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir, _, _, _ = _create_test_dataset(tmpdir, num_samples=100)

        val_source = MemmapDataSource(
            output_dir / "manifest.json",
            split="val",
            split_ratios=(0.8, 0.1, 0.1),
        )

        # 10% of 100 = 10 samples
        assert len(val_source) == 10


def test_split_test():
    """Test test split."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir, _, _, _ = _create_test_dataset(tmpdir, num_samples=100)

        test_source = MemmapDataSource(
            output_dir / "manifest.json",
            split="test",
            split_ratios=(0.8, 0.1, 0.1),
        )

        # 10% of 100 = 10 samples
        assert len(test_source) == 10


def test_split_none_uses_all_data():
    """Test that split=None uses all data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir, _, _, _ = _create_test_dataset(tmpdir, num_samples=100)

        source = MemmapDataSource(output_dir / "manifest.json", split=None)

        assert len(source) == 100


def test_split_no_overlap():
    """Test that train/val/test splits have no overlapping samples."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir, audio_data, _, labels = _create_test_dataset(
            tmpdir, num_samples=100
        )

        train_source = MemmapDataSource(
            output_dir / "manifest.json",
            split="train",
            split_ratios=(0.8, 0.1, 0.1),
        )
        val_source = MemmapDataSource(
            output_dir / "manifest.json",
            split="val",
            split_ratios=(0.8, 0.1, 0.1),
        )
        test_source = MemmapDataSource(
            output_dir / "manifest.json",
            split="test",
            split_ratios=(0.8, 0.1, 0.1),
        )

        # Collect labels from each split (labels are unique integers 0-99)
        train_labels = {train_source[i]["label"][0] for i in range(len(train_source))}
        val_labels = {val_source[i]["label"][0] for i in range(len(val_source))}
        test_labels = {test_source[i]["label"][0] for i in range(len(test_source))}

        # Check no overlap
        assert train_labels.isdisjoint(val_labels)
        assert train_labels.isdisjoint(test_labels)
        assert val_labels.isdisjoint(test_labels)

        # Check all samples covered
        all_labels = train_labels | val_labels | test_labels
        assert all_labels == set(range(100))


def test_split_reproducible():
    """Test that splits are reproducible with same seed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir, _, _, _ = _create_test_dataset(tmpdir, num_samples=100)

        source1 = MemmapDataSource(
            output_dir / "manifest.json",
            split="train",
            split_seed=42,
        )
        source2 = MemmapDataSource(
            output_dir / "manifest.json",
            split="train",
            split_seed=42,
        )

        # Same seed should give same samples
        for i in range(len(source1)):
            np.testing.assert_array_equal(
                source1[i]["label"], source2[i]["label"]
            )


def test_split_different_seeds():
    """Test that different seeds give different splits."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir, _, _, _ = _create_test_dataset(tmpdir, num_samples=100)

        source1 = MemmapDataSource(
            output_dir / "manifest.json",
            split="train",
            split_seed=42,
        )
        source2 = MemmapDataSource(
            output_dir / "manifest.json",
            split="train",
            split_seed=123,
        )

        # Different seeds should give different samples
        labels1 = [source1[i]["label"][0] for i in range(len(source1))]
        labels2 = [source2[i]["label"][0] for i in range(len(source2))]

        assert labels1 != labels2


def test_split_with_num_records():
    """Test combining split with num_records limit."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir, _, _, _ = _create_test_dataset(tmpdir, num_samples=100)

        source = MemmapDataSource(
            output_dir / "manifest.json",
            split="train",
            split_ratios=(0.8, 0.1, 0.1),
            num_records=50,
        )

        # 80% of 100 = 80, but limited to 50
        assert len(source) == 50


def test_split_invalid_ratios():
    """Test that invalid split_ratios raises error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir, _, _, _ = _create_test_dataset(tmpdir, num_samples=100)

        with pytest.raises(ValueError, match="split_ratios must sum to 1.0"):
            MemmapDataSource(
                output_dir / "manifest.json",
                split="train",
                split_ratios=(0.5, 0.5, 0.5),  # Sum = 1.5
            )


def test_split_wrong_number_of_ratios():
    """Test that wrong number of split_ratios raises error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir, _, _, _ = _create_test_dataset(tmpdir, num_samples=100)

        with pytest.raises(ValueError, match="exactly 3 values"):
            MemmapDataSource(
                output_dir / "manifest.json",
                split="train",
                split_ratios=(0.8, 0.2),  # Only 2 values
            )


def test_split_invalid_name():
    """Test that invalid split name raises error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir, _, _, _ = _create_test_dataset(tmpdir, num_samples=100)

        with pytest.raises(ValueError, match="Invalid split"):
            MemmapDataSource(
                output_dir / "manifest.json",
                split="invalid",
            )


def test_get_slice_not_supported_with_split():
    """Test that get_slice raises error when split is active."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir, _, _, _ = _create_test_dataset(tmpdir, num_samples=100)

        source = MemmapDataSource(
            output_dir / "manifest.json",
            split="train",
        )

        with pytest.raises(NotImplementedError, match="not supported when using splits"):
            source.get_slice(0, 5)


# === load_into_memory tests ===


def test_load_into_memory_basic():
    """Test basic loading with load_into_memory=True."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir, audio_data, params_data, labels = _create_test_dataset(tmpdir)

        source = MemmapDataSource(
            output_dir / "manifest.json",
            load_into_memory=True,
        )

        assert len(source) == 10

        sample = source[0]
        assert "audio" in sample
        assert "params" in sample
        assert "label" in sample

        # Check shapes (batch dimension added)
        assert sample["audio"].shape == (1, 2, 100)
        assert sample["params"].shape == (1, 5)
        assert sample["label"].shape == (1,)


def test_load_into_memory_data_matches_memmap():
    """Test that in-memory data matches memmap data exactly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir, audio_data, params_data, labels = _create_test_dataset(tmpdir)

        memmap_source = MemmapDataSource(output_dir / "manifest.json")
        memory_source = MemmapDataSource(
            output_dir / "manifest.json",
            load_into_memory=True,
        )

        for i in range(10):
            memmap_sample = memmap_source[i]
            memory_sample = memory_source[i]

            np.testing.assert_array_equal(
                memmap_sample["audio"], memory_sample["audio"]
            )
            np.testing.assert_array_equal(
                memmap_sample["params"], memory_sample["params"]
            )
            np.testing.assert_array_equal(
                memmap_sample["label"], memory_sample["label"]
            )


def test_load_into_memory_get_slice():
    """Test get_slice with load_into_memory=True."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir, audio_data, params_data, labels = _create_test_dataset(tmpdir)

        source = MemmapDataSource(
            output_dir / "manifest.json",
            load_into_memory=True,
        )

        batch = source.get_slice(2, 5)
        assert batch["audio"].shape == (3, 2, 100)
        assert batch["params"].shape == (3, 5)
        assert batch["label"].shape == (3,)

        np.testing.assert_array_almost_equal(batch["audio"], audio_data[2:5], decimal=5)


def test_load_into_memory_with_split():
    """Test load_into_memory combined with splits."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir, _, _, _ = _create_test_dataset(tmpdir, num_samples=100)

        train_memmap = MemmapDataSource(
            output_dir / "manifest.json",
            split="train",
            split_ratios=(0.8, 0.1, 0.1),
        )
        train_memory = MemmapDataSource(
            output_dir / "manifest.json",
            split="train",
            split_ratios=(0.8, 0.1, 0.1),
            load_into_memory=True,
        )

        assert len(train_memmap) == len(train_memory) == 80

        # Data should match
        for i in range(len(train_memmap)):
            np.testing.assert_array_equal(
                train_memmap[i]["label"], train_memory[i]["label"]
            )


def test_load_into_memory_from_directory():
    """Test load_into_memory with from_directory constructor."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir, _, _, _ = _create_test_dataset(tmpdir)

        source = MemmapDataSource.from_directory(output_dir, load_into_memory=True)
        assert len(source) == 10

        sample = source[0]
        assert "audio" in sample


# === audio_dtype tests ===


def test_audio_dtype_write_float16_read_float32():
    """Test writing audio as float16 and reading back as float32."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        num_samples = 5

        # Use _audio_data suffix so audio_dtype applies
        field_specs = [
            FieldSpec("dry_audio_data", np.float32, (2, 100)),
            FieldSpec("params", np.float32, (3,)),
        ]

        audio_data = np.random.randn(num_samples, 2, 100).astype(np.float32)
        params_data = np.random.randn(num_samples, 3).astype(np.float32)

        # Write with audio_dtype=float16 (casts _audio_data fields)
        with MemmapWriter(
            output_dir,
            field_specs=field_specs,
            expected_samples=num_samples,
            audio_dtype=np.float16,
        ) as writer:
            writer.write_batch(
                {"dry_audio_data": audio_data, "params": params_data}
            )

        # Verify on-disk dtype is float16 for audio, float32 for params
        import json

        with open(output_dir / "manifest.json") as f:
            manifest = json.load(f)
        assert manifest["fields"]["dry_audio_data"]["dtype"] == "float16"
        assert manifest["fields"]["params"]["dtype"] == "float32"
        assert manifest["audio_dtype"] == "float16"

        # Read back with audio_dtype=float32 (upcast)
        source = MemmapDataSource(
            output_dir / "manifest.json",
            audio_dtype=np.float32,
            reconstruct_audiotree=False,
        )

        for i in range(num_samples):
            sample = source[i]
            assert sample["dry_audio_data"].dtype == np.float32
            assert sample["params"].dtype == np.float32
            # Values close within float16 precision (atol ~1e-3)
            np.testing.assert_allclose(
                sample["dry_audio_data"][0],
                audio_data[i],
                atol=1e-3,
                rtol=1e-3,
            )
            # Params should be exact (not cast)
            np.testing.assert_array_almost_equal(
                sample["params"][0], params_data[i], decimal=5
            )


def test_audio_dtype_write_float16_read_float32_get_slice():
    """Test audio_dtype with get_slice."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        num_samples = 10

        field_specs = [
            FieldSpec("wet_audio_data", np.float32, (2, 50)),
        ]

        audio_data = np.random.randn(num_samples, 2, 50).astype(np.float32)

        with MemmapWriter(
            output_dir,
            field_specs=field_specs,
            expected_samples=num_samples,
            audio_dtype=np.float16,
        ) as writer:
            writer.write_batch({"wet_audio_data": audio_data})

        source = MemmapDataSource(
            output_dir / "manifest.json",
            audio_dtype=np.float32,
            reconstruct_audiotree=False,
        )

        batch = source.get_slice(2, 5)
        assert batch["wet_audio_data"].dtype == np.float32
        assert batch["wet_audio_data"].shape == (3, 2, 50)
        np.testing.assert_allclose(
            batch["wet_audio_data"], audio_data[2:5], atol=1e-3, rtol=1e-3
        )


def test_audio_dtype_none_preserves_original():
    """Test that audio_dtype=None leaves data unchanged."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        num_samples = 3

        field_specs = [
            FieldSpec("dry_audio_data", np.float32, (2, 50)),
        ]

        audio_data = np.random.randn(num_samples, 2, 50).astype(np.float32)

        # Write without audio_dtype
        with MemmapWriter(
            output_dir,
            field_specs=field_specs,
            expected_samples=num_samples,
        ) as writer:
            writer.write_batch({"dry_audio_data": audio_data})

        # Read without audio_dtype
        source = MemmapDataSource(
            output_dir / "manifest.json",
            reconstruct_audiotree=False,
        )

        sample = source[0]
        assert sample["dry_audio_data"].dtype == np.float32
        np.testing.assert_array_almost_equal(
            sample["dry_audio_data"][0], audio_data[0], decimal=5
        )
