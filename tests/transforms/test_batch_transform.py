import numpy as np
import grain

import audiotree
from audiotree import AudioTree
import audiotree.transforms


def test_batch_transform_with_dataloader():
    """AudioTree.batch collates correctly as grain's batch_fn."""

    # Create a simple data source that yields AudioTree objects with filepath metadata
    audio_trees = []
    sample_rate = 44100
    duration = 0.1
    num_samples = int(sample_rate * duration)

    for i in range(10):
        # Create fake audio data
        waveform = np.random.randn(1, num_samples).astype(np.float32)

        # Create AudioTree with filepath metadata
        audio_tree = audiotree.AudioTree.create(
            waveform=waveform,
            sample_rate=sample_rate,
            filepaths=f"/fake/path/audio_{i:04d}.wav",
        )
        audio_trees.append(audio_tree)

    # Create a DataLoader with Batch transform
    dataloader = (
        grain.MapDataset.source(audio_trees)
        .to_iter_dataset()
        .batch(4, batch_fn=AudioTree.batch)
    )

    # Iterate and check batched filepaths
    batch_count = 0
    for audio_tree in dataloader:
        batch_count += 1

        # Check that we have an AudioTree object
        assert audio_tree.waveform.ndim == 3

        # Get the filepaths - should be a list of 4 (or less for last batch)
        filepaths = audio_tree.filepath

        print(f"Batch {batch_count}: {filepaths}")

        # Verify we have the expected number of filepaths
        if batch_count < 3:  # First two batches should have 4 items each
            assert len(filepaths) == 4
        else:  # Last batch has remaining 2 items
            assert len(filepaths) == 2

        # Verify the filepaths are correct
        for fp in filepaths:
            assert fp.startswith("/fake/path/audio_")
            assert fp.endswith(".wav")

    # We should have 3 batches total (10 items / 4 batch size = 2.5 -> 3 batches)
    assert batch_count == 3


def test_batch_with_iter_dataset():
    """Test that AudioTree.batch works with IterDataset.batch() API."""

    # Create a simple data source that yields AudioTree objects
    audio_trees = []
    sample_rate = 44100
    duration = 0.1
    num_samples = int(sample_rate * duration)

    for i in range(10):
        waveform = np.random.randn(1, num_samples).astype(np.float32)
        audio_tree = AudioTree.create(
            waveform=waveform,
            sample_rate=sample_rate,
            filepaths=f"/fake/path/audio_{i:04d}.wav",
        )
        audio_trees.append(audio_tree)

    # Create MapDataset and convert to IterDataset with batch_fn
    ds = grain.MapDataset.source(audio_trees)
    iter_ds = ds.to_iter_dataset().batch(4, batch_fn=AudioTree.batch)

    # Iterate and verify batching
    batch_count = 0
    for batch in iter_ds:
        batch_count += 1

        # Verify shape: should be (batch, channels, samples), not (batch, 1, channels, samples)
        assert batch.waveform.ndim == 3, (
            f"Expected 3D array, got {batch.waveform.ndim}D"
        )

        # Verify batch size
        if batch_count < 3:
            assert batch.waveform.shape[0] == 4
            assert len(batch.filepath) == 4
        else:
            assert batch.waveform.shape[0] == 2
            assert len(batch.filepath) == 2

    assert batch_count == 3


def test_batch_drop_remainder():
    """Test that AudioTree.batch works with drop_remainder=True."""

    audio_trees = []
    sample_rate = 44100
    num_samples = int(sample_rate * 0.1)

    for i in range(10):
        waveform = np.random.randn(1, num_samples).astype(np.float32)
        audio_tree = AudioTree.create(
            waveform=waveform,
            sample_rate=sample_rate,
        )
        audio_trees.append(audio_tree)

    ds = grain.MapDataset.source(audio_trees)
    iter_ds = ds.to_iter_dataset().batch(
        4, drop_remainder=True, batch_fn=AudioTree.batch
    )

    batches = list(iter_ds)

    # With drop_remainder=True, 10 items / 4 batch size = 2 complete batches
    assert len(batches) == 2
    for batch in batches:
        assert batch.waveform.shape[0] == 4


def test_batch_with_dict_elements():
    """Test that AudioTree.batch works when iterator yields dicts containing AudioTrees."""

    sample_rate = 44100
    num_samples = int(sample_rate * 0.1)

    # Create a data source that yields {"src": AudioTree, "tgt": AudioTree} dicts
    dict_items = []
    for i in range(10):
        src_data = np.random.randn(1, num_samples).astype(np.float32)
        tgt_data = np.random.randn(1, num_samples).astype(np.float32)

        src_tree = AudioTree.create(
            waveform=src_data,
            sample_rate=sample_rate,
            filepaths=f"/fake/path/src_{i:04d}.wav",
        )
        tgt_tree = AudioTree.create(
            waveform=tgt_data,
            sample_rate=sample_rate,
            filepaths=f"/fake/path/tgt_{i:04d}.wav",
        )
        dict_items.append({"src": src_tree, "tgt": tgt_tree})

    ds = grain.MapDataset.source(dict_items)
    iter_ds = ds.to_iter_dataset().batch(4, batch_fn=AudioTree.batch)

    batch_count = 0
    for batch in iter_ds:
        batch_count += 1

        # batch should be a dict with "src" and "tgt" keys
        assert isinstance(batch, dict), f"Expected dict, got {type(batch)}"
        assert "src" in batch
        assert "tgt" in batch

        # Each value should be a batched AudioTree
        assert batch["src"].waveform.ndim == 3, (
            f"Expected 3D array, got {batch['src'].waveform.ndim}D"
        )
        assert batch["tgt"].waveform.ndim == 3, (
            f"Expected 3D array, got {batch['tgt'].waveform.ndim}D"
        )

        # Verify batch sizes
        if batch_count < 3:
            assert batch["src"].waveform.shape[0] == 4
            assert batch["tgt"].waveform.shape[0] == 4
            assert len(batch["src"].filepath) == 4
            assert len(batch["tgt"].filepath) == 4
        else:
            assert batch["src"].waveform.shape[0] == 2
            assert batch["tgt"].waveform.shape[0] == 2
            assert len(batch["src"].filepath) == 2
            assert len(batch["tgt"].filepath) == 2

    assert batch_count == 3


if __name__ == "__main__":
    test_batch_transform_with_dataloader()
    test_batch_with_iter_dataset()
    test_batch_drop_remainder()
    test_batch_with_dict_elements()
