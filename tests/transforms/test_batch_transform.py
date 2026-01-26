import numpy as np
import grain

import audiotree
from audiotree import AudioTree
import audiotree.transforms


def test_batch_transform_with_dataloader():
    """Test that Batch transform works with grain.DataLoader API."""

    # Create a simple data source that yields AudioTree objects with filepath metadata
    audio_trees = []
    sample_rate = 44100
    duration = 0.1
    num_samples = int(sample_rate * duration)

    for i in range(10):
        # Create fake audio data
        audio_data = np.random.randn(1, num_samples).astype(np.float32)

        # Create AudioTree with filepath metadata
        audio_tree = audiotree.AudioTree.create(
            audio_data=audio_data,
            sample_rate=sample_rate,
            filepaths=f"/fake/path/audio_{i:04d}.wav"
        )
        audio_trees.append(audio_tree)

    # Create a DataLoader with Batch transform
    dataloader = (
        grain.DataLoader(
            data_source=audio_trees,
            sampler=grain.samplers.SequentialSampler(len(audio_trees)),
            operations=[audiotree.transforms.Batch(4)],
        )
    )

    # Iterate and check batched filepaths
    batch_count = 0
    for audio_tree in dataloader:
        batch_count += 1

        # Check that we have an AudioTree object
        assert audio_tree.audio_data.ndim == 3

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


def test_batch_fn_with_iter_dataset():
    """Test that AudioTree.batch_fn works with IterDataset.batch() API."""

    # Create a simple data source that yields AudioTree objects
    audio_trees = []
    sample_rate = 44100
    duration = 0.1
    num_samples = int(sample_rate * duration)

    for i in range(10):
        audio_data = np.random.randn(1, num_samples).astype(np.float32)
        audio_tree = AudioTree.create(
            audio_data=audio_data,
            sample_rate=sample_rate,
            filepaths=f"/fake/path/audio_{i:04d}.wav"
        )
        audio_trees.append(audio_tree)

    # Create MapDataset and convert to IterDataset with batch_fn
    ds = grain.MapDataset.source(audio_trees)
    iter_ds = ds.to_iter_dataset().batch(4, batch_fn=AudioTree.batch_fn)

    # Iterate and verify batching
    batch_count = 0
    for batch in iter_ds:
        batch_count += 1

        # Verify shape: should be (batch, channels, samples), not (batch, 1, channels, samples)
        assert batch.audio_data.ndim == 3, f"Expected 3D array, got {batch.audio_data.ndim}D"

        # Verify batch size
        if batch_count < 3:
            assert batch.audio_data.shape[0] == 4
            assert len(batch.filepath) == 4
        else:
            assert batch.audio_data.shape[0] == 2
            assert len(batch.filepath) == 2

    assert batch_count == 3


def test_batch_fn_drop_remainder():
    """Test that AudioTree.batch_fn works with drop_remainder=True."""

    audio_trees = []
    sample_rate = 44100
    num_samples = int(sample_rate * 0.1)

    for i in range(10):
        audio_data = np.random.randn(1, num_samples).astype(np.float32)
        audio_tree = AudioTree.create(
            audio_data=audio_data,
            sample_rate=sample_rate,
        )
        audio_trees.append(audio_tree)

    ds = grain.MapDataset.source(audio_trees)
    iter_ds = ds.to_iter_dataset().batch(4, drop_remainder=True, batch_fn=AudioTree.batch_fn)

    batches = list(iter_ds)

    # With drop_remainder=True, 10 items / 4 batch size = 2 complete batches
    assert len(batches) == 2
    for batch in batches:
        assert batch.audio_data.shape[0] == 4


def test_batch_fn_with_dict_elements():
    """Test that AudioTree.batch_fn works when iterator yields dicts containing AudioTrees."""

    sample_rate = 44100
    num_samples = int(sample_rate * 0.1)

    # Create a data source that yields {"src": AudioTree, "tgt": AudioTree} dicts
    dict_items = []
    for i in range(10):
        src_data = np.random.randn(1, num_samples).astype(np.float32)
        tgt_data = np.random.randn(1, num_samples).astype(np.float32)

        src_tree = AudioTree.create(
            audio_data=src_data,
            sample_rate=sample_rate,
            filepaths=f"/fake/path/src_{i:04d}.wav",
        )
        tgt_tree = AudioTree.create(
            audio_data=tgt_data,
            sample_rate=sample_rate,
            filepaths=f"/fake/path/tgt_{i:04d}.wav",
        )
        dict_items.append({"src": src_tree, "tgt": tgt_tree})

    ds = grain.MapDataset.source(dict_items)
    iter_ds = ds.to_iter_dataset().batch(4, batch_fn=AudioTree.batch_fn)

    batch_count = 0
    for batch in iter_ds:
        batch_count += 1

        # batch should be a dict with "src" and "tgt" keys
        assert isinstance(batch, dict), f"Expected dict, got {type(batch)}"
        assert "src" in batch
        assert "tgt" in batch

        # Each value should be a batched AudioTree
        assert batch["src"].audio_data.ndim == 3, f"Expected 3D array, got {batch['src'].audio_data.ndim}D"
        assert batch["tgt"].audio_data.ndim == 3, f"Expected 3D array, got {batch['tgt'].audio_data.ndim}D"

        # Verify batch sizes
        if batch_count < 3:
            assert batch["src"].audio_data.shape[0] == 4
            assert batch["tgt"].audio_data.shape[0] == 4
            assert len(batch["src"].filepath) == 4
            assert len(batch["tgt"].filepath) == 4
        else:
            assert batch["src"].audio_data.shape[0] == 2
            assert batch["tgt"].audio_data.shape[0] == 2
            assert len(batch["src"].filepath) == 2
            assert len(batch["tgt"].filepath) == 2

    assert batch_count == 3


if __name__ == "__main__":
    test_batch_transform_with_dataloader()
    test_batch_fn_with_iter_dataset()
    test_batch_fn_drop_remainder()
    test_batch_fn_with_dict_elements()