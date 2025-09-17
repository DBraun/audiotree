import numpy as np
import grain

import audiotree
import audiotree.transforms


def test_batch_transform():
    """Test that Batch transform properly batches AudioTree objects with filepath metadata."""

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
    print("Test passed!")


if __name__ == "__main__":
    test_batch_transform()