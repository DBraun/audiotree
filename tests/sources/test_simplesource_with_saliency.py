"""Test AudioDataSimpleSource with SaliencyParams and grain.DataLoader with Batch transform."""

import tempfile
from pathlib import Path

import numpy as np
import soundfile
import grain

from audiotree import AudioTree
from audiotree.core import SaliencyParams
from audiotree.sources import AudioDataSimpleSource
from audiotree.transforms import Batch


def test_simplesource_saliency_with_grain_dataloader():
    """Test AudioDataSimpleSource with SaliencyParams using grain.DataLoader and Batch transform."""

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        sample_rate = 44_100
        duration = 2.0
        num_samples = int(sample_rate * duration)
        num_files = 8

        # Create audio files with louder sections in the middle
        for i in range(num_files):
            audio_data = np.random.randn(num_samples, 1).astype(np.float32) * 0.01
            if i % 2 == 0:
                start_idx = num_samples // 4
                end_idx = 3 * num_samples // 4
                audio_data[start_idx:end_idx] *= 10.0
            filepath = output_dir / f"audio_{i:04d}.wav"
            soundfile.write(filepath, audio_data, sample_rate)

        saliency_params = SaliencyParams(enabled=1, loudness_cutoff=None)
        source = AudioDataSimpleSource(
            sources={"test": [str(output_dir)]},
            num_records=num_files,
            sample_rate=sample_rate,
            duration=1.0,
            mono=True,
            saliency_params=saliency_params
        )

        assert len(source) == num_files

        # Load items and prepare for batching
        items = []
        for i in range(len(source)):
            item = source[i]
            # Remove metadata to avoid batching issues with variable-length arrays
            item = item.replace(metadata={})
            items.append(item)

        assert len(items) == num_files
        for item in items:
            assert isinstance(item, AudioTree)
            assert item.audio_data.shape == (1, 1, sample_rate)

        # Use grain.DataLoader with Batch transform
        batch_size = 4
        dataloader = grain.DataLoader(
            data_source=items,
            sampler=grain.samplers.SequentialSampler(len(items)),
            operations=[Batch(batch_size=batch_size)],
        )

        batch_count = 0
        total_items = 0

        for batch in dataloader:
            batch_count += 1
            assert isinstance(batch, AudioTree)
            assert batch.audio_data.ndim == 3
            current_batch_size = batch.audio_data.shape[0]

            if batch_count < (num_files // batch_size):
                assert current_batch_size == batch_size
            else:
                assert current_batch_size <= batch_size

            assert batch.sample_rate == sample_rate
            assert batch.audio_data.shape[1] == 1
            assert batch.audio_data.shape[2] == sample_rate

            total_items += current_batch_size
            print(f"Batch {batch_count}: {current_batch_size} items, shape={batch.audio_data.shape}")

        assert total_items == num_files
        assert batch_count == (num_files + batch_size - 1) // batch_size

        print(f"Successfully processed {total_items} items in {batch_count} batches")


if __name__ == "__main__":
    test_simplesource_saliency_with_grain_dataloader()
    print("Test passed!")
