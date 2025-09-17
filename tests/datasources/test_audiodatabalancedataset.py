import os.path

import numpy as np
import soundfile

import audiotree.transforms
from audiotree.datasources import AudioDataBalancedDataset


def test_audiodatabalancedataset():

    sample_rate = 44_100

    data = np.zeros((5000, 1))
    for g in range(1, 3):
        os.makedirs(f"group{g}", exist_ok=True)
        for i in range(5 + g * 5):
            outpath = f"group{g}/tmp_{str(i).zfill(4)}.wav"
            if not os.path.exists(outpath):
                soundfile.write(outpath, data, sample_rate)

    batch_size = 4

    dataset = AudioDataBalancedDataset(
        sources={"group1": ["group1"], "group2": ["group2"]},
        sample_rate=sample_rate,
        duration=0.01,
        weights={"group1": 1, "group2": 2},  # show group2 twice as much as group 1.
    ).batch(batch_size)  # Use grain's batch method directly

    i = 0
    for audio_tree in dataset:
        audio_tree = audio_tree.unbatch()
        print(audio_tree.filepath)
        assert audio_tree.audio_data.ndim == 3
        assert len(audio_tree.filepath) == batch_size
        assert audio_tree.audio_data.shape[0] == batch_size
        i += 1
        if i > 1:
            break
