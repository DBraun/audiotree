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

    dataset = (
        AudioDataBalancedDataset(
            sources={"group1": ["group1"], "group2": ["group2"]},
            sample_rate=sample_rate,
            duration=0.01,
            weights={"group1": 1, "group2": 2},  # show group2 twice as much as group 1.
        )
        .map(audiotree.transforms.Batch(1))
    )
    i = 0
    for item in dataset:
        print(item.filepath)
        i += 1
        if i > 100:
            break
