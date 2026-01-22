from functools import partial
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import jax
import librosa
import pytest
from scipy.io import wavfile

from audiotree.resample import resample


def _resample(
    y: np.ndarray,
    old_sr: int,
    new_sr: int,
    output_path: str = None,
    do_jit: bool = True,
):

    y = jnp.array(y)
    # print('y shape: ', y.shape)

    if do_jit:

        @partial(
            jax.jit,
            static_argnames=(
                "old_sr",
                "new_sr",
            ),
        )
        def resample_fn(x, old_sr, new_sr):
            return resample(x, old_sr, new_sr)

    else:
        resample_fn = resample

    y = resample_fn(y, old_sr=old_sr, new_sr=new_sr)
    # print('y shape: ', y.shape)
    y = np.array(y)

    # todo: use the torch version of julius and confirm the outputs match.
    # (DBraun did this manually once but didn't automate it.)

    if output_path is not None:
        for i, audio in enumerate(y):
            wavfile.write(f"{output_path}_{str(i).zfill(3)}.wav", new_sr, audio.T)


def test_resample_001():
    # Use test assets - stereo file loaded 3 times for batch testing
    assets_dir = Path(__file__).parent / "assets"
    filepath = str(assets_dir / "musdb18hq" / "train" / "A Classic Education - NightOwl" / "mixture.wav")

    all_audio = []

    # Load the same file 3 times with different offsets to create a batch
    for offset in [0.0, 4.0, 8.0]:
        y, old_sr = librosa.load(filepath, sr=44_100, mono=False, duration=4, offset=offset)
        all_audio.append(jnp.array(y))
    y = jnp.stack(all_audio, axis=0)

    new_sr = 96_000

    # Write to test_outputs directory
    test_outputs_dir = Path(__file__).parent.parent / "test_outputs"
    test_outputs_dir.mkdir(exist_ok=True)
    output_path = test_outputs_dir / "test_resample_001"
    _resample(y, int(old_sr), new_sr, str(output_path))


@pytest.mark.parametrize("new_sr", [96_000])
def test_resample_002(new_sr: int):

    old_sr = 44_100

    B = 4
    C = 2

    y = np.zeros((B, C, old_sr * 10))

    _resample(y, old_sr=old_sr, new_sr=new_sr)
