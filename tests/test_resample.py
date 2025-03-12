from functools import partial

import numpy as np
import jax.numpy as jnp
import jax
import pytest

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
    # from scipy.io import wavfile
    # if output_path is not None:
    #     for i, audio in enumerate(y):
    #         wavfile.write(f"{output_path}_{str(i).zfill(3)}.wav", new_sr, audio.T)


# def test_resample_001():
#     filepaths = [
#         "60988__folktelemetry__crash-fast-14.wav",
#         "42v8.wav",
#         "60v8.wav",
#     ]
#
#     all_audio = []
#     import librosa
#     for filepath in filepaths:
#         y, old_sr = librosa.load(filepath, sr=44_100, mono=False, duration=4)
#         all_audio.append(jnp.array(y))
#     y = jnp.stack(all_audio, axis=0)
#
#     new_sr = 96_000
#
#     _resample(y, int(old_sr), new_sr, "tmp_test_resample_001")


@pytest.mark.parametrize("new_sr", [96_000])
def test_resample_002(new_sr: int):

    old_sr = 44_100

    B = 4
    C = 2

    y = np.zeros((B, C, old_sr * 10))

    _resample(y, old_sr=old_sr, new_sr=new_sr)
