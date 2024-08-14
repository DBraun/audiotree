import numpy as np
import jax.numpy as jnp

from audiotree.resample import resample


def _resample(y: np.ndarray, sr: int, new_sr: int, output_path: str):

    y = jnp.array(y)

    y = jnp.expand_dims(y, 0)
    y = y[:, :1, :]
    # print('y shape: ', y.shape)

    y = resample(y, old_sr=sr, new_sr=new_sr)
    # print('y shape: ', y.shape)
    y = y.squeeze(0).T
    y = np.array(y)

    # todo: use the torch version of julius and confirm the outputs match.
    # (DBraun did this manually once but didn't automate it.)
    # from scipy.io import wavfile
    # wavfile.write(output_path, new_sr, y)


# def test_resample_001(filepath='assets/60013__qubodup__whoosh.flac', new_sr=96_000):
#     import librosa
#     y, sr = librosa.load(filepath, sr=None, mono=False, duration=10)
#
#     _resample(y, sr, new_sr, "tmp_test_resample_001.wav")


def test_resample_002(new_sr=96_000):

    sr = 44100

    y = np.zeros((1, sr * 10))

    _resample(y, sr, new_sr, "tmp_test_resample_002.wav")
