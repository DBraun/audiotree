"""Tests for the codec-encoding transforms.

The central concern is picklability: grain spawns its worker processes, so a
transform has to pickle by reference. These transforms hold their codec as an
instance attribute of a module-level class, so both the transform and (given a
picklable codec) the codec travel through pickle intact.
"""

import pickle

import numpy as np

from audiotree import AudioTree
from audiotree.transforms import volume_norm
from audiotree.transforms.codec import encode_latents, encode_with_codec


class FakeCodec:
    """A minimal, module-level (hence picklable) codec.

    Satisfies both :class:`~audiotree.transforms.codec.AudioCodec` and
    :class:`~audiotree.transforms.codec.LatentAudioCodec`. Outputs are a
    deterministic function of the waveform so a round-tripped transform can be
    checked to behave identically.
    """

    def __init__(self, n_codebooks: int = 2):
        self.n_codebooks = n_codebooks

    def encode(self, audio: AudioTree):
        batch, frames = audio.waveform.shape[0], 5
        codes = np.tile(np.arange(frames, dtype=np.int32), (batch, self.n_codebooks, 1))
        return codes, None

    def encode_to_latent(self, audio: AudioTree):
        batch = audio.waveform.shape[0]
        return np.broadcast_to(
            np.arange(8, dtype=np.float32)[None, :, None], (batch, 8, 5)
        ).copy()


def _audio_tree(batch: int = 3, num_samples: int = 100) -> AudioTree:
    waveform = np.random.randn(batch, num_samples).astype(np.float32)
    return AudioTree.create(waveform=waveform, sample_rate=16000)


def test_encode_with_codec_pickle_roundtrip():
    transform = encode_with_codec(FakeCodec())
    rehydrated = pickle.loads(pickle.dumps(transform))

    audio_tree = _audio_tree()
    np.testing.assert_array_equal(
        transform.map(audio_tree).codes, rehydrated.map(audio_tree).codes
    )


def test_encode_latents_pickle_roundtrip():
    transform = encode_latents(FakeCodec())
    rehydrated = pickle.loads(pickle.dumps(transform))

    audio_tree = _audio_tree()
    np.testing.assert_array_equal(
        transform.map(audio_tree).latents, rehydrated.map(audio_tree).latents
    )


def test_volume_norm_pickle_roundtrip_control():
    """A normal module-level transform still pickles (regression guard)."""
    transform = volume_norm(min_db=-20, max_db=-15, prob=0.5)
    pickle.loads(pickle.dumps(transform))
