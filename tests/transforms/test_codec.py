"""Tests for the codec-encoding transforms.

The central concern is picklability: grain spawns its worker processes, so a
transform has to pickle by reference. These transforms hold their codec as an
instance attribute of a module-level class, so both the transform and (given a
picklable codec) the codec travel through pickle intact.
"""

import pickle

import numpy as np
import pytest

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
        return np.tile(np.arange(frames, dtype=np.int32), (batch, self.n_codebooks, 1))

    def encode_to_latent(self, audio: AudioTree):
        batch = audio.waveform.shape[0]
        return np.broadcast_to(
            np.arange(8, dtype=np.float32)[None, :, None], (batch, 8, 5)
        ).copy()


def _audio_tree(batch: int = 3, num_samples: int = 100) -> AudioTree:
    waveform = np.random.randn(batch, 1, num_samples).astype(np.float32)
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


def test_reencode_after_clearing_codes_uses_the_new_codec():
    """The docs' re-encode recipe (clear codes, encode again) takes effect."""
    tree = _audio_tree()
    encoded_a = encode_with_codec(FakeCodec(n_codebooks=2)).map(tree)
    assert encoded_a.codes.shape[1] == 2

    encoded_b = encode_with_codec(FakeCodec(n_codebooks=3)).map(
        encoded_a.replace(codes=None)
    )
    assert encoded_b.codes.shape[1] == 3


def test_string_output_key_with_multiple_in_scope_leaves_raises():
    """The codec entry points inherit the string-output_key collision guard."""
    element = {"a": _audio_tree(), "b": _audio_tree()}
    transform = encode_with_codec(FakeCodec(), output_key="enc")
    with pytest.raises(ValueError, match="output_key"):
        transform.map(element)


def test_encode_latents_passes_through_existing_latents():
    tree = _audio_tree()
    encoded = encode_latents(FakeCodec()).map(tree)
    reencoded = encode_latents(FakeCodec(n_codebooks=4)).map(encoded)
    assert reencoded.latents is encoded.latents
