"""Tests for function-based transforms with decorators."""

import tempfile
from pathlib import Path

import argbind
import jax
import numpy as np
import pytest
import soundfile as sf

from audiotree import AudioTree
from audiotree.sources import create_audio_dataset
from audiotree.transforms.codec import (
    AudioCodec,
    LatentAudioCodec,
    encode_latents,
    encode_with_codec,
)
from audiotree.transforms.functional import (
    choose,
    volume_norm,
    volume_change,
    trim,
    invert_phase,
    mono,
    stereo,
    swap_stereo,
    roll,
)
from audiotree.transforms.jax.functional import roll as roll_jax


def _create_test_audio_file(tmpdir, filename, duration=3.0, sample_rate=44100):
    """Helper to create a test audio file."""
    audio_path = Path(tmpdir) / filename
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    sf.write(str(audio_path), audio, sample_rate)
    return str(audio_path)


class TestRandomTransformDecorator:
    """Test @random_transform decorator."""

    def test_volume_norm_basic(self):
        """Test volume_norm function transform."""
        audio_tree = AudioTree(
            np.random.randn(4, 1, 44100).astype(np.float32) * 0.1,
            sample_rate=44100,
        )
        audio_tree = audio_tree.replace_lufs()

        # Create transform with flat parameters
        transform = volume_norm(min_db=-20, max_db=-15)

        # Apply transform
        rng = np.random.default_rng(42)
        result = transform.random_map(audio_tree, rng)

        # Verify loudness is in range
        assert result.lufs is not None
        assert np.all(result.lufs >= -20 - 1)
        assert np.all(result.lufs <= -15 + 1)

    def test_volume_norm_with_prob(self):
        """Test volume_norm with probability."""
        audio_tree = AudioTree(
            np.random.randn(4, 1, 44100).astype(np.float32) * 0.1,
            sample_rate=44100,
        )
        audio_tree = audio_tree.replace_lufs()

        # Create transform with prob
        transform = volume_norm(min_db=-20, max_db=-15, prob=0.5)

        # Apply multiple times - some should be unchanged
        results = []
        for i in range(10):
            rng = np.random.default_rng(i)
            result = transform.random_map(audio_tree, rng)
            results.append(result)

        # Check that some were transformed and some weren't
        # (This is probabilistic but with 10 samples should be reliable)
        loudness_values = [float(r.lufs[0]) for r in results]
        assert len(set(loudness_values)) > 1  # Should have different values

    def test_volume_change_basic(self):
        """Test volume_change function transform."""
        audio_tree = AudioTree(
            np.random.randn(4, 1, 44100).astype(np.float32) * 0.1,
            sample_rate=44100,
        )

        # Create transform
        transform = volume_change(min_db=-6, max_db=6)

        # Apply transform
        rng = np.random.default_rng(42)
        result = transform.random_map(audio_tree, rng)

        # Verify audio was changed
        assert not np.array_equal(result.waveform, audio_tree.waveform)

    def test_invert_phase_basic(self):
        """Test invert_phase function transform."""
        audio_tree = AudioTree(
            np.random.randn(4, 1, 44100).astype(np.float32) * 0.1,
            sample_rate=44100,
        )

        # Create transform
        transform = invert_phase()

        # Apply transform
        rng = np.random.default_rng(42)
        result = transform.random_map(audio_tree, rng)

        # Verify phase was inverted
        np.testing.assert_array_almost_equal(
            result.waveform,
            -audio_tree.waveform,
        )


class TestMapTransformDecorator:
    """Test @map_transform decorator."""

    def test_trim_basic(self):
        """Test trim function transform."""
        audio_tree = AudioTree(
            np.random.randn(4, 1, 44100 * 5).astype(np.float32) * 0.1,
            sample_rate=44100,
        )

        # Create transform
        transform = trim(length=3.0)

        # Apply transform
        result = transform.map(audio_tree)

        # Verify length
        expected_length = int(3.0 * 44100)
        assert result.waveform.shape[-1] == expected_length

    def test_trim_shorter_audio(self):
        """Test trim with audio shorter than target pads audio."""
        audio_tree = AudioTree(
            np.random.randn(4, 1, 44100 * 2).astype(np.float32) * 0.1,
            sample_rate=44100,
        )

        # Create transform with longer target (default mode="wrap")
        transform = trim(length=5.0)

        # Apply transform
        result = transform.map(audio_tree)

        # Audio should be padded to target length
        assert result.waveform.shape[-1] == int(5.0 * 44100)

    def test_mono_conversion(self):
        """Test mono function transform."""
        audio_tree = AudioTree(
            np.random.randn(4, 2, 44100).astype(np.float32) * 0.1,
            sample_rate=44100,
        )

        # Create transform
        transform = mono()

        # Apply transform
        result = transform.map(audio_tree)

        # Verify mono
        assert result.waveform.shape[1] == 1

    def test_stereo_conversion(self):
        """Test stereo function transform."""
        audio_tree = AudioTree(
            np.random.randn(4, 1, 44100).astype(np.float32) * 0.1,
            sample_rate=44100,
        )

        # Create transform
        transform = stereo()

        # Apply transform
        result = transform.map(audio_tree)

        # Verify stereo
        assert result.waveform.shape[1] == 2


class TestDatasetChaining:
    """Test transforms work with dataset chaining."""

    def test_chain_with_dataset(self):
        """Test chaining function transforms with dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_test_audio_file(tmpdir, "test.wav", duration=5.0)

            # Create dataset
            ds = create_audio_dataset(
                sources=tmpdir,
                shuffle=False,
                num_epochs=1,
                sample_rate=44100,
                duration=5.0,
            ).slice(slice(0, 5))
            ds = ds.seed(42)

            # Chain transforms
            ds = ds.random_map(volume_norm(min_db=-20, max_db=-15))
            ds = ds.map(trim(length=3.0))

            # Load item
            item = ds[0]

            # Verify transforms were applied. trim runs last and changes the
            # audio length, so it invalidates the loudness set by volume_norm.
            assert item.lufs is None
            assert item.waveform.shape[-1] == int(3.0 * 44100)

    def test_multiple_random_transforms(self):
        """Test chaining multiple random transforms."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_test_audio_file(tmpdir, "test.wav", duration=3.0)

            # Create dataset
            ds = create_audio_dataset(
                sources=tmpdir,
                shuffle=False,
                num_epochs=1,
                sample_rate=44100,
                duration=3.0,
            ).slice(slice(0, 5))
            ds = ds.seed(42)

            # Chain multiple random transforms
            ds = ds.random_map(volume_norm(min_db=-20, max_db=-15))
            ds = ds.random_map(volume_change(min_db=-6, max_db=6))
            ds = ds.random_map(invert_phase(prob=0.5), seed=44)

            # Load item
            item = ds[0]

            # Verify all transforms were applied
            assert item.lufs is not None


class TestArgBindIntegration:
    """Test integration with argbind."""

    def test_bind_volume_norm(self):
        """Test binding volume_norm with argbind."""
        volume_norm_bound = argbind.bind(volume_norm)

        # Create audio
        audio_tree = AudioTree(
            np.random.randn(4, 1, 44100).astype(np.float32) * 0.1,
            sample_rate=44100,
        )
        audio_tree = audio_tree.replace_lufs()

        # Set args
        args = {
            "volume_norm.min_db": -30,
            "volume_norm.max_db": -10,
            "volume_norm.prob": 0.9,
        }

        # Apply with scope
        with argbind.scope(args):
            transform = volume_norm_bound()
            rng = np.random.default_rng(42)
            result = transform.random_map(audio_tree, rng)

        # Verify parameters were used
        assert result.lufs is not None

    def test_bind_trim(self):
        """Test binding trim with argbind."""
        trim_bound = argbind.bind(trim)

        # Create audio
        audio_tree = AudioTree(
            np.random.randn(4, 1, 44100 * 5).astype(np.float32) * 0.1,
            sample_rate=44100,
        )

        # Set args
        args = {
            "trim.length": 2.0,
        }

        # Apply with scope
        with argbind.scope(args):
            transform = trim_bound()
            result = transform.map(audio_tree)

        # Verify length
        expected_length = int(2.0 * 44100)
        assert result.waveform.shape[-1] == expected_length

    def test_bind_with_dataset(self):
        """Test argbind with dataset chaining."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_test_audio_file(tmpdir, "test.wav", duration=5.0)

            # Bind transforms
            volume_norm_bound = argbind.bind(volume_norm)
            trim_bound = argbind.bind(trim)

            # Create dataset
            ds = create_audio_dataset(
                sources=tmpdir,
                shuffle=False,
                num_epochs=1,
                sample_rate=44100,
                duration=5.0,
            ).slice(slice(0, 5))

            # Set args
            args = {
                "volume_norm.min_db": -25,
                "volume_norm.max_db": -15,
                "trim.length": 3.0,
            }

            # Chain with scope
            with argbind.scope(args):
                ds = ds.random_map(volume_norm_bound(), seed=42)
                ds = ds.map(trim_bound())

            # Load item
            item = ds[0]

            # Verify transforms applied with correct parameters. trim runs last
            # and changes the audio length, invalidating volume_norm's loudness.
            assert item.lufs is None
            assert item.waveform.shape[-1] == int(3.0 * 44100)


class TestDefaultParameters:
    """Test that default parameters work correctly."""

    def test_volume_norm_defaults(self):
        """Test volume_norm with default parameters."""
        audio_tree = AudioTree(
            np.random.randn(4, 1, 44100).astype(np.float32) * 0.1,
            sample_rate=44100,
        )
        audio_tree = audio_tree.replace_lufs()

        # Create transform with defaults
        transform = volume_norm()

        # Apply transform
        rng = np.random.default_rng(42)
        result = transform.random_map(audio_tree, rng)

        # Should work but not change much (min=max=0)
        assert result.lufs is not None

    def test_trim_default(self):
        """Test trim with default parameters."""
        audio_tree = AudioTree(
            np.random.randn(4, 1, 44100 * 3).astype(np.float32) * 0.1,
            sample_rate=44100,
        )

        # Create transform with defaults (length=1.0)
        transform = trim()

        # Apply transform
        result = transform.map(audio_tree)

        # Verify default length was used
        expected_length = int(1.0 * 44100)
        assert result.waveform.shape[-1] == expected_length


def _tree(channels: int = 2, samples: int = 16, sample_rate: int = 16000) -> AudioTree:
    """A small deterministic AudioTree whose channel ``c`` is filled with ``c + 1``."""
    values = np.arange(1, channels + 1, dtype=np.float32)
    waveform = np.tile(values[None, :, None], (1, 1, samples))
    return AudioTree(waveform=waveform, sample_rate=sample_rate)


class TestSwapStereo:
    """``swap_stereo`` is defined for mono (no-op) and stereo, and nothing else."""

    def test_swaps_stereo(self):
        result = swap_stereo().random_map(_tree(2), np.random.default_rng(0))
        np.testing.assert_array_equal(
            np.asarray(result.waveform), _tree(2).waveform[:, ::-1]
        )

    def test_mono_passes_through(self):
        """One channel has only the identity permutation, so this is not an error."""
        result = swap_stereo().random_map(_tree(1), np.random.default_rng(0))
        np.testing.assert_array_equal(np.asarray(result.waveform), _tree(1).waveform)

    def test_rejects_more_than_two_channels(self):
        """Reversing the channel order of a 5.1 mix is not a stereo swap."""
        with pytest.raises(ValueError, match="4 channels"):
            swap_stereo().random_map(_tree(4), np.random.default_rng(0))


class TestChoose:
    """``choose`` picks ``c`` of its transforms and applies them."""

    # x10 and x0.1 gains: deterministic (min_db == max_db) and order-independent.
    def _louder(self):
        return volume_change(min_db=20.0, max_db=20.0)

    def _quieter(self):
        return volume_change(min_db=-20.0, max_db=-20.0)

    def test_applies_every_transform_when_c_equals_all(self):
        audio_tree = _tree()
        transform = choose(self._louder(), invert_phase(), c=2)
        result = transform.random_map(audio_tree, np.random.default_rng(0))
        np.testing.assert_allclose(
            np.asarray(result.waveform), -10.0 * audio_tree.waveform, rtol=1e-5
        )

    def test_weights_select_a_transform(self):
        """A weight of 1.0 pins the choice, so every seed gives the same result."""
        audio_tree = _tree()
        transform = choose(self._louder(), self._quieter(), c=1, weights=[1.0, 0.0])
        for seed in range(5):
            result = transform.random_map(audio_tree, np.random.default_rng(seed))
            np.testing.assert_allclose(
                np.asarray(result.waveform), 10.0 * audio_tree.waveform, rtol=1e-5
            )

    def test_mixes_map_and_random_map_transforms(self):
        """A ``Map`` goes through ``.map()``, a ``RandomMap`` through ``.random_map()``."""
        audio_tree = _tree(samples=16000)
        transform = choose(trim(length=0.5), invert_phase(), c=2)
        result = transform.random_map(audio_tree, np.random.default_rng(0))
        assert result.waveform.shape[-1] == 8000
        np.testing.assert_allclose(
            np.asarray(result.waveform), -audio_tree.waveform[..., :8000], rtol=1e-5
        )

    def test_prob_zero_returns_the_element_untouched(self):
        audio_tree = _tree()
        transform = choose(invert_phase(), c=1, prob=0.0)
        result = transform.random_map(audio_tree, np.random.default_rng(0))
        assert result is audio_tree

    def test_same_seed_gives_the_same_choice(self):
        audio_tree = _tree()
        transform = choose(self._louder(), self._quieter(), c=1)
        first = transform.random_map(audio_tree, np.random.default_rng(7))
        second = transform.random_map(audio_tree, np.random.default_rng(7))
        np.testing.assert_array_equal(
            np.asarray(first.waveform), np.asarray(second.waveform)
        )

    def test_rejects_a_non_transform(self):
        with pytest.raises(TypeError, match="not a grain Map/RandomMap"):
            choose(invert_phase(), lambda x: x, c=1)

    def test_rejects_choosing_more_transforms_than_it_has(self):
        with pytest.raises(ValueError, match="c=3"):
            choose(invert_phase(), c=3)

    def test_rejects_mismatched_weights(self):
        with pytest.raises(ValueError, match="one weight per transform"):
            choose(invert_phase(), swap_stereo(), c=1, weights=[1.0])

    def test_rejects_out_of_range_prob(self):
        with pytest.raises(ValueError, match=r"prob=1.5"):
            choose(invert_phase(), c=1, prob=1.5)


class _EncodeOnlyCodec:
    """A codec that can only produce codes — it has no latent representation."""

    def encode(self, audio_tree: AudioTree):
        batch = audio_tree.waveform.shape[0]
        return np.zeros((batch, 2, 4), dtype=np.int32), None


class _LatentOnlyCodec:
    """A codec that can only produce latents."""

    def encode_to_latent(self, audio_tree: AudioTree):
        return np.zeros((audio_tree.waveform.shape[0], 8), dtype=np.float32)


class TestCodecProtocols:
    """The codec protocols are runtime-checkable and independent of each other."""

    def test_encode_only_codec_satisfies_audio_codec(self):
        codec = _EncodeOnlyCodec()
        assert isinstance(codec, AudioCodec)
        assert not isinstance(codec, LatentAudioCodec)

    def test_latent_only_codec_satisfies_latent_audio_codec(self):
        codec = _LatentOnlyCodec()
        assert isinstance(codec, LatentAudioCodec)
        assert not isinstance(codec, AudioCodec)

    def test_a_plain_object_satisfies_neither(self):
        assert not isinstance(object(), AudioCodec)
        assert not isinstance(object(), LatentAudioCodec)

    def test_encode_only_codec_works_with_its_transform(self):
        """An encode-only codec is usable, not just expressible."""
        result = encode_with_codec(_EncodeOnlyCodec()).map(_tree())
        assert result.codes.shape == (1, 2, 4)


class TestCodecTransformScopeAndOutputKey:
    """``encode_with_codec``/``encode_latents`` take ``scope`` and ``output_key``."""

    def test_scope_limits_which_leaves_are_encoded(self):
        element = {"dry": _tree(), "wet": _tree()}
        result = encode_with_codec(_EncodeOnlyCodec(), scope=["wet"]).map(element)
        assert result["wet"].codes is not None
        assert result["dry"].codes is None

    def test_output_key_writes_a_new_leaf(self):
        element = {"dry": _tree()}
        result = encode_latents(
            _LatentOnlyCodec(), scope=["dry"], output_key="dry_latents"
        ).map(element)
        assert result["dry"].latents is None
        assert result["dry_latents"].latents.shape == (1, 8)

    def test_defaults_still_encode_everything(self):
        element = {"dry": _tree(), "wet": _tree()}
        result = encode_latents(_LatentOnlyCodec()).map(element)
        assert result["dry"].latents is not None
        assert result["wet"].latents is not None


class TestRoll:
    """``roll`` validates its range identically on both backends.

    ``_roll_np`` draws with ``rng.integers`` and ``_roll_jax`` with
    ``jax.random.randint``; the latter silently clamps an inverted (min > max)
    range to a constant roll instead of erroring, so both backends share one
    range check that raises the same ``ValueError``.
    """

    # (build_transform, make_rng) for each backend; ``roll`` returns an argbind
    # transform applied with ``random_map``, mirroring the other tests here.
    _BACKENDS = [
        pytest.param(roll, lambda: np.random.default_rng(0), id="numpy"),
        pytest.param(roll_jax, lambda: jax.random.key(0), id="jax"),
    ]

    @pytest.mark.parametrize("build,make_rng", _BACKENDS)
    def test_inverted_range_raises(self, build, make_rng):
        with pytest.raises(ValueError, match="min_seconds <= max_seconds"):
            build(min_seconds=0.5, max_seconds=0.1).random_map(_tree(), make_rng())

    @pytest.mark.parametrize("build,make_rng", _BACKENDS)
    def test_valid_range_works(self, build, make_rng):
        tree = _tree()
        result = build(min_seconds=-0.1, max_seconds=0.1).random_map(tree, make_rng())
        assert np.asarray(result.waveform).shape == tree.waveform.shape
