"""Test scope functionality with Dict[str, AudioTree] inputs."""

import numpy as np
import pytest

from audiotree import AudioTree
from audiotree.transforms.functional import (
    volume_norm,
    volume_change,
    trim,
    invert_phase,
)


class TestScopeWithDict:
    """Test that scope works correctly with Dict[str, AudioTree]."""

    def test_scope_single_key(self):
        """Test applying transform only to 'src' key in dict."""
        # Create batch with multiple AudioTrees
        audio1 = AudioTree(
            np.random.randn(2, 1, 44100).astype(np.float32) * 0.1,
            sample_rate=44100,
        )
        audio2 = AudioTree(
            np.random.randn(2, 1, 44100).astype(np.float32) * 0.1,
            sample_rate=44100,
        )

        audio1 = audio1.replace_lufs()
        audio2 = audio2.replace_lufs()

        batch = {"src": audio1, "target": audio2}

        # Create transform with scope - only transform 'src'
        transform = volume_norm(
            min_db=-20,
            max_db=-15,
            scope={"src": {"scope": True}},
        )

        # Apply to batch
        rng = np.random.default_rng(42)
        result_batch = transform.random_map(batch, rng)

        # Verify only src was transformed
        assert not np.array_equal(result_batch["src"].lufs, audio1.lufs)
        assert np.array_equal(result_batch["target"].lufs, audio2.lufs)

    def test_scope_all_keys(self):
        """Test applying transform to all keys (no scope = transform everything)."""
        # Create batch
        audio1 = AudioTree(
            np.random.randn(2, 1, 44100).astype(np.float32) * 0.1,
            sample_rate=44100,
        )
        audio2 = AudioTree(
            np.random.randn(2, 1, 44100).astype(np.float32) * 0.1,
            sample_rate=44100,
        )

        audio1 = audio1.replace_lufs()
        audio2 = audio2.replace_lufs()

        batch = {"src": audio1, "target": audio2}

        # Create transform with no scope - should transform both
        transform = volume_norm(
            min_db=-20,
            max_db=-15,
        )

        # Apply to batch
        rng = np.random.default_rng(42)
        result_batch = transform.random_map(batch, rng)

        # Verify both were transformed
        assert not np.array_equal(result_batch["src"].lufs, audio1.lufs)
        assert not np.array_equal(result_batch["target"].lufs, audio2.lufs)

    def test_scope_multiple_keys(self):
        """Test applying transform to multiple specific keys."""
        # Create batch with 3 keys
        audio1 = AudioTree(
            np.random.randn(2, 1, 44100).astype(np.float32) * 0.1,
            sample_rate=44100,
        )
        audio2 = AudioTree(
            np.random.randn(2, 1, 44100).astype(np.float32) * 0.1,
            sample_rate=44100,
        )
        audio3 = AudioTree(
            np.random.randn(2, 1, 44100).astype(np.float32) * 0.1,
            sample_rate=44100,
        )

        audio1 = audio1.replace_lufs()
        audio2 = audio2.replace_lufs()
        audio3 = audio3.replace_lufs()

        batch = {"dry": audio1, "wet": audio2, "reference": audio3}

        # Create transform - only transform 'dry' and 'wet'
        transform = volume_norm(
            min_db=-20,
            max_db=-15,
            scope={
                "dry": {"scope": True},
                "wet": {"scope": True},
            },
        )

        # Apply to batch
        rng = np.random.default_rng(42)
        result_batch = transform.random_map(batch, rng)

        # Verify dry and wet transformed, reference not
        assert not np.array_equal(result_batch["dry"].lufs, audio1.lufs)
        assert not np.array_equal(result_batch["wet"].lufs, audio2.lufs)
        assert np.array_equal(result_batch["reference"].lufs, audio3.lufs)

    def test_scope_with_nested_dict(self):
        """Test scope with nested dictionary structure."""
        # Create nested batch
        audio1 = AudioTree(
            np.random.randn(2, 1, 44100).astype(np.float32) * 0.1,
            sample_rate=44100,
        )
        audio2 = AudioTree(
            np.random.randn(2, 1, 44100).astype(np.float32) * 0.1,
            sample_rate=44100,
        )
        audio3 = AudioTree(
            np.random.randn(2, 1, 44100).astype(np.float32) * 0.1,
            sample_rate=44100,
        )

        audio1 = audio1.replace_lufs()
        audio2 = audio2.replace_lufs()
        audio3 = audio3.replace_lufs()

        batch = {
            "input": {"dry": audio1, "wet": audio2},
            "target": audio3,
        }

        # Create transform - only transform input.dry
        transform = volume_norm(
            min_db=-20,
            max_db=-15,
            scope={"input": {"dry": {"scope": True}}},
        )

        # Apply to batch
        rng = np.random.default_rng(42)
        result_batch = transform.random_map(batch, rng)

        # Verify only input.dry transformed
        assert not np.array_equal(result_batch["input"]["dry"].lufs, audio1.lufs)
        assert np.array_equal(result_batch["input"]["wet"].lufs, audio2.lufs)
        assert np.array_equal(result_batch["target"].lufs, audio3.lufs)


class TestScopeWithOutputKey:
    """Test scope combined with output_key."""

    def test_scope_with_output_key(self):
        """Test applying transform with scope and output_key."""
        # Create batch
        audio1 = AudioTree(
            np.random.randn(2, 1, 44100).astype(np.float32) * 0.1,
            sample_rate=44100,
        )
        audio2 = AudioTree(
            np.random.randn(2, 1, 44100).astype(np.float32) * 0.1,
            sample_rate=44100,
        )

        audio1 = audio1.replace_lufs()
        audio2 = audio2.replace_lufs()

        batch = {"src": audio1, "target": audio2}

        # Transform only 'src' and output to 'src_modified'
        transform = volume_norm(
            min_db=-20,
            max_db=-15,
            scope={"src": {"scope": True}},
            output_key="modified",
        )

        # Apply to batch
        rng = np.random.default_rng(42)
        result_batch = transform.random_map(batch, rng)

        # Should have: src (original), target (original), modified (new)
        assert "src" in result_batch
        assert "target" in result_batch
        assert "modified" in result_batch

        # Original should be unchanged
        assert np.array_equal(result_batch["src"].lufs, audio1.lufs)
        assert np.array_equal(result_batch["target"].lufs, audio2.lufs)

        # Modified should be transformed
        assert not np.array_equal(result_batch["modified"].lufs, audio1.lufs)


class TestScopeWithMapTransforms:
    """Test scope with deterministic map transforms."""

    def test_trim_with_scope(self):
        """Test trim transform with scope."""
        # Create batch
        audio1 = AudioTree(
            np.random.randn(2, 1, 44100 * 5).astype(np.float32) * 0.1,
            sample_rate=44100,
        )
        audio2 = AudioTree(
            np.random.randn(2, 1, 44100 * 5).astype(np.float32) * 0.1,
            sample_rate=44100,
        )

        batch = {"dry": audio1, "wet": audio2}

        # Trim only 'dry'
        transform = trim(
            length=3.0,
            scope={"dry": {"scope": True}},
        )

        # Apply to batch
        result_batch = transform.map(batch)

        # Verify only dry was trimmed
        assert result_batch["dry"].waveform.shape[-1] == int(3.0 * 44100)
        assert result_batch["wet"].waveform.shape[-1] == int(5.0 * 44100)


class TestCommonTrainingPipelinePattern:
    """Test the common training pipeline pattern with Dict[str, AudioTree]."""

    def test_dry_wet_processing(self):
        """Test processing dry and wet signals differently."""
        # Create dry and wet audio
        dry_audio = AudioTree(
            np.random.randn(4, 1, 44100 * 3).astype(np.float32) * 0.1,
            sample_rate=44100,
        )
        wet_audio = AudioTree(
            np.random.randn(4, 1, 44100 * 3).astype(np.float32) * 0.1,
            sample_rate=44100,
        )

        dry_audio = dry_audio.replace_lufs()
        wet_audio = wet_audio.replace_lufs()

        batch = {"dry": dry_audio, "wet": wet_audio}

        # Process dry and wet differently
        # 1. Normalize both
        transform1 = volume_norm(min_db=-20, max_db=-15)
        batch = transform1.random_map(batch, np.random.default_rng(42))

        # 2. Add volume change only to wet
        transform2 = volume_change(
            min_db=-6,
            max_db=6,
            scope={"wet": {"scope": True}},
        )
        batch = transform2.random_map(batch, np.random.default_rng(43))

        # 3. Invert phase on dry only (50% prob)
        transform3 = invert_phase(
            prob=0.5,
            scope={"dry": {"scope": True}},
        )
        batch = transform3.random_map(batch, np.random.default_rng(44))

        # Verify structure preserved
        assert "dry" in batch
        assert "wet" in batch
        assert isinstance(batch["dry"], AudioTree)
        assert isinstance(batch["wet"], AudioTree)

    def test_input_target_pattern(self):
        """Test common input/target pattern."""
        # Create input and target
        input_audio = AudioTree(
            np.random.randn(4, 1, 44100 * 3).astype(np.float32) * 0.1,
            sample_rate=44100,
        )
        target_audio = AudioTree(
            np.random.randn(4, 1, 44100 * 3).astype(np.float32) * 0.1,
            sample_rate=44100,
        )

        input_audio = input_audio.replace_lufs()
        target_audio = target_audio.replace_lufs()

        batch = {"input": input_audio, "target": target_audio}

        # Apply shared transforms to both
        transform1 = volume_norm(min_db=-20, max_db=-15)
        batch = transform1.random_map(batch, np.random.default_rng(42))

        # Apply additional augmentation only to input
        transform2 = volume_change(
            min_db=-12,
            max_db=12,
            prob=0.9,
            scope={"input": {"scope": True}},
        )
        batch = transform2.random_map(batch, np.random.default_rng(43))

        # Both should be normalized
        assert batch["input"].lufs is not None
        assert batch["target"].lufs is not None

    def test_multi_key_batch(self):
        """Test batch with multiple audio keys (dry, wet, reference)."""
        # Create multiple audio sources
        dry = AudioTree(
            np.random.randn(4, 1, 44100 * 2).astype(np.float32) * 0.1,
            sample_rate=44100,
        )
        wet = AudioTree(
            np.random.randn(4, 1, 44100 * 2).astype(np.float32) * 0.1,
            sample_rate=44100,
        )
        reference = AudioTree(
            np.random.randn(4, 1, 44100 * 2).astype(np.float32) * 0.1,
            sample_rate=44100,
        )

        dry = dry.replace_lufs()
        wet = wet.replace_lufs()
        reference = reference.replace_lufs()

        batch = {"dry": dry, "wet": wet, "reference": reference}

        original_ref_loudness = reference.lufs.copy()

        # Normalize dry and wet, but not reference
        transform = volume_norm(
            min_db=-20,
            max_db=-15,
            scope={
                "dry": {"scope": True},
                "wet": {"scope": True},
            },
        )

        rng = np.random.default_rng(42)
        result = transform.random_map(batch, rng)

        # Verify dry and wet were transformed
        assert not np.array_equal(result["dry"].lufs, dry.lufs)
        assert not np.array_equal(result["wet"].lufs, wet.lufs)

        # Verify reference was not transformed
        assert np.array_equal(result["reference"].lufs, original_ref_loudness)

    def test_split_seed_false_keeps_the_pair_aligned(self):
        """`split_seed=False` normalizes dry and wet to the *same* loudness.

        The NumPy backend shared one stateful Generator across leaves, so each
        leaf advanced the stream and drew its own target -- decorrelating the
        pair the flag is meant to lock together.
        """
        waveform = np.random.randn(4, 1, 44100).astype(np.float32) * 0.1
        batch = {
            "dry": AudioTree(waveform, 44100).replace_lufs(),
            "wet": AudioTree(waveform, 44100).replace_lufs(),
        }

        transform = volume_norm(min_db=-30, max_db=-10, split_seed=False)
        result = transform.random_map(batch, np.random.default_rng(0))

        np.testing.assert_allclose(result["dry"].lufs, result["wet"].lufs)
        np.testing.assert_allclose(result["dry"].waveform, result["wet"].waveform)


# === scope spellings ===


def _scoped_pair(scope):
    """Apply +6 dB under `scope` to a {dry, wet} element; report what changed."""
    element = {
        "dry": AudioTree(np.ones((1, 1, 4), dtype=np.float32), 16000),
        "wet": AudioTree(np.ones((1, 1, 4), dtype=np.float32), 16000),
    }
    out = volume_change(min_db=6, max_db=6, scope=scope).random_map(
        element, np.random.default_rng(0)
    )
    return {k: not np.allclose(out[k].waveform, 1.0) for k in ("dry", "wet")}


def test_scope_list_form_selects_one_leaf():
    """The list form names the subtrees to transform."""
    assert _scoped_pair(["wet"]) == {"dry": False, "wet": True}


def test_scope_dict_sentinel_form_still_works():
    """The nested-dict form with the 'scope' sentinel is unchanged."""
    assert _scoped_pair({"wet": {"scope": True}}) == {"dry": False, "wet": True}


def test_scope_bare_bool_shorthand_raises():
    """`scope={'wet': True}` used to silently transform *everything*.

    A scope entry is matched one level above where it sits, so a bool directly
    under a top-level key has an empty match prefix and selects every leaf --
    in a dry/wet pipeline that silently augments the target signal.
    """
    with pytest.raises(ValueError, match="selects every leaf"):
        _scoped_pair({"wet": True})


def test_scope_list_form_nested_path():
    """A dotted path (or tuple) selects a nested subtree."""
    element = {
        "d": {
            "e": AudioTree(np.ones((1, 1, 4), dtype=np.float32), 16000),
            "f": AudioTree(np.ones((1, 1, 4), dtype=np.float32), 16000),
        }
    }
    for scope in (["d.e"], [("d", "e")]):
        out = volume_change(min_db=6, max_db=6, scope=scope).random_map(
            element, np.random.default_rng(0)
        )
        changed = {k: not np.allclose(out["d"][k].waveform, 1.0) for k in ("e", "f")}
        assert changed == {"e": True, "f": False}, scope


def test_scope_rejects_bad_types():
    with pytest.raises(TypeError, match="scope must be a list of paths"):
        volume_change(scope="wet")
    with pytest.raises(TypeError, match="scope path must be a string"):
        volume_change(scope=[123])


def test_scope_dict_with_parameter_override_raises():
    """Parameters inside a scope entry never worked as per-key overrides.

    The docs used to teach ``{'dry': {'scope': True, 'min_db': -30}}`` as a
    per-key parameter override, but a non-scope key in a scope dict was read as
    an extra scope marker, not as an override -- the example ran and silently
    used the default parameters everywhere. That spelling now raises at
    construction; per-key parameters mean one transform instance per key.
    """
    with pytest.raises(ValueError, match="overrides.*not supported"):
        volume_norm(scope={"dry": {"scope": True, "min_db": -30}})
    # Even without a 'scope' marker alongside it.
    with pytest.raises(ValueError, match="overrides.*not supported"):
        volume_norm(scope={"dry": {"min_db": -30}})


def test_scope_dict_nested_exclusion_still_works():
    """The legitimate dict-form keys -- nested paths ending in 'scope' -- stay valid."""
    element = {
        "d": {
            "e": AudioTree(np.ones((1, 1, 4), dtype=np.float32), 16000),
            "f": AudioTree(np.ones((1, 1, 4), dtype=np.float32), 16000),
        }
    }
    scope = {"d": {"scope": True, "f": {"scope": False}}}
    out = volume_change(min_db=6, max_db=6, scope=scope).random_map(
        element, np.random.default_rng(0)
    )
    changed = {k: not np.allclose(out["d"][k].waveform, 1.0) for k in ("e", "f")}
    assert changed == {"e": True, "f": False}


# === output_key + scope regressions (both backends) ===


def _backend(name):
    """Return (invert_phase transform factory, make_rng) for a backend."""
    if name == "numpy":
        from audiotree.transforms import invert_phase as ip

        return ip, lambda: np.random.default_rng(0)
    import jax

    from audiotree.transforms.jax import invert_phase as ip

    return ip, lambda: jax.random.PRNGKey(0)


def _const(value):
    return AudioTree.create(np.full((1, 1, 64), value, np.float32), 16000)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
@pytest.mark.parametrize("order", [("dry", "wet"), ("wet", "dry")])
def test_output_key_with_out_of_scope_first_key(backend, order):
    """B1: an out-of-scope alphabetically-first key must not crash `_post_process`.

    jax rebuilds dicts in sorted-key order, so with ``scope=['wet']`` the
    sorted-first key 'dry' is out of scope; the empty-dict guard in the
    `is_leaf` helper must keep it from being mis-descended (IndexError).
    Insertion order must not matter.
    """
    invert_phase, make_rng = _backend(backend)

    element = {k: _const(1.0) for k in order}
    result = invert_phase(scope=["wet"], output_key="wet_aug").random_map(
        element, make_rng()
    )

    assert set(result) == {"dry", "wet", "wet_aug"}
    # Only the in-scope 'wet' subtree is transformed, written to 'wet_aug'.
    assert np.allclose(np.asarray(result["wet_aug"].waveform), -1.0)
    # Originals are untouched.
    assert np.allclose(np.asarray(result["dry"].waveform), 1.0)
    assert np.allclose(np.asarray(result["wet"].waveform), 1.0)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_output_key_control_in_scope_first_key(backend):
    """B1 control: transforming the sorted-first key still works."""
    invert_phase, make_rng = _backend(backend)

    result = invert_phase(scope=["dry"], output_key="dry_aug").random_map(
        {"dry": _const(1.0), "wet": _const(1.0)}, make_rng()
    )

    assert set(result) == {"dry", "wet", "dry_aug"}
    assert np.allclose(np.asarray(result["dry_aug"].waveform), -1.0)
    assert np.allclose(np.asarray(result["dry"].waveform), 1.0)
    assert np.allclose(np.asarray(result["wet"].waveform), 1.0)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_output_key_overwriting_existing_key_writes_transformed(backend):
    """M1: `output_key` colliding with an existing key must write the transform.

    Previously the merge let the OLD untransformed value win on collision, so
    the user's explicitly requested output was silently discarded.
    """
    invert_phase, make_rng = _backend(backend)

    result = invert_phase(scope=["dry"], output_key="wet").random_map(
        {"dry": _const(1.0), "wet": _const(0.5)}, make_rng()
    )

    # The transformed 'dry' (-1.0) wins for 'wet', not the original 0.5.
    assert np.allclose(np.asarray(result["wet"].waveform), -1.0)
    # The transformed key's source is left untouched.
    assert np.allclose(np.asarray(result["dry"].waveform), 1.0)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_string_output_key_with_multiple_in_scope_leaves_raises(backend):
    """m8: a string `output_key` used to keep only the last transformed leaf.

    ``rename_node`` mapped every in-scope key to the same output name, so the
    rebuilt dict silently discarded all but one result (whichever came last in
    jax's sorted-key order). Now it raises and points at a callable
    ``output_key``.
    """
    invert_phase, make_rng = _backend(backend)
    element = {"dry": _const(1.0), "wet": _const(0.5)}

    with pytest.raises(ValueError, match="callable output_key"):
        invert_phase(output_key="aug").random_map(element, make_rng())


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_callable_output_key_names_each_in_scope_leaf(backend):
    """m8 control: a callable that gives distinct names keeps every result."""
    invert_phase, make_rng = _backend(backend)
    element = {"dry": _const(1.0), "wet": _const(0.5)}

    result = invert_phase(output_key=lambda path: path[-1] + "_aug").random_map(
        element, make_rng()
    )

    assert set(result) == {"dry", "wet", "dry_aug", "wet_aug"}
    assert np.allclose(np.asarray(result["dry_aug"].waveform), -1.0)
    assert np.allclose(np.asarray(result["wet_aug"].waveform), -0.5)
