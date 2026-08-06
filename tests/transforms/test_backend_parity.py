"""Every transform that exists on both backends, run on identical inputs.

``audiotree.transforms`` (NumPy/grain) and ``audiotree.transforms.jax`` publish
the same names and are supposed to be the same transforms. Nothing was checking
that, and the two drifted: ``shift_phase`` rotated the phase of each *channel*
on JAX and of each *item* on NumPy, which is invisible to a mono test and
shreds a stereo image. Every case here therefore runs at least one stereo
shape.

Random transforms cannot be compared draw-for-draw -- ``np.random.Generator``
and ``jax.random`` are different streams -- so parity is pinned two ways:

* **Degenerate parameters.** ``min == max``, ``amount=0.0``, ``prob=0.0``: the
  draw still happens but no longer influences the output, so the two backends
  must agree numerically. This is where the interesting machinery (the STFT
  round trip, the loudness meter, the roll indexing) actually gets compared.
* **Structural invariants.** Properties that must hold for *any* draw, asserted
  on both backends: which fields get invalidated, whether a shift is shared
  across channels, whether a corruption is not.

Where the backends legitimately differ -- the FIR (NumPy) versus IIR (JAX)
loudness meter, and librosa's versus Julius's resampler -- the tolerance is
asserted and justified rather than the transform being skipped.
"""

import jax
from jax import numpy as jnp
import numpy as np
import pytest

from audiotree import AudioTree
import audiotree.transforms as np_transforms
import audiotree.transforms.jax as jax_transforms

SAMPLE_RATE = 44100

#: Half a second, deliberately not a multiple of any STFT hop used below, so
#: the phase transforms exercise their tail handling.
LENGTH = 22050

#: (batch, channels). Stereo is covered at every batch size: the divergence
#: this module exists to catch was channel-shaped.
SHAPES = [(1, 1), (1, 2), (2, 2), (5, 1), (5, 2)]

#: Transforms exported by both backends that this module does not compare, and
#: why. Checked against the real export lists by
#: ``test_every_shared_transform_is_covered``.
UNCOVERED = {
    # Not transforms: the decorators used to build them.
    "map_transform",
    "random_transform",
    # Codec transforms need a pretrained neural codec checkpoint; they are the
    # same object imported into both namespaces (``transforms.codec``), so
    # there is no backend pair to compare. Covered by test_functional.py.
    # (Both codec protocols are also Protocols, not callables at all.)
    "AudioCodec",
    "LatentAudioCodec",
    "encode_with_codec",
    "encode_latents",
}


def _waveform(batch: int, channels: int, length: int = LENGTH, seed: int = 0):
    """A deterministic ``(batch, channels, length)`` waveform in [-0.5, 0.5]."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((batch, channels, length)) * 0.15).astype(np.float32)


def _pair(waveform: np.ndarray, with_lufs: bool = False):
    """The same audio as a NumPy-backed and a JAX-backed ``AudioTree``."""
    np_tree = AudioTree(waveform=waveform, sample_rate=SAMPLE_RATE)
    jax_tree = AudioTree(waveform=jnp.asarray(waveform), sample_rate=SAMPLE_RATE)
    if with_lufs:
        np_tree, jax_tree = np_tree.replace_lufs(), jax_tree.replace_lufs()
    return np_tree, jax_tree


def _apply(transform, audio_tree: AudioTree, seed: int, backend: str) -> AudioTree:
    """Run a grain ``Map`` or ``RandomMap`` transform with the backend's RNG."""
    if hasattr(transform, "random_map"):
        rng = np.random.default_rng(seed) if backend == "np" else jax.random.key(seed)
        return transform.random_map(audio_tree, rng)
    return transform.map(audio_tree)


def _run_both(name: str, kwargs: dict, waveform: np.ndarray, seed: int = 0, **pair_kw):
    """Apply ``name(**kwargs)`` from each backend to the same input."""
    np_out = _apply(
        getattr(np_transforms, name)(**kwargs),
        _pair(waveform, **pair_kw)[0],
        seed,
        "np",
    )
    jax_out = _apply(
        getattr(jax_transforms, name)(**kwargs),
        _pair(waveform, **pair_kw)[1],
        seed,
        "jax",
    )
    return np_out, jax_out


def _assert_same_structure(np_out: AudioTree, jax_out: AudioTree) -> None:
    """The backends must invalidate (or keep) the same cached fields."""
    assert np_out.sample_rate == jax_out.sample_rate
    for field in ("lufs", "lufs_windows", "codes", "latents"):
        assert (getattr(np_out, field) is None) == (getattr(jax_out, field) is None), (
            f"backends disagree on whether {field} survives the transform"
        )
    assert sorted(np_out.metadata) == sorted(jax_out.metadata)


# =============================================================================
# Exact parity: transforms whose output does not depend on the random draw
# =============================================================================

#: ``(id, transform name, kwargs, minimum channel count)``. Every entry is
#: either a map transform or a random transform whose parameters are pinned so
#: the draw cannot change the result.
EXACT_CASES = [
    ("identity", "identity", {}, 1),
    ("mono", "mono", {}, 1),
    ("stereo", "stereo", {}, 1),
    ("rescale_audio", "rescale_audio", {}, 1),
    ("peak_norm", "peak_norm", {}, 1),
    ("invert_phase", "invert_phase", {}, 1),
    ("swap_stereo", "swap_stereo", {}, 1),
    ("trim-shorter", "trim", {"length": 0.25}, 1),
    ("trim-same", "trim", {"length": LENGTH / SAMPLE_RATE}, 1),
    ("trim-pad-wrap", "trim", {"length": 0.75, "mode": "wrap"}, 1),
    ("trim-pad-constant", "trim", {"length": 0.75, "mode": "constant"}, 1),
    ("volume_change", "volume_change", {"min_db": 6.0, "max_db": 6.0}, 1),
    ("volume_change-zero", "volume_change", {"min_db": 0.0, "max_db": 0.0}, 1),
    ("roll-wrap", "roll", {"min_seconds": 0.1, "max_seconds": 0.1}, 1),
    (
        "roll-constant",
        "roll",
        {"min_seconds": -0.1, "max_seconds": -0.1, "mode": "constant"},
        1,
    ),
    ("roll-zero", "roll", {"min_seconds": 0.0, "max_seconds": 0.0}, 1),
    ("corrupt_phase-none", "corrupt_phase", {"amount": 0.0}, 1),
    ("corrupt_phase-hop", "corrupt_phase", {"amount": 0.0, "hop_factor": 0.25}, 1),
    ("shift_phase-none", "shift_phase", {"amount": 0.0}, 1),
    # prob=0.0 must be a no-op on both backends, draw or no draw.
    ("prob-zero", "corrupt_phase", {"amount": 1.0, "prob": 0.0}, 1),
    ("prob-zero-shift", "shift_phase", {"amount": 1.0, "prob": 0.0}, 1),
]


@pytest.mark.parametrize("batch,channels", SHAPES)
@pytest.mark.parametrize(
    "name,kwargs",
    [(name, kwargs) for _, name, kwargs, _ in EXACT_CASES],
    ids=[case_id for case_id, _, _, _ in EXACT_CASES],
)
def test_backends_agree(name: str, kwargs: dict, batch: int, channels: int):
    """The two backends produce the same audio from the same input.

    ``atol=1e-5`` is float32 slack for the transforms that do real arithmetic
    (the STFT round trip accumulates ~1e-7 on a 0.45-peak signal); the purely
    structural ones agree bit for bit.
    """
    waveform = _waveform(batch, channels)
    np_out, jax_out = _run_both(name, kwargs, waveform)

    np.testing.assert_allclose(
        np.asarray(jax_out.waveform), np.asarray(np_out.waveform), atol=1e-5
    )
    _assert_same_structure(np_out, jax_out)


@pytest.mark.parametrize("batch,channels", SHAPES)
def test_resample_backends_agree_on_band_limited_audio(batch: int, channels: int):
    """librosa (soxr) and the Julius port agree on content both keep.

    The two use different anti-alias filters, so they only have to agree on
    signal well inside the new passband. A 0.3-amplitude 440 Hz tone resampled
    44.1 -> 16 kHz matches to 8.5e-5 absolute, so ``atol=4e-4`` leaves a factor
    of ~5 of headroom. On *broadband* input the filters disagree far more --
    white noise resamples to only ~24 dB SNR between the backends, because
    almost all of the difference is in the 8 kHz transition band that one
    filter rolls off sooner than the other. That is a property of the two
    filter designs, not a bug, so this test uses a tone.
    """
    time = np.arange(LENGTH) / SAMPLE_RATE
    tone = (0.3 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
    waveform = np.broadcast_to(tone, (batch, channels, LENGTH)).copy()

    np_out, jax_out = _run_both("resample", {"sample_rate": 16000}, waveform)

    assert np_out.sample_rate == jax_out.sample_rate == 16000
    np.testing.assert_allclose(
        np.asarray(jax_out.waveform), np.asarray(np_out.waveform), atol=4e-4
    )


@pytest.mark.parametrize("batch,channels", SHAPES)
def test_volume_norm_backends_agree_up_to_the_loudness_meter(batch: int, channels: int):
    """The only difference is the meter: a pure gain, within 0.05 dB of 1.

    ``volume_norm`` targets a LUFS value, so the two backends' gains differ by
    exactly their loudness estimates. NumPy measures with ``pyloudnorm``'s FIR
    K-weighting and JAX with an IIR one; on the signals here the estimates
    differ by at most 0.031 dB. Asserting a pure ratio rather than a raw
    tolerance keeps the test sharp: a real divergence in *shape* fails even
    though the amplitude tolerance is loose.
    """
    waveform = _waveform(batch, channels)
    np_out, jax_out = _run_both(
        "volume_norm", {"min_db": -16.0, "max_db": -16.0}, waveform, with_lufs=True
    )

    np_wave = np.asarray(np_out.waveform)
    jax_wave = np.asarray(jax_out.waveform)
    gains = jax_wave / np_wave

    # Per item, the ratio is one constant (the meters disagree, the transform
    # does not) ...
    per_item = gains.reshape(batch, -1)
    np.testing.assert_allclose(
        per_item, np.broadcast_to(per_item[:, :1], per_item.shape), rtol=1e-4
    )
    # ... and that constant is a fraction of a dB.
    decibels = 20 * np.log10(np.abs(per_item[:, 0]))
    assert np.abs(decibels).max() < 0.05, f"gain differs by {decibels} dB"

    _assert_same_structure(np_out, jax_out)


# =============================================================================
# Structural parity: invariants that hold for any draw
# =============================================================================


@pytest.mark.parametrize("seed", range(4))
def test_shift_phase_is_one_rotation_per_item_on_both_backends(seed: int):
    """Identical channels stay identical; distinct items stay distinct.

    This is the invariant the JAX backend broke by drawing per channel. It is
    checked here as well as in ``test_helpers`` because it is the parity claim:
    both backends must mean the same thing by "shift the phase".
    """
    channel = _waveform(2, 1, seed=seed)
    waveform = np.tile(channel, (1, 2, 1))

    np_out, jax_out = _run_both("shift_phase", {"amount": 1.0}, waveform, seed=seed)

    for backend, out in (("np", np_out), ("jax", jax_out)):
        audio = np.asarray(out.waveform)
        np.testing.assert_allclose(
            audio[:, 0], audio[:, 1], atol=1e-5, err_msg=f"{backend} split the channels"
        )
        # A per-item draw means the two items really did get different angles;
        # if they had not, the invariant above would be trivially satisfied.
        assert not np.allclose(audio[0], audio[1], atol=1e-3)


@pytest.mark.parametrize("seed", range(4))
def test_corrupt_phase_is_per_channel_on_both_backends(seed: int):
    """Unlike ``shift_phase``, corruption is drawn per channel on purpose."""
    channel = _waveform(1, 1, seed=seed)
    waveform = np.tile(channel, (1, 2, 1))

    np_out, jax_out = _run_both("corrupt_phase", {"amount": 1.0}, waveform, seed=seed)

    for backend, out in (("np", np_out), ("jax", jax_out)):
        audio = np.asarray(out.waveform)
        assert not np.allclose(audio[0, 0], audio[0, 1], atol=1e-3), (
            f"{backend} corrupted both channels identically"
        )


@pytest.mark.parametrize("channels", [1, 2])
def test_shift_phase_preserves_energy_on_both_backends(channels: int):
    """One rotation of the whole spectrum is an all-pass: energy is unchanged.

    The 2% tolerance absorbs the window's edge effects -- the first and last
    frames are not fully overlap-added, so the rotation does not cancel there.

    ``corrupt_phase`` is deliberately not asserted this way: rotating each
    frequency independently makes each windowed frame's content spill outside
    its own window support, so the overlap-add cancels part of it and the
    output loses ~40% of the input energy on both backends. That is inherent
    to STFT phase corruption, not a backend difference.
    """
    waveform = _waveform(3, channels)
    np_out, jax_out = _run_both("shift_phase", {"amount": 1.0}, waveform)

    reference = (waveform**2).sum(axis=(-2, -1))
    for backend, out in (("np", np_out), ("jax", jax_out)):
        energy = (np.asarray(out.waveform) ** 2).sum(axis=(-2, -1))
        np.testing.assert_allclose(
            energy, reference, rtol=0.02, err_msg=f"{backend} changed the energy"
        )


@pytest.mark.parametrize("channels", [3, 6])
def test_swap_stereo_rejects_surround_on_both_backends(channels: int):
    """Neither backend may quietly reverse the channel order of a 5.1 mix."""
    waveform = _waveform(1, channels, length=64)
    np_tree, jax_tree = _pair(waveform)

    with pytest.raises(ValueError, match=f"{channels} channels"):
        np_transforms.swap_stereo().random_map(np_tree, np.random.default_rng(0))
    with pytest.raises(ValueError, match=f"{channels} channels"):
        jax_transforms.swap_stereo().random_map(jax_tree, jax.random.key(0))


@pytest.mark.parametrize("name", ["resample", "mono", "stereo"])
def test_degenerate_shapes_agree(name: str):
    """A single sample and a single item still round-trip identically."""
    waveform = _waveform(1, 1, length=1) if name != "resample" else _waveform(1, 1)
    kwargs = {"sample_rate": SAMPLE_RATE} if name == "resample" else {}

    np_out, jax_out = _run_both(name, kwargs, waveform)

    np.testing.assert_allclose(
        np.asarray(jax_out.waveform), np.asarray(np_out.waveform), atol=1e-5
    )


# =============================================================================
# The list above has to keep up with the modules
# =============================================================================


def test_every_shared_transform_is_covered():
    """Adding a transform to both backends forces a parity case for it.

    Without this, the next ``shift_phase`` ships uncompared.
    """
    shared = set(np_transforms.__all__) & set(jax_transforms.__all__)
    covered = {name for _, name, _, _ in EXACT_CASES} | {
        "resample",
        "volume_norm",
        "swap_stereo",
    }

    assert shared - covered - UNCOVERED == set(), (
        "transform(s) exported by both backends with no parity case; add one to "
        "EXACT_CASES, or to UNCOVERED with a reason"
    )
    assert UNCOVERED <= shared, "UNCOVERED names a transform that is not shared"
    # ``choose`` is NumPy-only by design (it branches in Python), so it must
    # not appear on the JAX side.
    assert "choose" in np_transforms.__all__
    assert "choose" not in jax_transforms.__all__
