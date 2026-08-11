"""Golden tests pinning the *data* a fixed seed produces.

Every other seeding test asserts a property -- "two seeds differ", "one seed
repeats" -- which stays true even if shuffling, excerpt selection, or seed
derivation is silently rewritten. These tests instead pin the exact
``(basename, offset)`` sequence a known corpus and a known seed yield, so any
change to :func:`~audiotree.sources.create_audio_dataset`,
:func:`~audiotree.sources.create_balanced_audio_dataset`, grain's shuffle, or
the ``_derive_seed_pair`` / ``_derive_group_seed`` helpers shows up here as a
diff rather than as a run that quietly trains on different audio.

**If one of these fails after an intentional seeding change**, the fix is to
regenerate the golden list -- the failure message prints a paste-ready literal
-- and to say so in the changelog, because every user's data order changed
too. If you did *not* intend to change seeding, the diff is the bug.
"""

import os
import tempfile
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pytest
import soundfile as sf

from audiotree.sources import create_audio_dataset, create_balanced_audio_dataset
from audiotree.sources.core import _derive_group_seed, _derive_seed_pair

#: Corpus parameters. The offsets below are a function of these, so changing
#: any of them invalidates every golden list in this file.
SAMPLE_RATE = 16_000
EXCERPT_DURATION = 0.5
#: Per-file lengths in seconds. Deliberately unequal: a uniformly drawn offset
#: is scaled by ``file_duration - excerpt_duration``, so unequal lengths make
#: the offset depend on *which* file an index landed on, not just on the index.
FILE_DURATIONS = (1.0, 1.25, 1.5, 1.75, 2.0, 2.25)

#: Number of decimals the golden offsets are rounded to. The draw is
#: ``lower + (upper - lower) * u`` in IEEE doubles from numpy's PCG64, whose
#: stream numpy documents as reproducible, so this is exact rather than
#: approximate -- the rounding only keeps the literals readable.
OFFSET_DECIMALS = 6

#: One item's identity: which file it came from and where in that file. Object
#: identity and waveform contents are useless for this; these two are what a
#: seeding change moves.
Item = Tuple[str, float]


def _write_corpus(directory: Path, prefix: str = "clip") -> str:
    """Write the fixed corpus the golden lists were generated from.

    Only the file *lengths* matter to the goldens (a random excerpt offset is
    drawn from ``[0, length - EXCERPT_DURATION]``), but the samples are written
    deterministically anyway so a future loudness-based golden could be added
    without regenerating these.

    Args:
        directory: Directory to write into; created if missing.
        prefix: Filename stem prefix, so two groups can have distinct names.

    Returns:
        The directory as a string, ready to pass as ``sources``.
    """
    directory.mkdir(parents=True, exist_ok=True)
    for i, duration in enumerate(FILE_DURATIONS):
        t = np.arange(int(SAMPLE_RATE * duration), dtype=np.float64) / SAMPLE_RATE
        waveform = (np.sin(2 * np.pi * (220 + 40 * i) * t) * 0.5).astype(np.float32)
        sf.write(str(directory / f"{prefix}_{i}.wav"), waveform, SAMPLE_RATE)
    return str(directory)


def _identity(item) -> Item:
    """Reduce one dataset item to ``(basename, offset)``."""
    return (
        os.path.basename(item.filepath[0]),
        round(float(item.offset[0]), OFFSET_DECIMALS),
    )


def _grouped_identity(item) -> Tuple[str, str, float]:
    """Like :func:`_identity`, prefixed with the item's source group."""
    basename, offset = _identity(item)
    return (item.source[0], basename, offset)


def _collect(dataset, grouped: bool = False) -> List[tuple]:
    """Iterate ``dataset`` and return each item's identity tuple.

    Args:
        dataset: A finite :class:`grain.MapDataset`.
        grouped: Whether to include the source group name, which only the
            balanced datasets set.
    """
    identity = _grouped_identity if grouped else _identity
    return [identity(item) for item in dataset.to_iter_dataset()]


def _format_golden(sequence: Sequence[tuple]) -> str:
    """Render a sequence as a paste-ready Python literal."""
    lines = "\n".join(f"    {item!r}," for item in sequence)
    return f"[\n{lines}\n]"


def _assert_golden(actual: Sequence[tuple], expected: Sequence[tuple], name: str):
    """Compare against a golden list, failing with a regeneration recipe.

    Args:
        actual: What the dataset produced now.
        expected: The literal committed in this file.
        name: The golden list's variable name, so the message says what to edit.
    """
    if list(actual) == list(expected):
        return

    first_diff = next(
        (i for i, (a, e) in enumerate(zip(actual, expected)) if a != e),
        min(len(actual), len(expected)),
    )
    pytest.fail(
        f"Seeded dataset no longer produces the golden sequence {name}.\n"
        f"First difference at index {first_diff}: "
        f"expected {expected[first_diff] if first_diff < len(expected) else '<end>'!r}, "
        f"got {actual[first_diff] if first_diff < len(actual) else '<end>'!r} "
        f"(lengths: expected {len(expected)}, got {len(actual)}).\n"
        "\n"
        "This is a golden test: it pins the exact data a fixed seed yields, so "
        "it fails for one of three reasons.\n"
        "  1. A bug changed shuffling, excerpt selection, or seed derivation. "
        "The diff is the bug -- fix the code, not the list.\n"
        "  2. grain changed its shuffle or its per-index RNG in an upgrade. "
        "Worth confirming against the grain changelog, because it silently "
        "reorders every existing user's data too.\n"
        f"  3. You changed seeding on purpose. Then regenerate {name} by "
        "pasting the block below, and note in the changelog that every user's "
        "data order changes with this release.\n"
        "\n"
        f"expected {name} = {_format_golden(expected)}\n"
        f"actual   {name} = {_format_golden(actual)}\n",
        pytrace=False,
    )


@pytest.fixture(scope="module")
def corpus():
    """A single-group corpus, shared by every golden in this module."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield _write_corpus(Path(tmpdir) / "corpus")


@pytest.fixture(scope="module")
def two_group_corpus():
    """Two groups with distinct filename prefixes, for the balanced goldens."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield {
            "speech": _write_corpus(Path(tmpdir) / "speech", prefix="speech"),
            "music": _write_corpus(Path(tmpdir) / "music", prefix="music"),
        }


# --------------------------------------------------------------------------
# create_audio_dataset
# --------------------------------------------------------------------------

#: ``create_audio_dataset(shuffle_seed=1234, num_epochs=2)`` over ``corpus``.
GOLDEN_SIMPLE_SHUFFLE_SEED_1234 = [
    ("clip_5.wav", 0.245261),
    ("clip_3.wav", 0.775029),
    ("clip_2.wav", 0.832257),
    ("clip_0.wav", 0.390252),
    ("clip_1.wav", 0.619268),
    ("clip_4.wav", 0.404085),
    ("clip_0.wav", 0.387507),
    ("clip_3.wav", 0.849808),
    ("clip_2.wav", 0.566839),
    ("clip_4.wav", 0.979807),
    ("clip_1.wav", 0.07089),
    ("clip_5.wav", 1.728505),
]

#: The same corpus and ``shuffle_seed``, with ``excerpt_seed`` set explicitly.
#: The file *order* must match the list above; only the offsets may move.
GOLDEN_SIMPLE_EXCERPT_SEED_99 = [
    ("clip_5.wav", 0.168053),
    ("clip_3.wav", 0.640706),
    ("clip_2.wav", 0.416304),
    ("clip_0.wav", 0.054937),
    ("clip_1.wav", 0.018522),
    ("clip_4.wav", 1.156923),
    ("clip_0.wav", 0.429),
    ("clip_3.wav", 0.421807),
    ("clip_2.wav", 0.992622),
    ("clip_4.wav", 1.497187),
    ("clip_1.wav", 0.624216),
    ("clip_5.wav", 0.928765),
]

#: ``shuffle=False`` visits the corpus in sorted order, once per epoch, but the
#: excerpt offsets still come from ``shuffle_seed`` via the derived pair.
GOLDEN_SIMPLE_UNSHUFFLED = [
    ("clip_0.wav", 0.070075),
    ("clip_1.wav", 0.465018),
    ("clip_2.wav", 0.832257),
    ("clip_3.wav", 0.975631),
    ("clip_4.wav", 1.238536),
    ("clip_5.wav", 0.471433),
]


def test_simple_dataset_golden_sequence(corpus):
    """A fixed ``shuffle_seed`` pins both the file order and the offsets."""
    dataset = create_audio_dataset(
        sources=corpus,
        shuffle=True,
        shuffle_seed=1234,
        num_epochs=2,
        sample_rate=SAMPLE_RATE,
        duration=EXCERPT_DURATION,
    )
    _assert_golden(
        _collect(dataset),
        GOLDEN_SIMPLE_SHUFFLE_SEED_1234,
        "GOLDEN_SIMPLE_SHUFFLE_SEED_1234",
    )


def test_simple_dataset_golden_sequence_is_repeatable(corpus):
    """Two datasets built with the same seed agree item for item.

    The golden above pins *what* the seed produces; this pins that it produces
    it every time, which is the property a resumed run depends on.
    """
    kwargs = dict(
        sources=corpus,
        shuffle=True,
        shuffle_seed=1234,
        num_epochs=2,
        sample_rate=SAMPLE_RATE,
        duration=EXCERPT_DURATION,
    )
    assert _collect(create_audio_dataset(**kwargs)) == _collect(
        create_audio_dataset(**kwargs)
    )


def test_simple_dataset_golden_excerpt_seed(corpus):
    """``excerpt_seed`` moves the offsets and leaves the file order alone."""
    dataset = create_audio_dataset(
        sources=corpus,
        shuffle=True,
        shuffle_seed=1234,
        excerpt_seed=99,
        num_epochs=2,
        sample_rate=SAMPLE_RATE,
        duration=EXCERPT_DURATION,
    )
    actual = _collect(dataset)
    _assert_golden(
        actual, GOLDEN_SIMPLE_EXCERPT_SEED_99, "GOLDEN_SIMPLE_EXCERPT_SEED_99"
    )

    # The two goldens are the same run with one knob turned: same files in the
    # same order, different excerpts out of them.
    assert [name for name, _ in actual] == [
        name for name, _ in GOLDEN_SIMPLE_SHUFFLE_SEED_1234
    ]
    assert [offset for _, offset in actual] != [
        offset for _, offset in GOLDEN_SIMPLE_SHUFFLE_SEED_1234
    ]


def test_simple_dataset_golden_unshuffled(corpus):
    """``shuffle=False`` pins sorted file order, and still pins the offsets."""
    dataset = create_audio_dataset(
        sources=corpus,
        shuffle=False,
        shuffle_seed=1234,
        num_epochs=1,
        sample_rate=SAMPLE_RATE,
        duration=EXCERPT_DURATION,
    )
    actual = _collect(dataset)
    _assert_golden(actual, GOLDEN_SIMPLE_UNSHUFFLED, "GOLDEN_SIMPLE_UNSHUFFLED")
    assert [name for name, _ in actual] == sorted(name for name, _ in actual)


def test_simple_dataset_shuffle_seed_actually_shuffles(corpus):
    """A different ``shuffle_seed`` gives a different order.

    Cheap insurance that the golden above pins a *seeded* order rather than a
    seed argument that silently stopped being plumbed through.
    """
    other = create_audio_dataset(
        sources=corpus,
        shuffle=True,
        shuffle_seed=4321,
        num_epochs=2,
        sample_rate=SAMPLE_RATE,
        duration=EXCERPT_DURATION,
    )
    assert [name for name, _ in _collect(other)] != [
        name for name, _ in GOLDEN_SIMPLE_SHUFFLE_SEED_1234
    ]


# --------------------------------------------------------------------------
# create_balanced_audio_dataset
# --------------------------------------------------------------------------

#: The first 12 items of an equally weighted two-group mixture with
#: ``shuffle_seed=2024``. Each group's own seeds are derived from the base seed
#: and the group's *name*, so this also pins ``_derive_group_seed``.
GOLDEN_BALANCED_SHUFFLE_SEED_2024 = [
    ("speech", "speech_1.wav", 0.31897),
    ("music", "music_2.wav", 0.298146),
    ("speech", "speech_4.wav", 0.623947),
    ("music", "music_3.wav", 1.237509),
    ("speech", "speech_0.wav", 0.239471),
    ("music", "music_4.wav", 0.058344),
    ("speech", "speech_5.wav", 1.397933),
    ("music", "music_5.wav", 0.394828),
    ("speech", "speech_3.wav", 1.21494),
    ("music", "music_0.wav", 0.370136),
    ("speech", "speech_2.wav", 0.487737),
    ("music", "music_1.wav", 0.633481),
]

#: The same mixture with ``excerpt_seed`` set explicitly: same groups, same
#: files, same order; different offsets.
GOLDEN_BALANCED_EXCERPT_SEED_7 = [
    ("speech", "speech_1.wav", 0.088404),
    ("music", "music_2.wav", 0.843525),
    ("speech", "speech_4.wav", 0.662797),
    ("music", "music_3.wav", 1.157383),
    ("speech", "speech_0.wav", 0.421081),
    ("music", "music_4.wav", 0.298071),
    ("speech", "speech_5.wav", 0.138982),
    ("music", "music_5.wav", 0.927634),
    ("speech", "speech_3.wav", 0.57075),
    ("music", "music_0.wav", 0.231263),
    ("speech", "speech_2.wav", 0.204552),
    ("music", "music_1.wav", 0.374943),
]


def _balanced(two_group_corpus, **overrides):
    """Build the balanced dataset the goldens were generated from."""
    kwargs = dict(
        sources={
            "speech": two_group_corpus["speech"],
            "music": two_group_corpus["music"],
        },
        shuffle=True,
        shuffle_seed=2024,
        num_epochs=1,
        sample_rate=SAMPLE_RATE,
        duration=EXCERPT_DURATION,
    )
    kwargs.update(overrides)
    return create_balanced_audio_dataset(**kwargs).slice(slice(0, 12))


def test_balanced_dataset_golden_sequence(two_group_corpus):
    """A fixed ``shuffle_seed`` pins the mixture, per group."""
    _assert_golden(
        _collect(_balanced(two_group_corpus), grouped=True),
        GOLDEN_BALANCED_SHUFFLE_SEED_2024,
        "GOLDEN_BALANCED_SHUFFLE_SEED_2024",
    )


def test_balanced_dataset_golden_excerpt_seed(two_group_corpus):
    """``excerpt_seed`` moves every group's offsets, not its file order."""
    actual = _collect(_balanced(two_group_corpus, excerpt_seed=7), grouped=True)
    _assert_golden(
        actual, GOLDEN_BALANCED_EXCERPT_SEED_7, "GOLDEN_BALANCED_EXCERPT_SEED_7"
    )

    assert [(group, name) for group, name, _ in actual] == [
        (group, name) for group, name, _ in GOLDEN_BALANCED_SHUFFLE_SEED_2024
    ]
    assert [offset for *_, offset in actual] != [
        offset for *_, offset in GOLDEN_BALANCED_SHUFFLE_SEED_2024
    ]


def test_balanced_group_streams_are_pinned_to_group_names(two_group_corpus):
    """Adding a third group leaves the original two groups' streams intact.

    Per-group seeds are derived from the group's *name*, so a new group must
    not reshuffle the existing ones. Without this, the golden above would still
    pass while every real run's data changed the day someone added a source.
    """
    third = _write_corpus(Path(two_group_corpus["music"]).parent / "noise", "noise")
    dataset = create_balanced_audio_dataset(
        sources={
            "speech": two_group_corpus["speech"],
            "music": two_group_corpus["music"],
            "noise": third,
        },
        shuffle=True,
        shuffle_seed=2024,
        num_epochs=1,
        sample_rate=SAMPLE_RATE,
        duration=EXCERPT_DURATION,
    ).slice(slice(0, 18))

    actual = _collect(dataset, grouped=True)
    speech_and_music = [item for item in actual if item[0] != "noise"]
    assert speech_and_music == GOLDEN_BALANCED_SHUFFLE_SEED_2024


# --------------------------------------------------------------------------
# Seed entropy: full 64-bit seeds must survive derivation
# --------------------------------------------------------------------------


def test_derived_seeds_distinguish_high_bits():
    """Seeds differing only above bit 31 must derive to distinct results.

    The helpers used to mask ``base_seed & 0xFFFFFFFF`` before feeding
    ``SeedSequence``, so a caller seeding from ``time_ns()`` or a 64-bit hash
    would alias distinct seeds to byte-identical shuffle orders and excerpt
    offsets. ``SeedSequence`` accepts arbitrary non-negative ints, so the mask
    only threw away entropy.
    """
    assert _derive_seed_pair(1) != _derive_seed_pair(2**32 + 1)
    assert _derive_group_seed(1, "g", "shuffle") != _derive_group_seed(
        2**32 + 1, "g", "shuffle"
    )


def test_derived_seeds_small_values_are_stable():
    """Removing the mask must not move any small-seed output.

    Masking a seed below ``2**32`` is a no-op, so these golden values pin that
    the entropy fix stayed backward-compatible for the common small-seed case
    (the golden data tests above depend on this).
    """
    assert _derive_seed_pair(1) == (1835504127, 1731038949)
    assert _derive_group_seed(1, "g", "shuffle") == 475119945
