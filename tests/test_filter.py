"""Tests for AudioTree filter functionality."""
import numpy as np

from audiotree.core import AudioTree


def test_filter_basic():
    """Test basic filtering functionality."""
    # Create AudioTree with batch size 4
    waveform = np.array([
        [[1.0, 2.0, 3.0]],  # batch 0
        [[4.0, 5.0, 6.0]],  # batch 1
        [[7.0, 8.0, 9.0]],  # batch 2
        [[10.0, 11.0, 12.0]]  # batch 3
    ])
    tree = AudioTree(waveform, 44100)

    # Filter to keep only even-indexed batches (0, 2)
    def keep_even_batches(mini_tree):
        # mini_tree will be individual AudioTree objects with batch size 1
        first_sample = mini_tree.waveform[0, 0, 0]  # Get first sample value
        return first_sample in [1.0, 7.0]  # Keep batches 0 and 2

    filtered_tree = tree.filter(keep_even_batches)

    # Should have 2 batches remaining (original batches 0 and 2)
    assert filtered_tree.waveform.shape == (2, 1, 3)
    # First filtered batch should be original batch 0
    np.testing.assert_array_equal(filtered_tree.waveform[0], [[1.0, 2.0, 3.0]])
    # Second filtered batch should be original batch 2
    np.testing.assert_array_equal(filtered_tree.waveform[1], [[7.0, 8.0, 9.0]])


def test_filter_single_item():
    """Test filtering that keeps only one item."""
    waveform = np.array([
        [[1.0, 2.0, 3.0]],
        [[4.0, 5.0, 6.0]],
        [[7.0, 8.0, 9.0]]
    ])
    tree = AudioTree(waveform, 44100)

    # Filter to keep only the second batch
    def keep_second_batch(mini_tree):
        first_sample = mini_tree.waveform[0, 0, 0]
        return first_sample == 4.0

    filtered_tree = tree.filter(keep_second_batch)

    # Should have 1 batch remaining
    assert filtered_tree.waveform.shape == (1, 1, 3)
    np.testing.assert_array_equal(filtered_tree.waveform[0], [[4.0, 5.0, 6.0]])


def test_filter_keep_all():
    """Test filtering that keeps all items."""
    waveform = np.array([
        [[1.0, 2.0, 3.0]],
        [[4.0, 5.0, 6.0]],
        [[7.0, 8.0, 9.0]]
    ])
    tree = AudioTree(waveform, 44100)

    # Filter that accepts everything
    def keep_all(mini_tree):
        return True

    filtered_tree = tree.filter(keep_all)

    # Should keep all batches
    assert filtered_tree.waveform.shape == (3, 1, 3)
    np.testing.assert_array_equal(filtered_tree.waveform, waveform)


def test_filter_preserves_sample_rate():
    """Test that filtering preserves sample rate."""
    waveform = np.array([[[1.0, 2.0]], [[3.0, 4.0]]])
    sample_rate = 48000
    tree = AudioTree(waveform, sample_rate)

    def keep_first(mini_tree):
        return mini_tree.waveform[0, 0, 0] == 1.0

    filtered_tree = tree.filter(keep_first)

    assert filtered_tree.sample_rate == sample_rate