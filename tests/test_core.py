"""Tests for AudioTree core functionality."""
from pathlib import Path

import jax
import numpy as np

from audiotree.core import AudioTree


def test_audiotree_create_with_filepaths():
    """Test that AudioTree.create accepts and processes filepaths parameter."""
    # Test with 1D audio data and single filepath string
    audio_1d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    tree1 = AudioTree.create(audio_1d, 44100, filepaths="test1.wav")
    
    assert tree1.audio_data.shape == (1, 1, 5)  # Should be expanded to (batch, channels, samples)
    assert tree1.filepath == ["test1.wav"]
    assert "filepath" in tree1.metadata
    
    # Test with 2D audio data and single filepath Path
    audio_2d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2 channels, 3 samples
    tree2 = AudioTree.create(audio_2d, 44100, filepaths=Path("test2.wav"))
    
    assert tree2.audio_data.shape == (1, 2, 3)  # Should be expanded to (batch, channels, samples)
    assert tree2.filepath == ["test2.wav"]
    
    # Test with 3D audio data and list of filepaths
    audio_3d = np.array([[[1.0, 2.0, 3.0]]])  # Already correct shape: (1, 1, 3)
    filepaths = ["file1.wav", "file2.wav", Path("file3.wav")]
    tree3 = AudioTree.create(audio_3d, 44100, filepaths=filepaths)
    
    assert tree3.audio_data.shape == (1, 1, 3)  # Should remain unchanged
    assert tree3.filepath == ["file1.wav", "file2.wav", "file3.wav"]
    
    # Test with no filepaths (should work as before)
    tree4 = AudioTree.create(audio_1d, 44100)
    
    assert tree4.audio_data.shape == (1, 1, 5)
    assert tree4.filepath == []  # Empty list when no filepaths provided
    assert "filepath" not in tree4.metadata


def test_audiotree_constructor_compatibility():
    """Test that original AudioTree constructor still works for backward compatibility."""
    audio_3d = np.array([[[1.0, 2.0, 3.0]]])  # Use 3D data since constructor won't reshape
    tree = AudioTree(audio_3d, 44100)
    
    assert tree.audio_data.shape == (1, 1, 3)
    assert tree.filepath == []


def test_audiotree_create_audio_dimensionality():
    """Test that AudioTree.create handles different audio dimensionalities correctly."""
    sample_rate = 44100
    
    # Test 1D audio
    audio_1d = np.array([1.0, 2.0, 3.0])
    tree1 = AudioTree.create(audio_1d, sample_rate)
    assert tree1.audio_data.shape == (1, 1, 3)
    
    # Test 2D audio
    audio_2d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    tree2 = AudioTree.create(audio_2d, sample_rate)
    assert tree2.audio_data.shape == (1, 2, 3)
    
    # Test 3D audio (already correct)
    audio_3d = np.array([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]])
    tree3 = AudioTree.create(audio_3d, sample_rate)
    assert tree3.audio_data.shape == (2, 1, 3)


def test_audiotree_create_metadata_handling():
    """Test that AudioTree.create handles metadata correctly with filepaths."""
    audio_data = np.array([1.0, 2.0, 3.0])
    sample_rate = 44100
    
    # Test with existing metadata and filepaths
    existing_metadata = {"custom_key": "custom_value"}
    tree1 = AudioTree.create(
        audio_data, 
        sample_rate, 
        metadata=existing_metadata, 
        filepaths="test.wav"
    )
    
    # Should preserve existing metadata and add filepath
    assert tree1.metadata["custom_key"] == "custom_value"
    assert "filepath" in tree1.metadata
    assert tree1.filepath == ["test.wav"]
    
    # Test that original metadata dict is not modified
    assert "filepath" not in existing_metadata
    
    # Test with filepaths but no existing metadata
    tree2 = AudioTree.create(audio_data, sample_rate, filepaths="test2.wav")
    assert "filepath" in tree2.metadata
    assert tree2.filepath == ["test2.wav"]


def test_split_by_batch():
    x = AudioTree(np.zeros((4, 1, 44100)), 44100)
    trees = [x, x, x]
    big_tree = jax.tree.map(lambda *xs: np.concatenate(xs, axis=0), *trees)
    assert big_tree.audio_data.shape == (12, 1, 44100)
    split_trees = big_tree.split_by_batch(2)
    assert len(split_trees) == 2
    assert split_trees[0].audio_data.shape == (6, 1, 44100)
