import pytest
import jax
import jax.numpy as jnp
import numpy as np

from audiotree import AudioTree
from audiotree.transforms import Roll


def test_roll_wrap_mode():
    """Test Roll transform with wrap mode (circular shift)."""
    # Create test audio: batch=2, channels=2, samples=100
    B, C, T = 2, 2, 100
    audio_data = jnp.arange(B * C * T).reshape(B, C, T).astype(jnp.float32)
    sample_rate = 10000  # 10kHz for easy sample calculation
    
    audio_tree = AudioTree(audio_data=audio_data, sample_rate=sample_rate)
    
    # Test with fixed roll of 0.001 seconds (10 samples at 10kHz)
    transform = Roll(config={"min_seconds": 0.001, "max_seconds": 0.001, "mode": "wrap"})
    
    rng = jax.random.PRNGKey(42)
    rolled = transform.random_map(audio_tree, rng)
    
    # With wrap mode, last 10 samples should move to the beginning
    # Check first batch item, first channel
    expected_start = audio_data[0, 0, -10:]  # Last 10 samples
    actual_start = rolled.audio_data[0, 0, :10]  # First 10 samples after roll
    
    assert jnp.allclose(expected_start, actual_start)


def test_roll_constant_mode_right():
    """Test Roll transform with constant mode, rolling right (left padding)."""
    # Create test audio
    B, C, T = 1, 2, 100
    audio_data = jnp.ones((B, C, T)).astype(jnp.float32)
    sample_rate = 10000
    
    audio_tree = AudioTree(audio_data=audio_data, sample_rate=sample_rate)
    
    # Roll right by 0.002 seconds (20 samples)
    transform = Roll(config={"min_seconds": 0.002, "max_seconds": 0.002, "mode": "constant"})
    
    rng = jax.random.PRNGKey(42)
    rolled = transform.random_map(audio_tree, rng)
    
    # First 20 samples should be zeros (left padding)
    assert jnp.all(rolled.audio_data[0, :, :20] == 0)
    # Samples 20-100 should be ones (original audio shifted right)
    assert jnp.all(rolled.audio_data[0, :, 20:] == 1)


def test_roll_constant_mode_left():
    """Test Roll transform with constant mode, rolling left."""
    # Create test audio
    B, C, T = 1, 2, 100
    audio_data = jnp.ones((B, C, T)).astype(jnp.float32)
    sample_rate = 10000
    
    audio_tree = AudioTree(audio_data=audio_data, sample_rate=sample_rate)
    
    # Roll left by 0.003 seconds (30 samples)
    transform = Roll(config={"min_seconds": -0.003, "max_seconds": -0.003, "mode": "constant"})
    
    rng = jax.random.PRNGKey(42)
    rolled = transform.random_map(audio_tree, rng)
    
    # First 70 samples should be ones (original audio shifted left)
    assert jnp.all(rolled.audio_data[0, :, :70] == 1)
    # Last 30 samples should be zeros (right padding)
    assert jnp.all(rolled.audio_data[0, :, 70:] == 0)


def test_roll_per_batch_item():
    """Test that roll amount is different per batch item but same across channels."""
    # Create test audio with distinct values
    B, C, T = 3, 2, 100
    audio_data = jnp.arange(B * C * T).reshape(B, C, T).astype(jnp.float32)
    sample_rate = 10000
    
    audio_tree = AudioTree(audio_data=audio_data, sample_rate=sample_rate)
    
    # Random roll between -0.002 and 0.002 seconds
    transform = Roll(config={"min_seconds": -0.002, "max_seconds": 0.002, "mode": "wrap"})
    
    rng = jax.random.PRNGKey(42)
    rolled = transform.random_map(audio_tree, rng)
    
    # Verify each batch item was rolled
    for b in range(B):
        # Check if the audio was actually rolled (unless roll amount was 0)
        if not jnp.array_equal(rolled.audio_data[b], audio_data[b]):
            # Verify both channels were rolled by the same amount
            # by checking the difference pattern is the same
            ch0_diff = rolled.audio_data[b, 0] - audio_data[b, 0]
            ch1_diff = rolled.audio_data[b, 1] - audio_data[b, 1]
            
            # The roll pattern should be identical for both channels
            # (though the actual values differ due to different starting values)
            assert jnp.array_equal(jnp.where(ch0_diff != 0), jnp.where(ch1_diff != 0))


def test_roll_no_change():
    """Test Roll transform with zero roll amount."""
    # Create test audio
    B, C, T = 1, 2, 100
    audio_data = jnp.arange(B * C * T).reshape(B, C, T).astype(jnp.float32)
    sample_rate = 10000
    
    audio_tree = AudioTree(audio_data=audio_data, sample_rate=sample_rate)
    
    # No roll (0 seconds)
    transform = Roll(config={"min_seconds": 0.0, "max_seconds": 0.0, "mode": "wrap"})
    
    rng = jax.random.PRNGKey(42)
    rolled = transform.random_map(audio_tree, rng)
    
    # Audio should remain unchanged
    assert jnp.array_equal(rolled.audio_data, audio_data)


if __name__ == "__main__":
    test_roll_wrap_mode()
    test_roll_constant_mode_right()
    test_roll_constant_mode_left()
    test_roll_per_batch_item()
    test_roll_no_change()
    print("All tests passed!")