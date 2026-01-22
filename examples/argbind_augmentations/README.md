# ArgBind Augmentations Examples

This directory contains examples demonstrating how to use [argbind](https://github.com/pseeth/argbind) with AudioTree transforms for configurable audio data augmentation.

## Overview

ArgBind allows you to:
- Configure transform parameters via YAML files or command-line arguments
- Use scopes for different configurations (e.g., train vs val)
- Save and load experiment configurations
- Build CLIs from function signatures without boilerplate

## Examples

### Example 1: Basic Usage (`main.py`)

Demonstrates basic argbind usage with AudioTree transforms.

**Features:**
- Binding individual transform functions
- Loading config from YAML
- Applying random (volume_norm) and deterministic (trim) transforms

**Run:**
```bash
# With config file (recommended)
python main.py --args.load=config.yml

# Save config after run
python main.py --args.load=config.yml --args.save=my_config.yml

# Debug mode (see how functions are called)
python main.py --args.load=config.yml --args.debug=1

# Note: Command-line override of nested dict parameters (like config)
# is not supported by argbind. Use YAML files for configuration.
```

**Config file (`config.yml`):**
```yaml
volume_norm.min_db: -20
volume_norm.max_db: -15

trim.length: 1.0
```

### Example 2: Advanced Usage with Scopes (`main2.py`)

Demonstrates advanced argbind features with scoped configurations.

**Features:**
- Using `bind_module()` to bind all transforms at once
- Scoped configs for train vs validation
- JAX JIT compilation with argbind
- Sequential transform application

**Run:**
```bash
# With config file
python main2.py --args.load=config2.yml

# Note: Argbind does not support JSON syntax for nested dict parameters from CLI.
# To customize configs, edit the YAML file or create a new one.
```

**Config file (`config2.yml`):**
```yaml
# Training augmentation (more aggressive)
train/volume_norm.min_db: -25
train/volume_norm.max_db: -15

# Validation augmentation (deterministic)
val/volume_norm.min_db: -20
val/volume_norm.max_db: -20

# Shared config (no scope prefix)
trim.length: 1.0

# List of transforms to apply
train/augment_batch.transforms:
  - volume_norm
  - trim

val/augment_batch.transforms:
  - volume_norm
  - trim
```

## Key Concepts

### Transform Config Structure

AudioTree transforms are functions with direct parameters:
```python
transform = volume_norm(min_db=-20, max_db=-15)
```

With argbind, you can set parameters via YAML:
```yaml
volume_norm.min_db: -20
volume_norm.max_db: -15
```

Or directly from the command line:
```bash
--volume_norm.min_db=-20 --volume_norm.max_db=-15
```

### Scoping

Scopes allow different configs for the same transform:
```yaml
train/volume_norm.min_db: -25  # Used in "train" scope
train/volume_norm.max_db: -15

val/volume_norm.min_db: -20    # Used in "val" scope
val/volume_norm.max_db: -20
```

Access via:
```python
with argbind.scope(args, "train"):
    transform = volume_norm()  # Uses train config

with argbind.scope(args, "val"):
    transform = volume_norm()  # Uses val config
```

You can also override from command line:
```bash
--train/volume_norm.min_db=-30 --val/volume_norm.min_db=-25
```

## Testing

Run the test suite:
```bash
# Test basic example
pytest examples/argbind_augmentations/test_main.py -v

# Test advanced example
pytest examples/argbind_augmentations/test_main2.py -v

# Test all
pytest examples/argbind_augmentations/ -v
```

## Available Transforms

See `audiotree.transforms` for all available transforms:
- `volume_norm` - Normalize to random loudness (random transform)
- `volume_change` - Add random gain (random transform)
- `trim` - Trim to fixed length (map transform)
- `invert_phase` - Invert audio phase (random transform)
- `swap_stereo` - Swap stereo channels (random transform)
- `roll` - Circular shift audio (random transform)
- `corrupt_phase` - Corrupt phase spectrum (random transform)
- `shift_phase` - Shift phase spectrum (random transform)
- `rescale_audio` - Rescale to [-1, 1] range (map transform)
- `mono` - Convert to mono (map transform)
- `stereo` - Convert to stereo (map transform)
- `identity` - No-op transform (map transform)
- `choose` - Randomly select from multiple transforms
- `encode_with_codec` - Encode with neural codec
- `encode_latents` - Encode to latent space
- And more...

## Further Reading

- [ArgBind Documentation](https://github.com/pseeth/argbind)
- [ArgBind Examples](https://github.com/pseeth/argbind/tree/main/examples)
- [AudioTree Transforms](../../src/audiotree/transforms/)