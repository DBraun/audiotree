"""Example demonstrating AudioWriter and ManifestDataSource round-trip workflow."""

import tempfile
from pathlib import Path

import numpy as np
from audiotree import AudioTree, AudioWriter
from audiotree.datasources import ManifestDataSource


def main():
    """Demonstrate writing and reading audio data with manifests."""

    # Create a temporary directory for our example
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "audio_output"

        print("=== AudioWriter Example ===\n")

        # 1. Create some example AudioTree objects with metadata
        print("1. Creating AudioTree objects with metadata...")

        trees = []
        for i in range(3):
            # Generate different types of audio
            if i == 0:
                # Low frequency sine wave
                t = np.linspace(0, 1, 44100)
                audio = np.sin(2 * np.pi * 220 * t)  # A3 note
                pitch = 57.0  # MIDI note for A3
            elif i == 1:
                # Higher frequency sine wave
                t = np.linspace(0, 1, 44100)
                audio = np.sin(2 * np.pi * 440 * t)  # A4 note
                pitch = 69.0  # MIDI note for A4
            else:
                # White noise
                audio = np.random.randn(44100) * 0.1
                pitch = None

            # Create AudioTree with metadata
            audio_tree = AudioTree.create(
                audio_data=audio.reshape(1, 1, -1),  # Shape: (batch=1, channels=1, samples)
                sample_rate=44100,
                pitch=np.array([pitch]) if pitch else None,
                velocity=np.array([64 + i * 20]),
                filepaths=[f"original_{i}.wav"]
            )

            # Calculate loudness
            audio_tree = audio_tree.replace_loudness()
            trees.append(audio_tree)

        print(f"  Created {len(trees)} AudioTree objects\n")

        # 2. Write audio files with AudioWriter
        print("2. Writing audio files with manifest...")

        with AudioWriter(
            output_dir,
            pattern="audio_{index:03d}.wav",
        ) as writer:
            for i, audio_tree in enumerate(trees):
                paths = writer.write(audio_tree, tags={
                    "category": "sine" if i < 2 else "noise",
                    "example_id": i
                })
                print(f"  Wrote {paths[0].name}")

            stats = writer.get_stats()
            print(f"\n  Total files written: {stats['total_files']}")

        # 3. Read back with ManifestDataSource
        print("3. Reading back with ManifestDataSource...")

        # Load all files
        source = ManifestDataSource.from_writer_output(
            output_dir,
        )

        print(f"  Loaded {len(source)} files from manifest\n")

        # 4. Demonstrate filtering capabilities
        print("4. Filtering examples:")

        # Filter by tag
        sine_only = source.filter_by_tag("category", "sine")
        print(f"  - Files with category='sine': {len(sine_only)}")

        # Filter by loudness
        loud_only = source.filter_by_loudness(min_lufs=-30)
        print(f"  - Files louder than -30 LUFS: {len(loud_only)}")

        print("\n5. Accessing metadata from loaded files:")

        # Load and inspect first file
        first_audio = source[0]
        print(f"  First file:")
        print(f"    - Shape: {first_audio.audio_data.shape}")
        print(f"    - Sample rate: {first_audio.sample_rate}")
        print(f"    - Loudness: {first_audio.loudness[0]:.1f} LUFS")
        if first_audio.pitch is not None:
            print(f"    - Pitch: MIDI {first_audio.pitch[0]:.0f}")
        print(f"    - Velocity: {first_audio.velocity[0]}")
        print(f"    - Original source: {first_audio.metadata.get('original_source', 'N/A')}")
        print(f"    - Tags: {first_audio.metadata.get('tags', {})}")

        # 6. Show manifest entry structure
        print("\n6. Raw manifest entry example:")
        entry = source.get_entry(0)
        print(f"  Keys in manifest: {list(entry.keys())}")
        print(f"  Filename: {entry['filename']}")
        print(f"  Duration: {entry['duration_seconds']:.2f} seconds")

        print("\n=== Example Complete ===")
        print(f"Output was written to: {output_dir}")
        print(f"Manifest saved as: {output_dir}/manifest.npz")


if __name__ == "__main__":
    main()