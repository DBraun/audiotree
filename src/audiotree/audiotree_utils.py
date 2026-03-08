"""Utilities for AudioTree field extraction and reconstruction."""

from typing import Dict, List
import numpy as np

from audiotree.core import AudioTree
from audiotree.memmap_writer import FieldSpec


# Main AudioTree fields (excluding sample_rate which is non-pytree and stored in manifest)
# These are the actual attribute names on AudioTree
AUDIOTREE_MAIN_FIELDS = {
    "audio_data",
    "loudness",
    "pitch",
    "velocity",
    "note_duration",
    "codes",
    "latents",
}


class AudioTreeFieldExtractor:
    """Extracts and reconstructs AudioTree objects for memmap storage."""

    @staticmethod
    def extract_fields(
        tree: AudioTree,
        prefix: str = "",
    ) -> Dict[str, np.ndarray]:
        """Extract all fields from AudioTree to flat dict.

        Args:
            tree: AudioTree to extract from
            prefix: Prefix for field names (e.g., "dry_")

        Returns:
            Dict mapping field names to numpy arrays

        Example:
            tree = AudioTree(audio_data=..., loudness=..., metadata={"mel": ...})
            fields = extract_fields(tree, prefix="dry_")
            # Returns: {"dry_audio_data": array, "dry_loudness": array, "dry_mel": array}
        """
        fields = {}

        # Extract main AudioTree fields dynamically
        for field_name in AUDIOTREE_MAIN_FIELDS:
            value = getattr(tree, field_name, None)
            if value is not None:
                fields[f"{prefix}{field_name}"] = value

        # Flatten metadata recursively
        if tree.metadata:
            metadata_fields = AudioTreeFieldExtractor._flatten_metadata(
                tree.metadata, prefix=prefix
            )
            fields.update(metadata_fields)

        return fields

    @staticmethod
    def _flatten_metadata(
        metadata: Dict,
        prefix: str = "",
    ) -> Dict[str, np.ndarray]:
        """Recursively flatten metadata dict.

        Args:
            metadata: Metadata dict (can contain nested dicts)
            prefix: Current prefix for field names

        Returns:
            Flat dict with all numpy arrays
        """
        fields = {}

        for key, value in metadata.items():
            field_name = f"{prefix}{key}"

            if isinstance(value, np.ndarray):
                fields[field_name] = value
            elif isinstance(value, dict):
                # Recursively flatten nested dicts
                nested_fields = AudioTreeFieldExtractor._flatten_metadata(
                    value, prefix=f"{field_name}_"
                )
                fields.update(nested_fields)
            elif isinstance(value, (int, float, str, bytes)):
                # Skip scalar/string metadata (would go in strings dict)
                pass
            # Ignore other types

        return fields

    @staticmethod
    def infer_field_specs(fields: Dict[str, np.ndarray]) -> List[FieldSpec]:
        """Infer FieldSpec from extracted fields.

        Args:
            fields: Dict of field names to arrays (with batch dimension)

        Returns:
            List of FieldSpec (shapes exclude batch dimension)
        """
        specs = []
        for name, array in fields.items():
            if not isinstance(array, np.ndarray):
                continue

            # Get shape excluding batch dimension (first axis)
            shape = array.shape[1:] if array.ndim > 1 else ()
            specs.append(FieldSpec(name, array.dtype, shape))

        return specs

    @staticmethod
    def reconstruct_audiotree(
        fields: Dict[str, np.ndarray],
        tree_name: str,
        sample_rate: int,
    ) -> AudioTree:
        """Reconstruct AudioTree from flattened fields.

        Args:
            fields: Dict of all fields (flat)
            tree_name: Name of the AudioTree (e.g., "dry", "wet")
            sample_rate: Audio sample rate

        Returns:
            Reconstructed AudioTree

        Example:
            fields = {"dry_audio_data": array, "dry_loudness": array, "dry_mel": array}
            tree = reconstruct_audiotree(fields, "dry", 48000)
            # Returns: AudioTree(audio_data=..., loudness=..., metadata={"mel": ...})
        """
        prefix = f"{tree_name}_"
        tree_data = {}
        metadata = {}

        # Extract fields for this AudioTree
        for field_name, array in fields.items():
            if not field_name.startswith(prefix):
                continue

            suffix = field_name[len(prefix):]

            # Categorize as main field or metadata
            if suffix in AUDIOTREE_MAIN_FIELDS:
                # Suffix is the actual AudioTree attribute name
                tree_data[suffix] = array
            else:
                # Goes in metadata
                metadata[suffix] = array

        # Create AudioTree with gathered fields dynamically
        # audio_data defaults to None if not present (e.g., when loading only AFx-Rep latents)
        kwargs = {
            "audio_data": tree_data.get("audio_data", None),
            "sample_rate": sample_rate,
            "metadata": metadata if metadata else {}
        }

        # Add optional fields
        for field_name in AUDIOTREE_MAIN_FIELDS:
            if field_name != "audio_data" and field_name in tree_data:
                kwargs[field_name] = tree_data[field_name]

        return AudioTree(**kwargs)

    @staticmethod
    def reconstruct_single_audiotree(
        fields: Dict[str, np.ndarray],
        sample_rate: int,
    ) -> AudioTree:
        """Reconstruct AudioTree from fields with no prefix.

        Used when a single AudioTree was written directly (not wrapped
        in a dict), so field names have no tree-name prefix.

        Args:
            fields: Dict of all fields (unprefixed, e.g. "audio_data", "loudness", "mel")
            sample_rate: Audio sample rate

        Returns:
            Reconstructed AudioTree
        """
        tree_data = {}
        metadata = {}

        for field_name, array in fields.items():
            if field_name in AUDIOTREE_MAIN_FIELDS:
                tree_data[field_name] = array
            else:
                metadata[field_name] = array

        kwargs = {
            "audio_data": tree_data.get("audio_data", None),
            "sample_rate": sample_rate,
            "metadata": metadata if metadata else {},
        }

        for field_name in AUDIOTREE_MAIN_FIELDS:
            if field_name != "audio_data" and field_name in tree_data:
                kwargs[field_name] = tree_data[field_name]

        return AudioTree(**kwargs)
