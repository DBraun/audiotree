"""Core transform classes.

This module contains only the Batch operation, which is kept for compatibility
with grain's BatchOperation pattern.

All other transforms have been migrated to function-based API in functional.py.
"""

from typing import Any, Sequence

import numpy as np

from grain._src.core import tree_lib
from grain._src.python.shared_memory_array import SharedMemoryArray
from grain.python import BatchOperation


class Batch(BatchOperation):
    """A special version of grain's BatchOperation that concatenates on the batch axis instead of stacking.

    This is kept as a class because it directly inherits from grain's BatchOperation
    and has special batching logic.

    Example:
        from audiotree.transforms import Batch

        batch_op = Batch(batch_size=32)
        # Use with grain's .batch() method on IterDataset
        iter_ds = ds.to_iter_dataset().batch(32)
    """

    def __post_init__(self):
        super().__post_init__()
        self._display_deprecation_message = False

    def _default_batch_fn(self, input_records: Sequence[Any]):
        """Batches records together and copies Numpy arrays to Shared Memory."""
        self._validate_structure(input_records)

        def stacking_function(*args):
            first_arg = np.asanyarray(args[0])
            shape, dtype = (len(args) * first_arg.shape[0],) + first_arg.shape[1:], first_arg.dtype
            if not self._use_shared_memory or dtype.hasobject:
                return np.concatenate(args, axis=0)
            return np.concatenate(args, axis=0, out=SharedMemoryArray(shape, dtype=dtype)).metadata

        return tree_lib.map_structure(
            stacking_function, input_records[0], *input_records[1:]
        )
