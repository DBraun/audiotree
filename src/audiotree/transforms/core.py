"""Core transform classes.

This module contains only the Batch operation, which is kept for compatibility
with grain's BatchOperation pattern.

All other transforms are in the function-based API in functional.py.
"""

from typing import Any, Sequence

from jax import numpy as jnp, tree_util
import numpy as np

from grain.python import BatchOperation, SharedMemoryArray


class Batch(BatchOperation):
    """Concatenates AudioTree objects along the batch axis instead of stacking.

    Use this with grain's DataLoader API. For the IterDataset API, use
    ``AudioTree.batch`` instead.

    Example:
        .. code-block:: python

            from audiotree.transforms import Batch
            import grain

            dataloader = grain.DataLoader(
                data_source=ds,
                sampler=grain.samplers.IndexSampler(
                    num_records=len(ds),
                    shuffle=True,
                    seed=0,
                    shard_options=grain.NoSharding(),
                ),
                operations=[Batch(batch_size=32)],
            )

    See Also:
        ``AudioTree.batch`` for use with ``IterDataset.batch()``.
    """

    def __post_init__(self):
        super().__post_init__()
        self._display_deprecation_message = False

    def _default_batch_fn(self, input_records: Sequence[Any]):
        """Batches records together and copies Numpy arrays to Shared Memory."""
        from audiotree.core import AudioTree, _batch_audiotrees

        self._validate_structure(input_records)

        def stacking_function(*args):
            first_arg = args[0]
            if isinstance(first_arg, AudioTree):
                return _batch_audiotrees(args)
            if not isinstance(first_arg, (np.ndarray, jnp.ndarray)):
                return list(args)
            first_arg = np.asanyarray(first_arg)
            shape, dtype = (
                (len(args) * first_arg.shape[0],) + first_arg.shape[1:],
                first_arg.dtype,
            )
            if not self._use_shared_memory or dtype.hasobject:
                return np.concatenate(args, axis=0)
            return np.concatenate(
                args, axis=0, out=SharedMemoryArray(shape, dtype=dtype)
            ).metadata

        return tree_util.tree_map(
            stacking_function,
            input_records[0],
            *input_records[1:],
            is_leaf=lambda x: isinstance(x, AudioTree),
        )
