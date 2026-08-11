"""Pytest configuration shared by the whole test suite.

Grain 0.2.17+ reads absl flags (e.g.
``--grain_enable_multiprocess_worker_profiling``) from inside its
multiprocessing prefetch path. Pytest is not an ``absl.app`` entry point, so
those flags are never parsed and the access raises
``absl.flags.UnparsedFlagAccessError``. Marking the flags as parsed resolves
them to their defaults — the same call any plain script using grain
multiprocessing without ``absl.app.run`` would need to make.
"""

from absl import flags

flags.FLAGS.mark_as_parsed()
