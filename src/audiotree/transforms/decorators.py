"""Decorators for creating transforms from functions."""

import inspect
from functools import wraps
from typing import Any, Callable, Dict

from audiotree.transforms.base import BaseRandomTransform, BaseMapTransform


def random_transform(fn: Callable) -> Callable:
    """Decorator to create a RandomMapTransform from a function.

    The decorated function should have signature:
        fn(audio_tree: AudioTree, rng, **params) -> AudioTree

    The decorator automatically handles:
    - prob: Probability of applying transform
    - scope: Which PyTree leaves to transform
    - output_key: Where to store transformed output
    - split_seed: Whether to use different RNG per leaf

    Usage:
        @random_transform
        def volume_norm(audio_tree, rng, min_db=-20.0, max_db=-15.0):
            # transform logic
            return audio_tree

        # Create transform instance
        transform = volume_norm(min_db=-30, max_db=-10, prob=0.9)
        ds = ds.random_map(transform, seed=42)

        # With argbind
        VolumeNorm = argbind.bind(volume_norm)
        with argbind.scope(args):
            transform = VolumeNorm()
            ds = ds.random_map(transform, seed=42)
    """

    # Extract default values from function signature
    # Exclude 'audio_tree' and 'rng' from the config
    sig = inspect.signature(fn)
    param_defaults = {}
    for name, param in sig.parameters.items():
        if name not in ['audio_tree', 'rng'] and param.default != inspect.Parameter.empty:
            param_defaults[name] = param.default

    @wraps(fn)
    def wrapper(
        prob: float = 1.0,
        split_seed: bool = True,
        scope: Dict[str, Any] = None,
        output_key = None,
        **transform_params
    ):
        """Create a transform instance with given parameters."""

        config = {**param_defaults, **transform_params}

        class FunctionBasedTransform(BaseRandomTransform):
            """Transform created from decorated function."""

            @staticmethod
            def get_default_config():
                return param_defaults.copy()

            @staticmethod
            def _apply_transform(audio_tree, rng, **params):
                return fn(audio_tree, rng, **params)

        return FunctionBasedTransform(
            config=config,
            prob=prob,
            split_seed=split_seed,
            scope=scope,
            output_key=output_key,
        )

    # Preserve original function's signature for argbind
    # Add the special parameters
    params = list(sig.parameters.values())
    # Remove audio_tree and rng from the exposed signature
    params = [p for p in params if p.name not in ['audio_tree', 'rng']]

    # Add special parameters
    special_params = [
        inspect.Parameter('prob', inspect.Parameter.KEYWORD_ONLY, default=1.0),
        inspect.Parameter('split_seed', inspect.Parameter.KEYWORD_ONLY, default=True),
        inspect.Parameter('scope', inspect.Parameter.KEYWORD_ONLY, default=None),
        inspect.Parameter('output_key', inspect.Parameter.KEYWORD_ONLY, default=None),
    ]

    wrapper.__signature__ = inspect.Signature(params + special_params)
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__

    return wrapper


def map_transform(fn: Callable) -> Callable:
    """Decorator to create a MapTransform from a function.

    The decorated function should have signature:
        fn(audio_tree: AudioTree, **params) -> AudioTree

    The decorator automatically handles:
    - scope: Which PyTree leaves to transform
    - output_key: Where to store transformed output

    Usage:
        @map_transform
        def trim(audio_tree, length=1.0):
            # transform logic
            return audio_tree

        # Create transform instance
        transform = trim(length=3.0)
        ds = ds.map(transform)

        # With argbind
        Trim = argbind.bind(trim)
        with argbind.scope(args):
            transform = Trim()
            ds = ds.map(transform)
    """

    # Extract default values from function signature
    # Exclude 'audio_tree' from the config
    sig = inspect.signature(fn)
    param_defaults = {}
    for name, param in sig.parameters.items():
        if name != 'audio_tree' and param.default != inspect.Parameter.empty:
            param_defaults[name] = param.default

    @wraps(fn)
    def wrapper(
        scope: Dict[str, Any] = None,
        output_key = None,
        **transform_params
    ):
        """Create a transform instance with given parameters."""

        config = {**param_defaults, **transform_params}

        class FunctionBasedTransform(BaseMapTransform):
            """Transform created from decorated function."""

            @staticmethod
            def get_default_config():
                return param_defaults.copy()

            @staticmethod
            def _apply_transform(audio_tree, **params):
                return fn(audio_tree, **params)

        return FunctionBasedTransform(
            config=config,
            scope=scope,
            output_key=output_key,
        )

    # Preserve original function's signature for argbind
    params = list(sig.parameters.values())
    # Remove audio_tree from the exposed signature
    params = [p for p in params if p.name != 'audio_tree']

    # Add special parameters
    special_params = [
        inspect.Parameter('scope', inspect.Parameter.KEYWORD_ONLY, default=None),
        inspect.Parameter('output_key', inspect.Parameter.KEYWORD_ONLY, default=None),
    ]

    wrapper.__signature__ = inspect.Signature(params + special_params)
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__

    return wrapper
