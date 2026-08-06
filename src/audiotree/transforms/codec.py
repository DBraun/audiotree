"""Transforms that encode audio with a neural audio codec.

These transforms take a *codec object* — any object implementing the relevant
protocol (:class:`AudioCodec` or :class:`LatentAudioCodec`) — rather than a
bare function. The codec owns its input conventions (resampling to its own
rate, channel handling) and its output shapes; the transforms simply store what
the codec returns on the :class:`~audiotree.AudioTree`.

Both transforms are deterministic ``Map`` transforms intended for on-device
(JIT) pipelines. They call the codec directly with no internal ``jax.jit``:
the caller decides where the JIT boundary is.
"""

from typing import Callable, List, Optional, Protocol, Tuple, Union, runtime_checkable

from jax.typing import ArrayLike

from audiotree import AudioTree
from audiotree.transforms.decorators import map_transform

#: A ``scope`` argument: a list of paths into a dict element (``["wet"]``,
#: ``["input.dry"]``) or the nested-dict spelling. See
#: :func:`~audiotree.transforms.base.normalize_scope`.
Scope = Optional[Union[list, tuple, dict]]

#: An ``output_key`` argument: a literal key, or a function from the leaf's
#: full path to the key to write it under.
OutputKey = Optional[Union[str, Callable[[List[str]], str]]]


@runtime_checkable
class AudioCodec(Protocol):
    """Structural protocol for codecs that encode audio to discrete codes.

    :meth:`encode` takes an :class:`~audiotree.AudioTree` and returns
    ``(codes, scale)``. ``codes`` is an integer array whose shape convention
    (e.g. ``(batch, codebooks, frames)``) is defined by the codec; ``scale`` is
    an optional loudness-normalization factor (or ``None`` for codecs that
    don't rescale).

    The codec is responsible for resampling the input to its own sample rate
    and for its channel handling (e.g. folding stereo into the batch for a
    mono encoder).

    This protocol is ``runtime_checkable``, so ``isinstance(codec, AudioCodec)``
    reports whether an object supplies ``encode``. It is deliberately separate
    from :class:`LatentAudioCodec`: a codec that only produces codes satisfies
    this protocol on its own, and one that does both satisfies both.
    """

    def encode(self, audio: AudioTree) -> Tuple[ArrayLike, Optional[ArrayLike]]: ...


@runtime_checkable
class LatentAudioCodec(Protocol):
    """Structural protocol for codecs that encode audio to continuous latents.

    :meth:`encode_to_latent` takes an :class:`~audiotree.AudioTree` and returns
    a latent array, whose shape convention is defined by the codec. As with
    :class:`AudioCodec`, the codec owns resampling and channel handling, and
    the protocol is ``runtime_checkable``.
    """

    def encode_to_latent(self, audio: AudioTree) -> ArrayLike: ...


def encode_with_codec(
    codec: AudioCodec,
    *,
    scope: Scope = None,
    output_key: OutputKey = None,
):
    """Create a transform that encodes audio to discrete codes.

    Calls ``codec.encode(audio_tree)`` and stores the returned ``codes`` on
    ``AudioTree.codes`` exactly as the codec produced them (the codec defines
    the shape convention). If the codec returns a non-``None`` ``scale``, it
    is stored under ``metadata["codec_scale"]`` so the codes can later be
    decoded faithfully. AudioTrees that already have ``codes`` pass through
    unchanged.

    Args:
        codec: Object implementing ``encode(AudioTree) -> (codes, scale)``
            (see :class:`AudioCodec`).
        scope: Which leaves of a dict-of-AudioTree element to transform.
            Defaults to ``None`` (all of them).
        output_key: Write the result under a new key instead of replacing the
            input.

    Returns:
        Transform for use with ``.map()``.

    Example:
        transform = encode_with_codec(codec)
        ds = ds.map(transform)
    """

    @map_transform
    def _encode_with_codec_transform(audio_tree: AudioTree) -> AudioTree:
        if audio_tree.codes is not None:
            return audio_tree
        codes, scale = codec.encode(audio_tree)
        metadata = audio_tree.metadata
        if scale is not None:
            metadata = {**metadata, "codec_scale": scale}
        return audio_tree.replace(codes=codes, metadata=metadata)

    return _encode_with_codec_transform(scope=scope, output_key=output_key)


def encode_latents(
    codec: LatentAudioCodec,
    *,
    scope: Scope = None,
    output_key: OutputKey = None,
):
    """Create a transform that encodes audio to continuous latents.

    Calls ``codec.encode_to_latent(audio_tree)`` and stores the result on
    ``AudioTree.latents``. AudioTrees that already have ``latents`` pass
    through unchanged.

    Args:
        codec: Object implementing ``encode_to_latent(AudioTree) -> latents``
            (see :class:`LatentAudioCodec`).
        scope: Which leaves of a dict-of-AudioTree element to transform.
            Defaults to ``None`` (all of them).
        output_key: Write the result under a new key instead of replacing the
            input.

    Returns:
        Transform for use with ``.map()``.

    Example:
        transform = encode_latents(codec)
        ds = ds.map(transform)
    """

    @map_transform
    def _encode_latents_transform(audio_tree: AudioTree) -> AudioTree:
        if audio_tree.latents is not None:
            return audio_tree
        latents = codec.encode_to_latent(audio_tree)
        return audio_tree.replace(latents=latents)

    return _encode_latents_transform(scope=scope, output_key=output_key)
