"""Patch-based image encoder."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from discretax.encoder.base import AbstractEncoder


class ImagePatchEncoder(AbstractEncoder):
    """Patchify an image sequence into hidden-dimension tokens."""

    patch_embedding: eqx.nn.Conv2d
    norm: eqx.nn.LayerNorm | None
    image_shape: tuple[int, int, int]
    sequence_layout: str
    patch_size: int

    def __init__(
        self,
        key: PRNGKeyArray,
        *args,
        out_features: int,
        image_shape: tuple[int, int, int],
        sequence_layout: str = "rows",
        patch_size: int = 4,
        use_bias: bool = False,
        use_layer_norm: bool = True,
        dtype: jnp.dtype = jnp.float32,
        **kwargs,
    ):
        """Initialize the image patch encoder."""
        del args, kwargs
        height, width, channels = image_shape
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(
                "image_shape must be divisible by patch_size, got "
                f"image_shape={image_shape} and patch_size={patch_size}"
            )
        self.patch_embedding = eqx.nn.Conv2d(
            in_channels=channels,
            out_channels=out_features,
            kernel_size=patch_size,
            stride=patch_size,
            use_bias=use_bias,
            dtype=dtype,
            key=key,
        )
        self.norm = eqx.nn.LayerNorm(out_features) if use_layer_norm else None
        self.image_shape = tuple(int(value) for value in image_shape)
        self.sequence_layout = sequence_layout
        self.patch_size = patch_size

    def __call__(
        self, x: Array, state: eqx.nn.State, *, key: PRNGKeyArray | None = None
    ) -> tuple[Array, eqx.nn.State]:
        """Project an image sequence to patch tokens."""
        del key
        image = _restore_image(
            x,
            image_shape=self.image_shape,
            sequence_layout=self.sequence_layout,
        ).astype(self.patch_embedding.weight.dtype)
        image = jnp.transpose(image, (2, 0, 1))
        tokens = self.patch_embedding(image)
        tokens = jnp.transpose(tokens, (1, 2, 0)).reshape(-1, tokens.shape[0])
        if self.norm is not None:
            tokens = jax.vmap(self.norm)(tokens)
        return tokens, state


def _restore_image(
    x: Array,
    *,
    image_shape: tuple[int, int, int],
    sequence_layout: str,
) -> Array:
    """Reconstruct an HWC image from a sequence-major representation."""
    height, width, channels = image_shape
    if sequence_layout == "rows":
        return x.reshape(height, width, channels)
    if sequence_layout == "pixels":
        return x.reshape(height, width, channels)
    raise ValueError(f"Unsupported image sequence layout: {sequence_layout}")
