"""Time-series patch encoder."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from discretax.encoder.base import AbstractEncoder


class TimeSeriesPatchEncoder(AbstractEncoder):
    """Project fixed-length temporal patches into token embeddings."""

    linear: eqx.nn.Linear
    positional_embedding: jax.Array
    channel_embedding: jax.Array
    patch_length: int = eqx.field(static=True)
    patch_stride: int = eqx.field(static=True)
    sequence_length: int = eqx.field(static=True)
    num_patches: int = eqx.field(static=True)
    value_dim: int = eqx.field(static=True)
    num_time_features: int = eqx.field(static=True)
    channel_independent: bool = eqx.field(static=True)
    use_decomposition: bool = eqx.field(static=True)
    decomposition_kernel_size: int = eqx.field(static=True)
    output_length: int = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        key: PRNGKeyArray,
        *args,
        out_features: int,
        sequence_length: int,
        patch_length: int = 16,
        patch_stride: int | None = None,
        value_dim: int | None = None,
        num_time_features: int = 0,
        channel_independent: bool = False,
        decomposition_kernel_size: int | None = None,
        use_bias: bool = False,
        use_positional_embedding: bool = True,
        dtype: jnp.dtype = jnp.float32,
        **kwargs,
    ):
        """Initialize the time-series patch encoder."""
        del args, kwargs
        if patch_length <= 0:
            raise ValueError("patch_length must be positive")
        patch_stride = patch_length if patch_stride is None else patch_stride
        if patch_stride <= 0:
            raise ValueError("patch_stride must be positive")
        if sequence_length < patch_length:
            raise ValueError(
                f"sequence_length={sequence_length} must be at least patch_length={patch_length}"
            )
        value_dim = in_features if value_dim is None else value_dim
        if value_dim <= 0 or value_dim > in_features:
            raise ValueError(f"value_dim={value_dim} must be in the range [1, {in_features}]")
        if num_time_features < 0 or value_dim + num_time_features > in_features:
            raise ValueError(
                f"num_time_features={num_time_features} is incompatible with "
                f"in_features={in_features} and value_dim={value_dim}"
            )
        if decomposition_kernel_size is not None and decomposition_kernel_size <= 0:
            raise ValueError("decomposition_kernel_size must be positive when provided")
        remainder = (sequence_length - patch_length) % patch_stride
        if remainder != 0:
            raise ValueError(
                "sequence_length and patch settings must align exactly; got "
                f"sequence_length={sequence_length}, patch_length={patch_length}, "
                f"patch_stride={patch_stride}"
            )
        linear_key, pos_key, channel_key = jax.random.split(key, 3)
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.sequence_length = sequence_length
        self.num_patches = 1 + (sequence_length - patch_length) // patch_stride
        self.value_dim = value_dim
        self.num_time_features = num_time_features
        self.channel_independent = channel_independent
        self.use_decomposition = decomposition_kernel_size is not None
        self.decomposition_kernel_size = decomposition_kernel_size or 1
        self.output_length = (
            self.num_patches * self.value_dim if self.channel_independent else self.num_patches
        )
        token_value_dim = 1 if self.channel_independent else self.value_dim
        input_multiplier = 2 if self.use_decomposition else 1
        self.linear = eqx.nn.Linear(
            in_features=patch_length
            * (input_multiplier * token_value_dim + self.num_time_features),
            out_features=out_features,
            use_bias=use_bias,
            dtype=dtype,
            key=linear_key,
        )
        if use_positional_embedding:
            scale = 1.0 / jnp.sqrt(float(out_features))
            self.positional_embedding = (
                jax.random.normal(pos_key, (self.num_patches, out_features), dtype=dtype) * scale
            )
        else:
            self.positional_embedding = jnp.zeros((self.num_patches, out_features), dtype=dtype)
        if self.channel_independent:
            scale = 1.0 / jnp.sqrt(float(out_features))
            self.channel_embedding = (
                jax.random.normal(channel_key, (self.value_dim, out_features), dtype=dtype) * scale
            )
        else:
            self.channel_embedding = jnp.zeros((1, out_features), dtype=dtype)

    def _moving_average(self, values: Array) -> Array:
        """Compute a simple centered moving average over time."""
        kernel_size = self.decomposition_kernel_size
        if kernel_size <= 1:
            return values
        pad_left = (kernel_size - 1) // 2
        pad_right = kernel_size - 1 - pad_left
        padded = jnp.pad(values, ((pad_left, pad_right), (0, 0)), mode="edge")
        return jnp.stack(
            [jnp.mean(padded[i : i + kernel_size], axis=0) for i in range(values.shape[0])],
            axis=0,
        )

    def __call__(
        self, x: Array, state: eqx.nn.State, *, key: PRNGKeyArray | None = None
    ) -> tuple[Array, eqx.nn.State]:
        """Patchify a time series and project each patch to a token."""
        del key
        x = x.astype(self.linear.weight.dtype)
        value_inputs = x[:, : self.value_dim]
        aux_inputs = x[:, self.value_dim : self.value_dim + self.num_time_features]
        if self.use_decomposition:
            trend_inputs = self._moving_average(value_inputs)
            residual_inputs = value_inputs - trend_inputs
        else:
            residual_inputs = value_inputs
            trend_inputs = None

        patch_starts = range(
            0,
            self.sequence_length - self.patch_length + 1,
            self.patch_stride,
        )
        tokens: list[Array] = []
        if self.channel_independent:
            for channel_index in range(self.value_dim):
                for patch_index, start in enumerate(patch_starts):
                    pieces = [
                        residual_inputs[
                            start : start + self.patch_length,
                            channel_index : channel_index + 1,
                        ]
                    ]
                    if trend_inputs is not None:
                        pieces.append(
                            trend_inputs[
                                start : start + self.patch_length,
                                channel_index : channel_index + 1,
                            ]
                        )
                    if self.num_time_features > 0:
                        pieces.append(aux_inputs[start : start + self.patch_length])
                    token_inputs = jnp.concatenate(pieces, axis=-1).reshape(-1)
                    token = self.linear(token_inputs)
                    token = token + self.positional_embedding[patch_index]
                    token = token + self.channel_embedding[channel_index]
                    tokens.append(token)
        else:
            for patch_index, start in enumerate(patch_starts):
                pieces = [residual_inputs[start : start + self.patch_length]]
                if trend_inputs is not None:
                    pieces.append(trend_inputs[start : start + self.patch_length])
                if self.num_time_features > 0:
                    pieces.append(aux_inputs[start : start + self.patch_length])
                token_inputs = jnp.concatenate(pieces, axis=-1).reshape(-1)
                token = self.linear(token_inputs)
                token = token + self.positional_embedding[patch_index]
                tokens.append(token)
        return jnp.stack(tokens, axis=0), state
