"""Sequence forecasting heads."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from discretax.heads.base import AbstractHead


class SequenceForecastHead(AbstractHead):
    """Project an encoded sequence into a future prediction horizon."""

    linear: eqx.nn.Linear
    input_length: int
    prediction_length: int
    target_dim: int

    def __init__(
        self,
        in_features: int,
        out_features: int,
        key: PRNGKeyArray,
        *args,
        input_length: int,
        prediction_length: int,
        dtype: jnp.dtype = jnp.float32,
        **kwargs,
    ):
        """Initialize the sequence forecasting head."""
        del args, kwargs
        self.linear = eqx.nn.Linear(
            in_features=input_length * in_features,
            out_features=prediction_length * out_features,
            dtype=dtype,
            key=key,
        )
        self.input_length = input_length
        self.prediction_length = prediction_length
        self.target_dim = out_features

    def __call__(
        self, x: Array, state: eqx.nn.State, *, key: PRNGKeyArray | None = None
    ) -> tuple[Array, eqx.nn.State]:
        """Project a hidden sequence into forecasting targets."""
        del key
        x = x.astype(self.linear.weight.dtype)
        x = x.reshape(self.input_length * x.shape[-1])
        x = self.linear(x)
        x = x.reshape(self.prediction_length, self.target_dim)
        return x, state


class TemporalProjectionForecastHead(AbstractHead):
    """Forecast by projecting along time, then decoding each predicted step."""

    time_projection: eqx.nn.Linear
    output_projection: eqx.nn.Linear
    input_length: int
    prediction_length: int
    target_dim: int

    def __init__(
        self,
        in_features: int,
        out_features: int,
        key: PRNGKeyArray,
        *args,
        input_length: int,
        prediction_length: int,
        dtype: jnp.dtype = jnp.float32,
        **kwargs,
    ):
        """Initialize the temporal projection forecasting head."""
        del args, kwargs
        time_key, output_key = jax.random.split(key)
        self.time_projection = eqx.nn.Linear(
            in_features=input_length,
            out_features=prediction_length,
            dtype=dtype,
            key=time_key,
        )
        self.output_projection = eqx.nn.Linear(
            in_features=in_features,
            out_features=out_features,
            dtype=dtype,
            key=output_key,
        )
        self.input_length = input_length
        self.prediction_length = prediction_length
        self.target_dim = out_features

    def __call__(
        self, x: Array, state: eqx.nn.State, *, key: PRNGKeyArray | None = None
    ) -> tuple[Array, eqx.nn.State]:
        """Project encoded features across time, then decode each predicted step."""
        del key
        x = x.astype(self.time_projection.weight.dtype)
        x = jnp.swapaxes(x, 0, 1)  # [hidden, time]
        x = jax.vmap(self.time_projection)(x)  # [hidden, pred]
        x = jnp.swapaxes(x, 0, 1)  # [pred, hidden]
        x = jax.vmap(self.output_projection)(x)  # [pred, target]
        return x, state


class ChannelIndependentTemporalProjectionForecastHead(AbstractHead):
    """Forecast independently per channel from channel-major patch tokens."""

    time_projection: eqx.nn.Linear
    output_projection: eqx.nn.Linear
    input_length: int
    prediction_length: int
    target_dim: int
    num_channels: int = eqx.field(static=True)
    num_patches: int = eqx.field(static=True)
    target_indices: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        key: PRNGKeyArray,
        *args,
        input_length: int,
        prediction_length: int,
        num_channels: int,
        target_indices: tuple[int, ...] | list[int] | None = None,
        dtype: jnp.dtype = jnp.float32,
        **kwargs,
    ):
        """Initialize the channel-independent temporal forecasting head."""
        del args, kwargs
        if num_channels <= 0:
            raise ValueError("num_channels must be positive")
        if input_length % num_channels != 0:
            raise ValueError(
                f"input_length={input_length} must be divisible by num_channels={num_channels}"
            )
        time_key, output_key = jax.random.split(key)
        self.num_channels = num_channels
        self.num_patches = input_length // num_channels
        if target_indices is None:
            target_indices = tuple(range(out_features))
        self.target_indices = tuple(int(index) for index in target_indices)
        self.time_projection = eqx.nn.Linear(
            in_features=self.num_patches,
            out_features=prediction_length,
            dtype=dtype,
            key=time_key,
        )
        self.output_projection = eqx.nn.Linear(
            in_features=in_features,
            out_features=1,
            dtype=dtype,
            key=output_key,
        )
        self.input_length = input_length
        self.prediction_length = prediction_length
        self.target_dim = out_features

    def __call__(
        self, x: Array, state: eqx.nn.State, *, key: PRNGKeyArray | None = None
    ) -> tuple[Array, eqx.nn.State]:
        """Forecast each selected channel independently from patch tokens."""
        del key
        x = x.astype(self.time_projection.weight.dtype)
        x = x.reshape(self.num_channels, self.num_patches, x.shape[-1])
        x = x[jnp.asarray(self.target_indices, dtype=jnp.int32)]
        x = jnp.swapaxes(x, 1, 2)  # [target, hidden, patches]
        x = jax.vmap(lambda channel_x: jax.vmap(self.time_projection)(channel_x))(x)
        x = jnp.swapaxes(x, 1, 2)  # [target, pred, hidden]
        x = jax.vmap(lambda channel_x: jax.vmap(self.output_projection)(channel_x))(x)
        x = jnp.squeeze(x, axis=-1)
        x = jnp.swapaxes(x, 0, 1)
        return x, state
