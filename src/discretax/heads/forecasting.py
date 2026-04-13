"""Sequence forecasting head."""

from __future__ import annotations

import equinox as eqx
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
