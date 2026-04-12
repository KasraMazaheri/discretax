"""Regression head."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from discretax.heads.base import AbstractHead


class RegressionHead(AbstractHead):
    """Regression head.

    Takes an input of shape (timesteps, in_features) and produces either:
    - A single vector of shape (out_features,) when reduce=True (global average pooling).
    - A sequence of shape (timesteps // output_step, out_features) when reduce=False,
      optionally subsampled by output_step before projection.

    Attributes:
        linear: Linear projection layer.
        reduce: Whether to reduce the time dimension by averaging before projection.
        output_step: Stride for subsampling the time dimension when reduce=False.
            E.g. output_step=128 keeps every 128th hidden state, yielding
            sequence_length // 128 predictions.
    """

    linear: eqx.nn.Linear
    reduce: bool
    output_step: int

    def __init__(
        self,
        in_features: int,
        out_features: int,
        key: PRNGKeyArray,
        *args,
        reduce: bool = True,
        output_step: int = 1,
        dtype: jnp.dtype = jnp.float32,
        **kwargs,
    ):
        """Initialize the regression head.

        Args:
            in_features: Input feature dimension.
            out_features: Output feature dimension (prediction dimension per timestep).
            key: JAX random key for initialization.
            reduce: If True, average-pool the time axis before projecting.
            output_step: Stride for subsampling along the time axis when reduce=False.
                Has no effect when reduce=True.
            dtype: dtype for the linear projection.
            *args: Additional positional arguments (ignored).
            **kwargs: Additional keyword arguments (ignored).
        """
        self.linear = eqx.nn.Linear(
            in_features=in_features,
            out_features=out_features,
            dtype=dtype,
            key=key,
        )
        self.reduce = reduce
        self.output_step = output_step

    def __call__(
        self, x: Array, state: eqx.nn.State, *, key: PRNGKeyArray | None = None
    ) -> tuple[Array, eqx.nn.State]:
        """Forward pass of the regression head.

        Args:
            x: Input tensor of shape (timesteps, in_features).
            state: Current state for stateful layers.
            key: JAX random key for stochastic operations (unused).

        Returns:
            Tuple of (output, state). Output shape is (out_features,) when reduce=True,
            or (timesteps // output_step, out_features) when reduce=False.
        """
        x = x.astype(self.linear.weight.dtype)
        if self.reduce:
            x = jnp.mean(x, axis=0)
        else:
            x = x[self.output_step - 1 :: self.output_step]
        x = jax.vmap(self.linear)(x) if x.ndim == 2 else self.linear(x)
        return x, state
