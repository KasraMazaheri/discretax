"""Sequence Mixer for LinOSS-IM, LinOSS-IMEX, and Damped LinOSS-IMEX models.

See: https://openreview.net/pdf?id=GRMfXcAAFh
"""

from __future__ import annotations

import math
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import nn, random
from jax.nn.initializers import normal
from jaxtyping import Array, PRNGKeyArray

from discretax.sequence_mixers.base import AbstractSequenceMixer


class LinOSSSequenceMixer(AbstractSequenceMixer):
    """LinOSS sequence mixer layer.

    This layer implements the LinOSS sequence mixer.

    Attributes:
        A_diag: Diagonal state matrix.
        G_diag: Diagonal damping matrix.
        B: Input matrix.
        C: Output matrix.
        D: Output matrix.
        steps: Learnable step sizes for the sequence mixer (parameterized via sigmoid).
        discretization: Discretization method to use.
        damping: Whether to use damping.
    """

    A_diag: jax.Array
    G_diag: jax.Array
    B: jax.Array
    C: jax.Array
    D: jax.Array
    steps: jax.Array

    # non learnable static fields
    discretization: Literal["IM", "IMEX"] = eqx.field(static=True)
    damping: bool = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        key: PRNGKeyArray,
        *args,
        state_dim: int = 64,
        discretization: Literal["IM", "IMEX"] = "IMEX",
        damping: bool = True,
        r_min: float = 0.9,
        theta_max: float = jnp.pi,
        dtype: jnp.dtype = jnp.float32,
        **kwargs,
    ):
        """Initialize the LinOSS sequence mixer layer.

        Args:
            in_features: dimension of the input features.
            key: JAX random key for initialization.
            state_dim: dimension of the state space.
            discretization: discretization method to use.
            damping: whether to use damping.
            r_min: minimum value for the radius.
            theta_max: maximum value for the theta parameter.
            dtype: dtype for sequence mixer parameters and computation.
            *args: Additional positional arguments (ignored).
            **kwargs: Additional keyword arguments (ignored).
        """
        dtype = jnp.dtype(dtype)
        A_key, G_key, B_key, C_key, D_key, step_key, key = jr.split(key, 7)

        self.steps = normal(stddev=0.5)(step_key, (state_dim,), dtype=dtype)
        steps = nn.sigmoid(self.steps)

        if discretization == "IMEX" and damping:
            r_max = 1.0
            mags = jnp.sqrt(
                random.uniform(G_key, shape=(state_dim,), dtype=dtype) * (r_max**2 - r_min**2)
                + r_min**2
            )
            self.G_diag = ((1 - mags**2) / (steps * mags**2)).astype(dtype)
            G_diag = nn.relu(self.G_diag)

            theta = random.uniform(A_key, shape=(state_dim,), dtype=dtype) * theta_max
            self.A_diag = _map_theta_to_A(theta, G_diag, steps).astype(dtype)
        else:
            self.G_diag = None
            self.A_diag = random.uniform(A_key, shape=(state_dim,), dtype=dtype)

        self.B = _simple_uniform_init(
            B_key,
            shape=(state_dim, in_features, 2),
            std=1.0 / math.sqrt(in_features),
            dtype=dtype,
        )
        self.C = _simple_uniform_init(
            C_key,
            shape=(in_features, state_dim, 2),
            std=1.0 / math.sqrt(state_dim),
            dtype=dtype,
        )
        self.D = normal(stddev=1.0)(D_key, (in_features,), dtype=dtype)

        self.discretization = discretization
        self.damping = damping

    def __call__(self, x: Array, key: PRNGKeyArray) -> Array:
        """Forward pass of the LinOSS sequence mixer layer.

        Args:
            x: Input sequence of features.
            key: JAX random key for initialization.

        Returns:
            The output of the LinOSS sequence mixer.
        """
        steps = nn.sigmoid(self.steps)

        if self.discretization == "IM":
            if self.damping:
                raise NotImplementedError(
                    "Discretization {} and damping = {} not implemented".format(
                        self.discretization, self.damping
                    )
                )
            else:
                A_diag = nn.relu(self.A_diag)
                ys = _apply_linoss_im(A_diag, self.B, x, steps)
        elif self.discretization == "IMEX":
            if self.damping:
                G_diag = nn.relu(self.G_diag)
                A_boundary_low = (2 + steps * G_diag - 2 * jnp.sqrt(1 + steps * G_diag)) / steps**2
                A_boundary_high = (
                    2 + steps * G_diag + 2 * jnp.sqrt(1 + steps * G_diag)
                ) / steps**2
                A_diag = (
                    A_boundary_low
                    + nn.relu(self.A_diag - A_boundary_low)
                    - nn.relu(self.A_diag - A_boundary_high)
                )
                ys = _apply_damped_linoss_imex(A_diag, G_diag, self.B, x, steps)
            else:
                A_diag = nn.relu(self.A_diag)
                ys = _apply_linoss_imex(A_diag, self.B, x, steps)
        else:
            raise NotImplementedError(f"Discretization {self.discretization} not implemented")

        # Apply SequenceMixer Output Operations Cx + Du
        Cy = jnp.einsum("hs,ls->lh", self.C[..., 0], ys[..., 0]) - jnp.einsum(
            "hs,ls->lh",
            self.C[..., 1],
            ys[..., 1],
        )
        Du = jax.vmap(lambda u: self.D * u)(x)
        xs = Cy + Du

        return xs


def _simple_uniform_init(rng, shape, std=1.0, dtype=jnp.float32):
    """Simple uniform initialization.

    Args:
        rng: JAX random key for initialization.
        shape: Shape of the weights.
        std: Standard deviation of the weight initialization.
        dtype: dtype of the initialized weights.

    Returns:
        Weights initialized using a simple uniform distribution.
    """
    weights = random.uniform(rng, shape, dtype=dtype) * 2.0 * std - std
    return weights


def _map_theta_to_A(thetas, G_diag, steps):  # noqa: N802
    """Map theta parameter to diagonal state matrix A.

    Args:
        thetas: Theta parameter values.
        G_diag: Diagonal damping matrix.
        steps: Discretization time-steps.

    Returns:
        Diagonal state matrix A computed from the input parameters.
    """
    A_plus = (
        4
        * jnp.sqrt(
            steps**4 * jnp.cos(thetas) ** (-2) + steps**5 * G_diag * jnp.cos(thetas) ** (-2)
        )
        - steps**2
        * (
            -4
            - 2 * steps * G_diag
            - 4 * jnp.tan(thetas) ** 2
            - 2 * steps * G_diag * jnp.tan(thetas) ** 2
        )
    ) / (2 * steps**4 * (1 + jnp.tan(thetas) ** 2))
    A_minus = (
        -4
        * jnp.sqrt(
            steps**4 * jnp.cos(thetas) ** (-2) + steps**5 * G_diag * jnp.cos(thetas) ** (-2)
        )
        - steps**2
        * (
            -4
            - 2 * steps * G_diag
            - 4 * jnp.tan(thetas) ** 2
            - 2 * steps * G_diag * jnp.tan(thetas) ** 2
        )
    ) / (2 * steps**4 * (1 + jnp.tan(thetas) ** 2))

    A_diag = jnp.where(thetas > jnp.pi / 2, A_plus, A_minus)

    return A_diag


# Parallel scan operations
@jax.vmap
def _binary_operator(q_i, q_j):  # noqa: N802
    """Binary operator for parallel scan of linear recurrence.

    Args:
        q_i: Tuple containing A_i and b_i at position i.
        q_j: Tuple containing A_j and b_j at position j.

    Returns:
        The binary operator applied to the input.
    """
    A_i, b_i = q_i
    A_j, b_j = q_j

    N = A_i.size // 4
    iA_ = A_i[0 * N : 1 * N]
    iB_ = A_i[1 * N : 2 * N]
    iC_ = A_i[2 * N : 3 * N]
    iD_ = A_i[3 * N : 4 * N]
    jA_ = A_j[0 * N : 1 * N]
    jB_ = A_j[1 * N : 2 * N]
    jC_ = A_j[2 * N : 3 * N]
    jD_ = A_j[3 * N : 4 * N]
    A_new = jA_ * iA_ + jB_ * iC_
    B_new = jA_ * iB_ + jB_ * iD_
    C_new = jC_ * iA_ + jD_ * iC_
    D_new = jC_ * iB_ + jD_ * iD_
    Anew = jnp.concatenate([A_new, B_new, C_new, D_new])

    b_i1 = b_i[0:N, :]
    b_i2 = b_i[N:, :]

    new_b1 = jA_[:, None] * b_i1 + jB_[:, None] * b_i2
    new_b2 = jC_[:, None] * b_i1 + jD_[:, None] * b_i2
    new_b = jnp.concatenate([new_b1, new_b2])

    return Anew, new_b + b_j


def _apply_linoss_scan(
    M_11: Array,
    M_12: Array,
    M_21: Array,
    M_22: Array,
    F1: Array,
    F2: Array,
) -> Array:
    """Run the shared LinOSS scan for paired-real forcing terms."""
    M = jnp.concatenate([M_11, M_12, M_21, M_22])
    M_elements = jnp.broadcast_to(M, (F1.shape[0], 4 * M_11.shape[0]))
    F = jnp.concatenate([F1, F2], axis=1)
    _, xs = jax.lax.associative_scan(_binary_operator, (M_elements, F))
    return xs[:, M_11.shape[0] :, :]


def _apply_linoss_im(A_diag, B, x, step):  # noqa: N802
    """Compute the LinOSS-IM recurrence.

    Args:
        A_diag: Diagonal state matrix.
        B: Input matrix.
        x: Input sequence of features.
        step: Discretization time-step.

    Returns:
        Hidden state sequence, shape (timesteps, state_dim).
    """
    Bu_elements = jnp.einsum("sfd,lf->lsd", B, x)

    schur_comp = 1.0 / (1.0 + step**2.0 * A_diag)
    M_11 = 1.0 - step**2.0 * A_diag * schur_comp
    M_12 = -1.0 * step * A_diag * schur_comp
    M_21 = step * schur_comp
    M_22 = schur_comp

    F1 = Bu_elements * (M_11 * step)[None, :, None]
    F2 = Bu_elements * (M_21 * step)[None, :, None]
    return _apply_linoss_scan(M_11, M_12, M_21, M_22, F1, F2)


def _apply_linoss_imex(A_diag, B, x, step):  # noqa: N802
    """Compute the LinOSS-IMEX recurrence.

    Args:
        A_diag: Diagonal state matrix.
        B: Input matrix.
        x: Input sequence of features.
        step: Discretization time-step.

    Returns:
        Hidden state sequence, shape (timesteps, state_dim).
    """
    Bu_elements = jnp.einsum("sfd,lf->lsd", B, x)

    A_ = jnp.ones_like(A_diag)
    B_ = -1.0 * step * A_diag
    C_ = step
    D_ = 1.0 - (step**2.0) * A_diag

    F1 = Bu_elements * step[None, :, None]
    F2 = Bu_elements * (step**2.0)[None, :, None]
    return _apply_linoss_scan(A_, B_, C_, D_, F1, F2)


def _apply_damped_linoss_imex(A_diag, G_diag, B, x, step):  # noqa: N802
    """Compute the Damped LinOSS-IMEX recurrence.

    Args:
        A_diag: Diagonal state matrix.
        G_diag: Diagonal damping matrix.
        B: Input matrix.
        x: Input sequence of features.
        step: Discretization time-step.

    Returns:
        Hidden state sequence, shape (timesteps, state_dim).
    """
    Bu_elements = jnp.einsum("sfd,lf->lsd", B, x)

    Identity = jnp.ones_like(A_diag)
    S = Identity + step * G_diag
    M_11 = 1.0 / S
    M_12 = -step / S * A_diag
    M_21 = step / S
    M_22 = Identity - step**2 / S * A_diag

    F1 = Bu_elements * (step * (1.0 / S))[None, :, None]
    F2 = Bu_elements * (step**2 * (1.0 / S))[None, :, None]
    return _apply_linoss_scan(M_11, M_12, M_21, M_22, F1, F2)
