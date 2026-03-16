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
import sympy as sp
from jax import nn, random
from jax.nn.initializers import normal
from jaxtyping import Array, PRNGKeyArray

from discretax.sequence_mixers.base import AbstractSequenceMixer

# --- LinOSSSequenceMixer ----------------------------------


class LinOSSSequenceMixer(AbstractSequenceMixer):
    """LinOSS sequence mixer layer.

    Implements the LinOSS-IM, LinOSS-IMEX, and Damped LinOSS variants.

    Attributes:
        A_diag: Diagonal state matrix.
        G_diag: Diagonal damping matrix (``None`` when ``damping=False``).
        B: Input matrix.
        C: Output matrix.
        D: Skip connection matrix.
        steps: Learnable step sizes for the sequence mixer (parameterized via sigmoid).
        discretization: Discretization method to use.
        initialization: Initialization strategy for damped variants (``"AG"`` or ``"RT"``).
        damping: Whether to use damping.
    """

    A_diag: jax.Array
    G_diag: jax.Array
    B: jax.Array
    C: jax.Array
    D: jax.Array
    steps: jax.Array
    head_gate: eqx.nn.Linear | None
    head_output_projection: eqx.nn.Linear | None

    # non learnable static fields
    discretization: Literal["IM", "IMEX", "IMEX2", "IMEX3", "EX"] = eqx.field(static=True)
    initialization: Literal["RT", "AG"] = eqx.field(static=True)
    damping: bool = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    head_hidden_dim: int = eqx.field(static=True)
    head_state_dim: int = eqx.field(static=True)
    use_head_gating: bool = eqx.field(static=True)
    use_head_output_projection: bool = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        key: PRNGKeyArray,
        *args,
        state_dim: int = 64,
        discretization: Literal["IM", "IMEX", "IMEX2", "IMEX3", "EX"] = "IMEX",
        initialization: Literal["RT", "AG"] = "AG",
        damping: bool = True,
        r_min: float = 0.9,
        theta_max: float = jnp.pi,
        num_heads: int = 1,
        use_head_gating: bool = False,
        use_head_output_projection: bool = False,
        A_max: float = 1.0,
        G_max: float = 1.0,
        dtype: jnp.dtype = jnp.float32,
        **kwargs,
    ):
        """Initialize the LinOSS sequence mixer layer.

        Args:
            in_features: dimension of the input features.
            key: JAX random key for initialization.
            state_dim: dimension of the state space.
            discretization: discretization method to use.
            initialization: initialization strategy for damped variants.
            damping: whether to use damping.
            r_min: minimum value for the radius.
            theta_max: maximum value for the theta parameter.
            num_heads: number of independent LinOSS heads.
            use_head_gating: whether to apply token-wise head gating before merge.
            use_head_output_projection: whether to apply a learned output projection after
                concatenating multi-head outputs.
            A_max: upper bound for A in AG initialization.
            G_max: upper bound for G in AG initialization.
            dtype: dtype for sequence mixer parameters and computation.
            *args: Additional positional arguments (ignored).
            **kwargs: Additional keyword arguments (ignored).
        """
        dtype = jnp.dtype(dtype)
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if in_features % num_heads != 0:
            raise ValueError(
                f"in_features={in_features} must be divisible by num_heads={num_heads}"
            )
        if state_dim % num_heads != 0:
            raise ValueError(f"state_dim={state_dim} must be divisible by num_heads={num_heads}")

        self.num_heads = num_heads
        self.head_hidden_dim = in_features // num_heads
        self.head_state_dim = state_dim // num_heads
        self.use_head_gating = use_head_gating and num_heads > 1
        self.use_head_output_projection = use_head_output_projection and num_heads > 1

        # Key generator
        def key_gen(key: PRNGKeyArray):
            while True:
                key, subkey = jr.split(key)
                yield subkey

        gen = key_gen(key)
        nxt = lambda: next(gen)  # noqa

        # A/G/steps: init helpers operate on total state_dim, then reshape to (num_heads, head_state_dim)
        if not damping:
            A_flat, G_flat, steps_flat = _init_linoss(nxt(), state_dim, 0.0, A_max, dtype=dtype)
        else:
            if initialization == "RT":
                A_flat, G_flat, steps_flat = _init_damped_linoss_rt(
                    nxt(), state_dim, discretization, r_min, 1.0, 0.0, theta_max, dtype=dtype
                )
            elif initialization == "AG":
                A_flat, G_flat, steps_flat = _init_damped_linoss_ag(
                    nxt(), state_dim, 0.0, A_max, 0.0, G_max, dtype=dtype
                )
            else:
                raise NotImplementedError(f"Initialization {initialization} not implemented")

        self.A_diag = A_flat.reshape(num_heads, self.head_state_dim)
        self.G_diag = G_flat.reshape(num_heads, self.head_state_dim) if G_flat is not None else None
        self.steps = steps_flat.reshape(num_heads, self.head_state_dim)

        self.B = _simple_uniform_init(
            nxt(),
            shape=(num_heads, self.head_state_dim, self.head_hidden_dim, 2),
            std=1.0 / math.sqrt(self.head_hidden_dim),
            dtype=dtype,
        )
        self.C = _simple_uniform_init(
            nxt(),
            shape=(num_heads, self.head_hidden_dim, self.head_state_dim, 2),
            std=1.0 / math.sqrt(self.head_state_dim),
            dtype=dtype,
        )
        self.D = normal(stddev=1.0)(nxt(), (num_heads, self.head_hidden_dim), dtype=dtype)
        self.head_gate = (
            eqx.nn.Linear(in_features, num_heads, key=nxt(), dtype=dtype)
            if self.use_head_gating
            else None
        )
        self.head_output_projection = (
            eqx.nn.Linear(in_features, in_features, key=nxt(), dtype=dtype)
            if self.use_head_output_projection
            else None
        )

        self.discretization = discretization
        self.initialization = initialization
        self.damping = damping

    def __call__(self, x: Array, key: PRNGKeyArray) -> Array:
        """Forward pass of the LinOSS sequence mixer layer.

        Args:
            x: Input sequence of features.
            key: JAX random key (unused; present for interface compatibility).

        Returns:
            The output of the LinOSS sequence mixer.
        """
        steps = nn.sigmoid(self.steps)
        return self._apply_multi_head(x, steps)

    def _apply_multi_head(self, x: Array, steps: Array) -> Array:
        """Apply independent LinOSS heads and merge them back into the hidden stream."""
        x_heads = x.reshape(x.shape[0], self.num_heads, self.head_hidden_dim)
        scan_inputs = jnp.swapaxes(x_heads, 0, 1)
        ys = jax.vmap(self._apply_recurrence)(self.A_diag, self.G_diag, self.B, scan_inputs, steps)
        ys = jnp.swapaxes(ys, 0, 1)
        head_outputs = jnp.einsum("hfs,lhs->lhf", self.C[..., 0], ys[..., 0]) - jnp.einsum(
            "hfs,lhs->lhf",
            self.C[..., 1],
            ys[..., 1],
        )
        head_outputs = head_outputs + x_heads * self.D[None, ...]

        if self.head_gate is not None:
            gate_logits = jax.vmap(self.head_gate)(x)
            gate_weights = nn.softmax(gate_logits, axis=-1)
            head_outputs = head_outputs * gate_weights[..., None]

        xs = head_outputs.reshape(x.shape[0], x.shape[1])
        if self.head_output_projection is not None:
            xs = jax.vmap(self.head_output_projection)(xs)
        return xs

    def _apply_recurrence(
        self,
        A_diag: Array,
        G_diag: Array | None,
        B: Array,
        x: Array,
        steps: Array,
    ) -> Array:
        """Apply the configured LinOSS recurrence for one head worth of parameters."""
        if not self.damping:
            A_diag = nn.relu(A_diag)
            if self.discretization == "IM":
                return _apply_linoss_im(A_diag, B, x, steps)
            elif self.discretization == "IMEX":
                return _apply_linoss_imex(A_diag, B, x, steps)
            else:
                raise NotImplementedError(
                    f"Discretization {self.discretization} not implemented for undamped"
                )
        else:
            A_diag, G_diag = _project_ag(self.discretization, A_diag, G_diag, steps)
            if self.discretization == "IM":
                return _apply_damped_linoss_im(A_diag, G_diag, B, x, steps)
            elif self.discretization == "IMEX":
                return _apply_damped_linoss_imex(A_diag, G_diag, B, x, steps)
            elif self.discretization == "IMEX2":
                return _apply_damped_linoss_imex2(A_diag, G_diag, B, x, steps)
            elif self.discretization == "IMEX3":
                return _apply_damped_linoss_imex3(A_diag, G_diag, B, x, steps)
            elif self.discretization == "EX":
                return _apply_damped_linoss_ex(A_diag, G_diag, B, x, steps)
            else:
                raise NotImplementedError(
                    f"Discretization {self.discretization} not implemented"
                )


# --- Initialization Helpers ----------------------------------


def _simple_uniform_init(
    rng: PRNGKeyArray,
    shape: tuple[int],
    std: float = 1.0,
    dtype: jnp.dtype = jnp.float32,
):
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


def _init_linoss(
    rng: PRNGKeyArray,
    state_dim: int,
    A_min: float,
    A_max: float,
    dtype: jnp.dtype = jnp.float32,
):
    """Initialize recurrence parameters for undamped LinOSS.

    Samples A_diag uniformly in [A_min, A_max]. G_diag is set to None.

    Args:
        rng: JAX random key for initialization.
        state_dim: Size of the state dimension.
        A_min: Lower bound for uniform A sampling.
        A_max: Upper bound for uniform A sampling.
        dtype: dtype for the returned arrays.

    Returns:
        Initialized (A_diag, G_diag, steps)
    """
    A_key, step_key = jr.split(rng, 2)
    A_diag = (A_min + random.uniform(A_key, shape=(state_dim,)) * (A_max - A_min)).astype(dtype)
    steps = normal(stddev=0.5)(step_key, (state_dim,), dtype=dtype)
    return A_diag, None, steps


def _init_damped_linoss_ag(
    rng: PRNGKeyArray,
    state_dim: int,
    A_min: float,
    A_max: float,
    G_min: float,
    G_max: float,
    dtype: jnp.dtype = jnp.float32,
):
    """Initialize recurrence parameters for Damped LinOSS (AG strategy).

    Samples A and G uniformly in their respective ranges.

    Args:
        rng: JAX random key for initialization.
        state_dim: Size of the state dimension.
        A_min: Lower bound for uniform A sampling.
        A_max: Upper bound for uniform A sampling.
        G_min: Lower bound for uniform G sampling.
        G_max: Upper bound for uniform G sampling.
        dtype: dtype for the returned arrays.

    Returns:
        Initialized (A_diag, G_diag, steps)
    """
    A_key, G_key, step_key = jr.split(rng, 3)
    A_diag = (A_min + random.uniform(A_key, shape=(state_dim,)) * (A_max - A_min)).astype(dtype)
    G_diag = (G_min + random.uniform(G_key, shape=(state_dim,)) * (G_max - G_min)).astype(dtype)
    steps = normal(stddev=0.5)(step_key, (state_dim,), dtype=dtype)
    return A_diag, G_diag, steps


def _init_damped_linoss_rt(
    rng: PRNGKeyArray,
    state_dim: int,
    discretization: Literal["IM", "IMEX", "IMEX2", "IMEX3", "EX"],
    r_min: float,
    r_max: float,
    theta_min: float,
    theta_max: float,
    dtype: jnp.dtype = jnp.float32,
):
    """Initialize recurrence parameters for Damped LinOSS (RT strategy).

    Samples uniformly in the 2D annulus specified by radius, theta bounds.
    Solves this symbolically using trace and determinant.

    Args:
        rng: JAX random key for initialization.
        state_dim: Size of the state dimension.
        discretization: discretization method to use.
        r_min: Lower bound for the radius.
        r_max: Upper bound for the radius.
        theta_min: Lower bound for the theta parameter.
        theta_max: Upper bound for the theta parameter.
        dtype: dtype for the returned arrays.

    Returns:
        Initialized (A_diag, G_diag, steps)
    """
    # Solve symbolically
    a, g, step, tr_sym, det_sym = sp.symbols("a g step tr det")

    # Characteristic recurrence for 1 decoupled 2x2 system
    # (Should line up with the _apply... functions below)
    if discretization == "IM":
        s = 1 + step * g + step**2 * a
        M_i = sp.Matrix(
            [
                [1 / s, -a * step / s],
                [step / s, (1 + step * g) / s],
            ]
        )
    elif discretization == "IMEX":
        s = 1 + step * g
        M_i = sp.Matrix(
            [
                [1 / s, -a * step / s],
                [step / s, 1 - a * step**2 / s],
            ]
        )
    elif discretization == "IMEX2":
        M_i = sp.Matrix(
            [
                [1 - step * g, -a * step],
                [step * (1 - step * g), 1 - step**2 * a],
            ]
        )
    elif discretization == "IMEX3":
        s = 1 + step**2 * a
        M_i = sp.Matrix(
            [
                [(1 - step * g) / s, -step * a / s],
                [step * (1 - step * g) / s, 1 / s],
            ]
        )
    elif discretization == "EX":
        M_i = sp.Matrix(
            [
                [1 - step * g, -step * a],
                [step, 1],
            ]
        )
    else:
        raise ValueError(f"Discretization {discretization} not implemented.")

    # Solve from trace and determinant (symmetric in eigenvalues)
    eqs = [sp.Eq(M_i.trace(), tr_sym), sp.Eq(M_i.det(), det_sym)]
    sol = sp.solve(eqs, (a, g))
    a_expr, g_expr = sol[a], sol[g]
    f = sp.lambdify((tr_sym, det_sym, step), (a_expr, g_expr), "numpy")

    # Sample timesteps
    mag_key, arg_key, step_key = jr.split(rng, 3)
    step_vals = normal(stddev=0.5)(step_key, (state_dim,))
    step_sigmoid = nn.sigmoid(step_vals)

    # Sample eigenvalues in ring
    mag = jnp.sqrt(jr.uniform(mag_key, shape=(state_dim,)) * (r_max**2 - r_min**2) + r_min**2)
    arg = jr.uniform(arg_key, shape=(state_dim,)) * (theta_max - theta_min) + theta_min
    tr_vals = 2 * mag * jnp.cos(arg)
    det_vals = mag**2

    # Convert to (A, G) representation
    a_vals, g_vals = f(tr_vals, det_vals, step_sigmoid)

    # Cast to real (imag part is nonzero, ~machine precision)
    return (
        jnp.array(a_vals.real, dtype=dtype),
        jnp.array(g_vals.real, dtype=dtype),
        step_vals.astype(dtype),
    )


# --- Projection Operations ----------------------------------


def _project_ag(discretization, A_diag, G_diag, steps):
    """Project A, G into the stable parameter space given the discretization.

    Args:
        discretization: discretization method to use.
        A_diag: un-projected A_diag parameters.
        G_diag: un-projected G_diag parameters.
        steps: pre-activated (e.g. sigmoid) timesteps.

    Returns:
        Projected (A_diag, G_diag)
    """
    if discretization == "IM":
        G_low = -steps * A_diag
        G_diag = G_low + nn.relu(G_diag - G_low)
        A_low = 1 / 4 * G_diag**2
        A_diag = A_low + nn.relu(A_diag - A_low)
    elif discretization == "IMEX":
        G_diag = nn.relu(G_diag)
        A_low = (2 + steps * G_diag - 2 * jnp.sqrt(1 + steps * G_diag)) / jnp.maximum(
            steps**2, 1e-6
        )
        A_high = (2 + steps * G_diag + 2 * jnp.sqrt(1 + steps * G_diag)) / jnp.maximum(
            steps**2, 1e-6
        )
        A_diag = A_low + nn.relu(A_diag - A_low) - nn.relu(A_diag - A_high)
    elif discretization == "IMEX2":
        G_diag = nn.relu(G_diag)
        A_low = (2 - steps * G_diag - 2 * jnp.sqrt(1 - steps * G_diag)) / jnp.maximum(
            steps**2, 1e-6
        )
        A_high = (2 - steps * G_diag + 2 * jnp.sqrt(1 - steps * G_diag)) / jnp.maximum(
            steps**2, 1e-6
        )
        A_diag = A_low + nn.relu(A_diag - A_low) - nn.relu(A_diag - A_high)
    elif discretization == "IMEX3":
        raise NotImplementedError
    elif discretization == "EX":
        G_low = steps * A_diag
        G_diag = G_low + nn.relu(G_diag - G_low)
        A_low = 1 / 4 * G_diag**2
        A_diag = A_low + nn.relu(A_diag - A_low)

    return A_diag, G_diag


# --- Scan Operations ----------------------------------


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
        Hidden state sequence of shape (L, state_dim, 2).
    """
    Bu_elements = jnp.einsum("sfd,lf->lsd", B, x)

    S = 1.0 + step**2.0 * A_diag
    M_11 = 1.0 / S
    M_12 = -1.0 * step * A_diag / S
    M_21 = step / S
    M_22 = 1.0 / S

    F1 = Bu_elements * (step / S)[None, :, None]
    F2 = Bu_elements * (step**2.0 / S)[None, :, None]
    return _apply_linoss_scan(M_11, M_12, M_21, M_22, F1, F2)


def _apply_linoss_imex(A_diag, B, x, step):  # noqa: N802
    """Compute the LinOSS-IMEX recurrence.

    Args:
        A_diag: Diagonal state matrix.
        B: Input matrix.
        x: Input sequence of features.
        step: Discretization time-step.

    Returns:
        Hidden state sequence of shape (L, state_dim, 2).
    """
    Bu_elements = jnp.einsum("sfd,lf->lsd", B, x)

    M_11 = jnp.ones_like(A_diag)
    M_12 = -1.0 * step * A_diag
    M_21 = step
    M_22 = 1.0 - (step**2.0) * A_diag

    F1 = Bu_elements * step[None, :, None]
    F2 = Bu_elements * (step**2.0)[None, :, None]
    return _apply_linoss_scan(M_11, M_12, M_21, M_22, F1, F2)


def _apply_damped_linoss_im(A_diag, G_diag, B, x, step):  # noqa: N802
    """Compute the Damped LinOSS-IM recurrence.

    Args:
        A_diag: Diagonal state matrix.
        G_diag: Diagonal damping matrix.
        B: Input matrix.
        x: Input sequence of features.
        step: Discretization time-step.

    Returns:
        Hidden state sequence of shape (L, state_dim, 2).
    """
    Bu_elements = jnp.einsum("sfd,lf->lsd", B, x)

    S = 1.0 + step * G_diag + step**2.0 * A_diag
    M_11 = 1.0 / S
    M_12 = -step * A_diag / S
    M_21 = step / S
    M_22 = (1.0 + step * G_diag) / S

    F1 = Bu_elements * (step / S)[None, :, None]
    F2 = Bu_elements * (step**2.0 / S)[None, :, None]
    return _apply_linoss_scan(M_11, M_12, M_21, M_22, F1, F2)


def _apply_damped_linoss_imex(A_diag, G_diag, B, x, step):  # noqa: N802
    """Compute the Damped LinOSS-IMEX1 recurrence.

    Args:
        A_diag: Diagonal state matrix.
        G_diag: Diagonal damping matrix.
        B: Input matrix.
        x: Input sequence of features.
        step: Discretization time-step.

    Returns:
        Hidden state sequence of shape (L, state_dim, 2).
    """
    Bu_elements = jnp.einsum("sfd,lf->lsd", B, x)

    S = 1.0 + step * G_diag
    M_11 = 1.0 / S
    M_12 = -step * A_diag / S
    M_21 = step / S
    M_22 = 1.0 - step**2.0 * A_diag / S

    F1 = Bu_elements * (step / S)[None, :, None]
    F2 = Bu_elements * (step**2.0 / S)[None, :, None]
    return _apply_linoss_scan(M_11, M_12, M_21, M_22, F1, F2)


def _apply_damped_linoss_imex2(A_diag, G_diag, B, x, step):  # noqa: N802
    """Compute the Damped LinOSS-IMEX2 recurrence.

    Args:
        A_diag: Diagonal state matrix.
        G_diag: Diagonal damping matrix.
        B: Input matrix.
        x: Input sequence of features.
        step: Discretization time-step.

    Returns:
        Hidden state sequence of shape (L, state_dim, 2).
    """
    Bu_elements = jnp.einsum("sfd,lf->lsd", B, x)

    M_11 = 1.0 - step * G_diag
    M_12 = -step * A_diag
    M_21 = step * (1.0 - step * G_diag)
    M_22 = 1.0 - step**2.0 * A_diag

    F1 = Bu_elements * step[None, :, None]
    F2 = Bu_elements * (step**2.0)[None, :, None]
    return _apply_linoss_scan(M_11, M_12, M_21, M_22, F1, F2)


def _apply_damped_linoss_imex3(A_diag, G_diag, B, x, step):  # noqa: N802
    """Compute the Damped LinOSS-IMEX3 recurrence.

    Args:
        A_diag: Diagonal state matrix.
        G_diag: Diagonal damping matrix.
        B: Input matrix.
        x: Input sequence of features.
        step: Discretization time-step.

    Returns:
        Hidden state sequence of shape (L, state_dim, 2).
    """
    Bu_elements = jnp.einsum("sfd,lf->lsd", B, x)

    S = 1.0 + step**2.0 * A_diag
    M_11 = (1.0 - step * G_diag) / S
    M_12 = -step * A_diag / S
    M_21 = step * (1.0 - step * G_diag) / S
    M_22 = 1.0 / S

    F1 = Bu_elements * (step / S)[None, :, None]
    F2 = Bu_elements * (step**2.0 / S)[None, :, None]
    return _apply_linoss_scan(M_11, M_12, M_21, M_22, F1, F2)


def _apply_damped_linoss_ex(A_diag, G_diag, B, x, step):  # noqa: N802
    """Compute the Damped LinOSS-EX recurrence.

    Args:
        A_diag: Diagonal state matrix.
        G_diag: Diagonal damping matrix.
        B: Input matrix.
        x: Input sequence of features.
        step: Discretization time-step.

    Returns:
        Hidden state sequence of shape (L, state_dim, 2).
    """
    Bu_elements = jnp.einsum("sfd,lf->lsd", B, x)

    M_11 = 1.0 - step * G_diag
    M_12 = -step * A_diag
    M_21 = step
    M_22 = jnp.ones_like(A_diag)

    F1 = Bu_elements * step[None, :, None]
    F2 = jnp.zeros_like(F1)
    return _apply_linoss_scan(M_11, M_12, M_21, M_22, F1, F2)
