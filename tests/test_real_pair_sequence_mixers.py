"""Reference-equivalence tests for real-pair sequence mixer implementations."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from discretax.sequence_mixers.linoss import LinOSSSequenceMixer


@jax.vmap
def _linoss_binary_operator_reference(q_i, q_j):  # noqa: N802
    A_i, b_i = q_i
    A_j, b_j = q_j

    state_dim = A_i.size // 4
    i_a = A_i[0 * state_dim : 1 * state_dim]
    i_b = A_i[1 * state_dim : 2 * state_dim]
    i_c = A_i[2 * state_dim : 3 * state_dim]
    i_d = A_i[3 * state_dim : 4 * state_dim]
    j_a = A_j[0 * state_dim : 1 * state_dim]
    j_b = A_j[1 * state_dim : 2 * state_dim]
    j_c = A_j[2 * state_dim : 3 * state_dim]
    j_d = A_j[3 * state_dim : 4 * state_dim]

    A_new = j_a * i_a + j_b * i_c
    B_new = j_a * i_b + j_b * i_d
    C_new = j_c * i_a + j_d * i_c
    D_new = j_c * i_b + j_d * i_d
    A_out = jnp.concatenate([A_new, B_new, C_new, D_new])

    b_i1 = b_i[:state_dim]
    b_i2 = b_i[state_dim:]
    b_new = jnp.concatenate(
        [
            j_a * b_i1 + j_b * b_i2,
            j_c * b_i1 + j_d * b_i2,
        ]
    )
    return A_out, b_new + b_j


def _apply_linoss_im_reference(A_diag, B_complex, x, step):  # noqa: N802
    Bu_elements = jax.vmap(lambda u: B_complex @ u)(x)

    schur_comp = 1.0 / (1.0 + step**2.0 * A_diag)
    M_11 = 1.0 - step**2.0 * A_diag * schur_comp
    M_12 = -1.0 * step * A_diag * schur_comp
    M_21 = step * schur_comp
    M_22 = schur_comp

    M = jnp.concatenate([M_11, M_12, M_21, M_22])
    M_elements = M * jnp.ones((x.shape[0], 4 * A_diag.shape[0]))

    F1 = M_11 * Bu_elements * step
    F2 = M_21 * Bu_elements * step
    F = jnp.hstack((F1, F2))

    _, xs = jax.lax.associative_scan(_linoss_binary_operator_reference, (M_elements, F))
    return xs[:, A_diag.shape[0] :]


def _apply_linoss_imex_reference(A_diag, B_complex, x, step):  # noqa: N802
    Bu_elements = jax.vmap(lambda u: B_complex @ u)(x)

    A_ = jnp.ones_like(A_diag)
    B_ = -1.0 * step * A_diag
    C_ = step
    D_ = 1.0 - (step**2.0) * A_diag

    M = jnp.concatenate([A_, B_, C_, D_])
    M_elements = M * jnp.ones((x.shape[0], 4 * A_diag.shape[0]))

    F1 = Bu_elements * step
    F2 = Bu_elements * (step**2.0)
    F = jnp.hstack((F1, F2))

    _, xs = jax.lax.associative_scan(_linoss_binary_operator_reference, (M_elements, F))
    return xs[:, A_diag.shape[0] :]


def _apply_damped_linoss_imex_reference(A_diag, G_diag, B_complex, x, step):  # noqa: N802
    Bu_elements = jax.vmap(lambda u: B_complex @ u)(x)

    identity = jnp.ones_like(A_diag)
    S = identity + step * G_diag
    M_11 = 1.0 / S
    M_12 = -step / S * A_diag
    M_21 = step / S
    M_22 = identity - step**2 / S * A_diag

    M = jnp.concatenate([M_11, M_12, M_21, M_22])
    M_elements = M * jnp.ones((x.shape[0], 4 * A_diag.shape[0]))

    F1 = step * (1.0 / S) * Bu_elements
    F2 = step**2 * (1.0 / S) * Bu_elements
    F = jnp.hstack((F1, F2))

    _, xs = jax.lax.associative_scan(_linoss_binary_operator_reference, (M_elements, F))
    return xs[:, A_diag.shape[0] :]


def _linoss_reference(mixer: LinOSSSequenceMixer, x: jax.Array) -> jax.Array:
    steps = jax.nn.sigmoid(mixer.steps)
    B_complex = mixer.B[..., 0] + 1j * mixer.B[..., 1]
    C_complex = mixer.C[..., 0] + 1j * mixer.C[..., 1]

    if mixer.discretization == "IM":
        A_diag = jax.nn.relu(mixer.A_diag)
        ys = _apply_linoss_im_reference(A_diag, B_complex, x, steps)
    elif mixer.damping:
        G_diag = jax.nn.relu(mixer.G_diag)
        A_boundary_low = (2 + steps * G_diag - 2 * jnp.sqrt(1 + steps * G_diag)) / steps**2
        A_boundary_high = (2 + steps * G_diag + 2 * jnp.sqrt(1 + steps * G_diag)) / steps**2
        A_diag = (
            A_boundary_low
            + jax.nn.relu(mixer.A_diag - A_boundary_low)
            - jax.nn.relu(mixer.A_diag - A_boundary_high)
        )
        ys = _apply_damped_linoss_imex_reference(A_diag, G_diag, B_complex, x, steps)
    else:
        A_diag = jax.nn.relu(mixer.A_diag)
        ys = _apply_linoss_imex_reference(A_diag, B_complex, x, steps)

    return jax.vmap(lambda hidden, inputs: (C_complex @ hidden).real + mixer.D * inputs)(ys, x)


def _assert_no_complex_leaves(module: eqx.Module) -> None:
    for leaf in jax.tree.leaves(module):
        if isinstance(leaf, jax.Array):
            assert not jnp.issubdtype(leaf.dtype, jnp.complexfloating)


@pytest.mark.parametrize(
    ("discretization", "damping"),
    [("IM", False), ("IMEX", False), ("IMEX", True)],
)
def test_linoss_sequence_mixer_matches_reference(discretization: str, damping: bool):
    """LinOSS paired-real forward matches the original complex formulation."""
    mixer = LinOSSSequenceMixer(
        in_features=5,
        state_dim=8,
        discretization=discretization,
        damping=damping,
        key=jr.PRNGKey(0),
    )
    x = jr.normal(jr.PRNGKey(1), (6, 5))

    actual = mixer(x, key=jr.PRNGKey(2))
    expected = _linoss_reference(mixer, x)

    assert jnp.allclose(actual, expected, atol=1e-3, rtol=1e-3)
    _assert_no_complex_leaves(mixer)


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
def test_linoss_real_pair_supports_low_precision(dtype):
    """LinOSS paired-real path can be cast to low precision and still execute."""
    linoss = jax.tree.map(
        lambda leaf: leaf.astype(dtype)
        if isinstance(leaf, jax.Array) and jnp.issubdtype(leaf.dtype, jnp.floating)
        else leaf,
        LinOSSSequenceMixer(in_features=4, state_dim=6, key=jr.PRNGKey(6)),
    )
    x = jr.normal(jr.PRNGKey(7), (5, 4)).astype(dtype)

    linoss_outputs = linoss(x, key=jr.PRNGKey(8))

    assert jnp.isfinite(linoss_outputs).all()
