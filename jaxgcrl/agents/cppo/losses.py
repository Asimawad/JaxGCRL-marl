"""Contrastive loss + energy functions for CPPO.

Ported verbatim from mava/systems/icrl/losses.py.
"""

import jax
import jax.numpy as jnp


def energy_fn(name: str, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    if name == "norm":
        return -jnp.sqrt(jnp.sum((x - y) ** 2, axis=-1) + 1e-6)
    elif name == "l2":
        return -jnp.sum((x - y) ** 2, axis=-1)
    elif name == "dot":
        return jnp.sum(x * y, axis=-1)
    elif name == "cosine":
        x_norm = jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-6
        y_norm = jnp.linalg.norm(y, axis=-1, keepdims=True) + 1e-6
        return jnp.sum((x / x_norm) * (y / y_norm), axis=-1)
    else:
        raise ValueError(f"Unknown energy function: {name}. Available: norm, l2, dot, cosine")


def compute_logits(energy_name: str, sa_repr: jnp.ndarray, g_repr: jnp.ndarray) -> jnp.ndarray:
    return energy_fn(energy_name, sa_repr[:, None, :], g_repr[None, :, :])


def contrastive_loss_fn(name: str, logits: jnp.ndarray, logsumexp_penalty_coeff: float = 0.0) -> jnp.ndarray:
    if name == "fwd_infonce":
        critic_loss = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1))
    elif name == "bwd_infonce":
        critic_loss = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=0))
    elif name == "sym_infonce":
        critic_loss = -jnp.mean(
            2 * jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1) - jax.nn.logsumexp(logits, axis=0)
        )
    elif name == "binary_nce":
        I = jnp.eye(logits.shape[0])
        pos_loss = -jnp.mean(jax.nn.log_sigmoid(jnp.diag(logits)))
        neg_logits = logits * (1 - I)
        neg_loss = -jnp.mean(jax.nn.log_sigmoid(-neg_logits) * (1 - I))
        critic_loss = pos_loss + neg_loss
    else:
        raise ValueError(
            f"Unknown contrastive loss: {name}. Available: fwd_infonce, bwd_infonce, sym_infonce, binary_nce"
        )

    if logsumexp_penalty_coeff > 0:
        logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
        critic_loss += logsumexp_penalty_coeff * jnp.mean(logsumexp**2)

    return critic_loss


def compute_contrastive_metrics(logits: jnp.ndarray) -> dict:
    I = jnp.eye(logits.shape[0])
    logits_pos = jnp.sum(logits * I) / jnp.sum(I)
    logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)
    correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
    categorical_accuracy = jnp.mean(correct)
    logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1).mean()
    return {
        "logits_pos": logits_pos,
        "logits_neg": logits_neg,
        "categorical_accuracy": categorical_accuracy,
        "logsumexp": logsumexp,
    }
