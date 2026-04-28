"""CPPO networks.

Ported from mava/networks/icrl.py with inline MLP torso (no Mava config dependency).
"""

from typing import Sequence, Tuple

import jax.numpy as jnp
from flax import linen as nn
from flax.linen.initializers import variance_scaling


class MLPTorso(nn.Module):
    """MLP torso with optional skip connections (every `skip_connections` layers).

    Matches JaxGCRL CRL Encoder/Actor pattern: residual added every `skip_connections`
    layers, with the first hidden activation as the skip source.
    """

    hidden_sizes: Sequence[int] = (256, 256)
    use_layer_norm: bool = False
    use_relu: bool = False
    skip_connections: int = 0  # 0 = disabled. Otherwise add skip every N layers.

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        act = nn.relu if self.use_relu else nn.swish
        skip = None
        for i, h in enumerate(self.hidden_sizes):
            x = nn.Dense(h, kernel_init=lecun_uniform, bias_init=bias_init)(x)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = act(x)
            if self.skip_connections:
                if i == 0:
                    skip = x
                elif i % self.skip_connections == 0 and skip is not None and skip.shape[-1] == x.shape[-1]:
                    x = x + skip
                    skip = x
        return x


class SAEncoder(nn.Module):
    """phi(s, a) — state-action encoder."""

    hidden_sizes: Sequence[int] = (256, 256)
    output_dim: int = 64
    action_embed_dim: int = 0
    use_film: bool = False
    use_layer_norm: bool = False
    use_relu: bool = False
    skip_connections: int = 0

    @nn.compact
    def __call__(self, s: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        torso = MLPTorso(self.hidden_sizes, self.use_layer_norm, self.use_relu, self.skip_connections)

        if self.use_film:
            state_features = torso(s)
            hidden_dim = state_features.shape[-1]
            action_hidden = nn.Dense(256, kernel_init=lecun_uniform, bias_init=bias_init)(a)
            action_hidden = nn.swish(action_hidden)
            film_gamma = nn.Dense(hidden_dim, kernel_init=lecun_uniform, bias_init=nn.initializers.ones)(action_hidden)
            film_beta = nn.Dense(hidden_dim, kernel_init=lecun_uniform, bias_init=bias_init)(action_hidden)
            x = film_gamma * state_features + film_beta
            x = nn.swish(x)
            return nn.Dense(self.output_dim, kernel_init=lecun_uniform, bias_init=bias_init)(x)

        if self.action_embed_dim > 0:
            a = nn.Dense(self.action_embed_dim, kernel_init=lecun_uniform, bias_init=bias_init)(a)
            a = nn.swish(a)
        x = jnp.concatenate([s, a], axis=-1)
        x = torso(x)
        return nn.Dense(self.output_dim, kernel_init=lecun_uniform, bias_init=bias_init)(x)


class GoalEncoder(nn.Module):
    """psi(g) — goal encoder."""

    hidden_sizes: Sequence[int] = (256, 256)
    output_dim: int = 64
    use_layer_norm: bool = False
    use_relu: bool = False
    skip_connections: int = 0

    @nn.compact
    def __call__(self, g: jnp.ndarray) -> jnp.ndarray:
        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        x = MLPTorso(self.hidden_sizes, self.use_layer_norm, self.use_relu, self.skip_connections)(g)
        return nn.Dense(self.output_dim, kernel_init=lecun_uniform, bias_init=bias_init)(x)


class Actor(nn.Module):
    """Gaussian actor for continuous action."""

    action_size: int
    hidden_sizes: Sequence[int] = (256, 256)
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    use_layer_norm: bool = False
    use_relu: bool = False
    skip_connections: int = 0

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        x = MLPTorso(self.hidden_sizes, self.use_layer_norm, self.use_relu, self.skip_connections)(x)
        mean = nn.Dense(self.action_size, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        log_std = nn.Dense(self.action_size, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        log_std = nn.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        return mean, log_std


class ValueNet(nn.Module):
    """V(s, g) value head."""

    hidden_sizes: Sequence[int] = (256, 256)
    use_layer_norm: bool = False
    use_relu: bool = False
    skip_connections: int = 0

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        x = MLPTorso(self.hidden_sizes, self.use_layer_norm, self.use_relu, self.skip_connections)(x)
        x = nn.Dense(1, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        return jnp.squeeze(x, axis=-1)
