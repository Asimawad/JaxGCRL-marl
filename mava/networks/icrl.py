# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple

import jax.numpy as jnp
from flax import linen as nn
from flax.linen.initializers import variance_scaling


class SAEncoder(nn.Module):
    """State-Action Encoder for ICRL - encodes (s,a) pairs using a configurable torso."""

    torso: nn.Module  # Configurable torso (e.g., MLPTorso)
    output_dim: int = 64
    action_embed_dim: int = 0  # If > 0, embed action to this dim before concat (amplifies action signal)
    use_film: bool = False  # If True, action modulates state features via FiLM

    @nn.compact
    def __call__(self, s: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        Args:
            s: State (observation without goal)
            a: Action

        Returns:
            output_dim-dimensional encoding
        """
        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        if self.use_film:
            # FiLM: action modulates state features via learned scale and shift
            # Process state through torso
            state_features = self.torso(s)
            hidden_dim = state_features.shape[-1]

            # Action produces scale (gamma) and shift (beta)
            action_hidden = nn.Dense(256, kernel_init=lecun_uniform, bias_init=bias_init)(a)
            action_hidden = nn.swish(action_hidden)
            film_gamma = nn.Dense(hidden_dim, kernel_init=lecun_uniform, bias_init=nn.initializers.ones)(action_hidden)
            film_beta = nn.Dense(hidden_dim, kernel_init=lecun_uniform, bias_init=bias_init)(action_hidden)

            # Apply FiLM: action controls how state features are transformed
            x = film_gamma * state_features + film_beta
            x = nn.swish(x)
            x = nn.Dense(self.output_dim, kernel_init=lecun_uniform, bias_init=bias_init)(x)
            return x

        # Default: concatenation-based encoding
        # Optional action embedding to amplify action signal
        if self.action_embed_dim > 0:
            a = nn.Dense(self.action_embed_dim, kernel_init=lecun_uniform, bias_init=bias_init)(a)
            a = nn.swish(a)

        # Concatenate state and (optionally embedded) action
        x = jnp.concatenate([s, a], axis=-1)

        # Use configurable torso instead of hardcoded layers
        x = self.torso(x)

        # Output layer: configurable dimension
        x = nn.Dense(self.output_dim, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        return x


class GoalEncoder(nn.Module):
    """Goal Encoder for ICRL - encodes goals using a configurable torso."""

    torso: nn.Module  # Configurable torso (e.g., MLPTorso)
    output_dim: int = 64

    @nn.compact
    def __call__(self, g: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        Args:
            g: Goal

        Returns:
            output_dim-dimensional encoding
        """
        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        # Use configurable torso
        x = self.torso(g)

        # Output layer: configurable dimension
        x = nn.Dense(self.output_dim, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        return x


class ICRLActor(nn.Module):
    """Actor network for ICRL - outputs continuous action distribution using a configurable torso."""

    torso: nn.Module  # Configurable torso (e.g., MLPTorso)
    action_size: int
    LOG_STD_MAX: float = 2.0
    LOG_STD_MIN: float = -5.0

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass.

        Args:
            x: Observation (base obs + goal concatenated)

        Returns:
            (mean, log_std) tuple for action distribution
        """
        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        # Use configurable torso
        x = self.torso(x)

        # Two output heads for mean and log_std
        mean = nn.Dense(self.action_size, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        log_std = nn.Dense(self.action_size, kernel_init=lecun_uniform, bias_init=bias_init)(x)

        # Clip log_std to reasonable range
        log_std = nn.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)

        return mean, log_std


class ICRLValueNet(nn.Module):
    """Value network for PPO-CRL - predicts V(s,g) for GAE advantage estimation."""

    torso: nn.Module

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        x = self.torso(x)
        x = nn.Dense(1, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        return jnp.squeeze(x, axis=-1)


class PQNStateActionEncoder(nn.Module):
    """
    State-Action Encoder for PQN-CRL - encodes (s,a) pairs using a configurable torso.
    """

    torso: nn.Module  # Configurable torso (e.g., MLPTorso)
    num_actions: int
    output_dim: int = 64

    @nn.compact
    def __call__(self, s: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        Args:
            s: State observation [batch, state_dim]

        Returns:
            Per-action representations [batch, num_actions, output_dim]
        """
        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        # Process state through torso
        x = self.torso(s)

        # Output layer: num_actions * output_dim
        x = nn.Dense(self.num_actions * self.output_dim, kernel_init=lecun_uniform, bias_init=bias_init)(x)

        # Reshape to [batch, num_actions, output_dim]
        batch_size = x.shape[:-1]
        x = x.reshape(*batch_size, self.num_actions, self.output_dim)

        return x
