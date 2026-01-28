"""
Environment-policy interaction and trajectory rollout utilities.

This module provides functions for simulating trajectories by rolling out
policies in environments. It includes:
- Single and batched rollouts with learned policies
- Policy-free environment rollouts
- Feature extraction during rollout
- Support for time-varying references

All functions are JIT-compiled for efficiency using Equinox and JAX.
"""

from typing import Callable
import jax
import jax.nn as jnn
import jax.numpy as jnp
import equinox as eqx
import diffrax
import optax
import time
from tqdm import tqdm


@eqx.filter_jit
def rollout_traj_env_policy(policy, init_obs, ref_obs, horizon_length, env, featurize):
    """Roll out environment-policy interaction for a single trajectory.
    
    Simulates a control trajectory by iteratively applying policy predictions
    and environment dynamics. The policy receives featurized observations and
    reference signals to compute control actions.
    
    Parameters
    ----------
    policy : eqx.Module
        Learned policy network mapping observations to actions.
    init_obs : jnp.ndarray
        Initial observation, shape (num_obs,).
    ref_obs : jnp.ndarray
        Reference observation(s). Shape (num_obs,) for constant reference
        or (horizon_length, num_obs) for time-varying reference.
    horizon_length : int
        Length of trajectory to generate.
    env : PMSM
        Motor environment instance.
    featurize : Callable
        Feature extraction function for policy input.
    
    Returns
    -------
    tuple
        (observations, actions, actions_alphabeta) where:
        - observations: trajectory of shape (horizon_length + 1, num_obs)
        - actions: dq-plane actions of shape (horizon_length, 2)
        - actions_alphabeta: alpha-beta plane actions of shape (horizon_length, 2)
    """
    init_state = env.generate_state_from_observation(init_obs, env.env_properties)

    # Handle reference format: expand scalar reference to horizon length
    if len(ref_obs.shape) == 1:
        ref_o = jnp.repeat(ref_obs[None, :], horizon_length, axis=0)
    else:
        ref_o = ref_obs
        assert ref_obs.shape[0] == horizon_length

    # Initialize feature state for integrated error tracking
    _, init_feat_state = featurize(init_obs, ref_o[0])
    init_feat_state = jnp.zeros_like(init_feat_state)
    
    def body_fun(carry, ref):
        """Body function for policy rollout scan."""
        obs, state, feat_state = carry

        # Extract features from current observation and reference
        policy_in, feat_state = featurize(obs, ref, feat_state)

        # Get policy action
        action = policy(policy_in)

        # Apply action to environment
        obs, state, action, action_albet = env.step(state, action, env.env_properties)

        return (obs, state, feat_state), (obs, action, action_albet)
    
    _, (observations, actions, action_albet) = jax.lax.scan(
        body_fun, (init_obs, init_state, init_feat_state), ref_o, horizon_length
    )
    observations = jnp.concatenate([init_obs[None, :], observations], axis=0)
    
    return observations, actions, action_albet


@eqx.filter_jit
def vmap_rollout_traj_env_policy(policy, init_obs, ref_obs, horizon_length, env, featurize):
    """Roll out environment-policy interaction for a batch of trajectories.
    
    Applies rollout_traj_env_policy to a batch of initial observations and
    references using vmap for efficient vectorized computation.
    
    Parameters
    ----------
    policy : eqx.Module
        Learned policy network (shared across batch).
    init_obs : jnp.ndarray
        Batch of initial observations, shape (batch_size, num_obs).
    ref_obs : jnp.ndarray
        Batch of reference observations, shape (batch_size, num_obs).
    horizon_length : int
        Length of trajectory to generate (shared across batch).
    env : PMSM
        Motor environment instance (shared across batch).
    featurize : Callable
        Feature extraction function (shared across batch).
    
    Returns
    -------
    tuple
        (observations, actions) where:
        - observations: shape (batch_size, horizon_length + 1, num_obs)
        - actions: shape (batch_size, horizon_length, 2)
    """
    observations, actions, _ = jax.vmap(
        rollout_traj_env_policy, in_axes=(None, 0, 0, None, None, None)
    )(policy, init_obs, ref_obs, horizon_length, env, featurize)
    return observations, actions


def rollout_traj_node(model, featurize, init_obs, actions, tau):
    """Roll out a learned dynamics model for a single trajectory.
    
    Parameters
    ----------
    model : eqx.Module
        Learned dynamics model.
    featurize : Callable
        Feature extraction function.
    init_obs : jnp.ndarray
        Initial observation.
    actions : jnp.ndarray
        Sequence of actions to apply.
    tau : float
        Integration time step.
    
    Returns
    -------
    jnp.ndarray
        Trajectory of observations.
    """
    feat_obs = featurize(init_obs)
    return model(feat_obs, actions, tau)


def rollout_traj_env(env, init_obs, actions):
    """Roll out environment without policy (action-prescribed trajectory).
    
    Applies a predetermined sequence of actions to the environment and records
    the resulting trajectory.
    
    Parameters
    ----------
    env : PMSM
        Motor environment instance.
    init_obs : jnp.ndarray
        Initial observation, shape (num_obs,).
    actions : jnp.ndarray
        Sequence of actions to apply, shape (horizon_length, action_dim).
    
    Returns
    -------
    jnp.ndarray
        Trajectory of observations, shape (horizon_length + 1, num_obs).
    """
    init_state = env.generate_state_from_observation(init_obs, env.env_properties)

    def body_fun(carry, action):
        state = carry
        obs, state = env.step(state, action, env.env_properties)
        return state, obs

    _, observations = jax.lax.scan(body_fun, init_state, actions)
    observations = jnp.concatenate([init_obs[None, :], observations], axis=0)
    return observations


@eqx.filter_jit
def vmap_rollout_traj_node(model, featurize, init_obs, actions, tau):
    """Roll out learned dynamics model for a batch of trajectories.
    
    Parameters
    ----------
    model : eqx.Module
        Learned dynamics model (shared across batch).
    featurize : Callable
        Feature extraction function (shared across batch).
    init_obs : jnp.ndarray
        Batch of initial observations, shape (batch_size, num_obs).
    actions : jnp.ndarray
        Batch of action sequences, shape (batch_size, horizon_length, action_dim).
    tau : float
        Integration time step (shared across batch).
    
    Returns
    -------
    jnp.ndarray
        Batch of trajectories, shape (batch_size, horizon_length + 1, num_obs).
    """
    observations = jax.vmap(
        rollout_traj_node, in_axes=(None, None, 0, 0, None)
    )(model, featurize, init_obs, actions, tau)
    return observations


@eqx.filter_jit
def vmap_rollout_traj_env(env, init_obs, actions):
    """Roll out environment for a batch of action-prescribed trajectories.
    
    Parameters
    ----------
    env : PMSM
        Motor environment instance (shared across batch).
    init_obs : jnp.ndarray
        Batch of initial observations, shape (batch_size, num_obs).
    actions : jnp.ndarray
        Batch of action sequences, shape (batch_size, horizon_length, action_dim).
    
    Returns
    -------
    jnp.ndarray
        Batch of trajectories, shape (batch_size, horizon_length + 1, num_obs).
    """
    observations = jax.vmap(
        rollout_traj_env, in_axes=(None, 0, 0)
    )(env, init_obs, actions)
    return observations