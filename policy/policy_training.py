"""
Policy training utilities for learning-based DPC control.

This module provides functions and classes for training neural network policies
to control PMSM motors using DPC (Direct Predictive Control). It includes:
- Loss computation functions (tracking, efficiency, constraints)
- Gradient computation and parameter updates
- Batch data generation
- Validation and evaluation utilities
- DPCTrainer class for managing training loops

All functions use JAX with Equinox for JIT compilation and vmapping
for efficient batch processing.
"""

from typing import Callable

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from utils.AnalyticalRG import *
from tqdm import tqdm
from utils.interactions import (
    vmap_rollout_traj_env_policy,
)


@eqx.filter_value_and_grad(has_aux=True)
def grad_loss(
    policy: eqx.Module,
    env,
    init_obs: jnp.ndarray,
    ref_obs: jnp.ndarray,
    horizon_length: int,
    featurize: Callable,
    loss_fcns: list,
    t: jnp.ndarray,
    loss_weights: list
) -> tuple:
    """Compute loss and gradients for policy parameters.

    Performs a trajectory rollout with the current policy, computes the total loss
    and individual loss components (tracking, efficiency, constraints), and returns
    both the loss values and gradients for parameter updates.

    Parameters
    ----------
    policy : eqx.Module
        Policy neural network (typically MLP).
    env : PMSM
        Motor environment instance.
    init_obs : jnp.ndarray
        Initial observation, shape (batch_size, 8).
    ref_obs : jnp.ndarray
        Reference observation, shape (batch_size, 8).
    horizon_length : int
        Rollout horizon (number of steps).
    featurize : Callable
        Feature extraction function.
    loss_fcns : list
        List of loss function callables for different objectives.
    t : jnp.ndarray
        Current training iteration (for time-dependent losses).
    loss_weights : list
        Weighting coefficients for each loss component.

    Returns
    -------
    tuple
        (loss, aux_outputs) where:
        - loss: scalar total loss value
        - aux_outputs: tuple of individual loss components
          (ref_loss, efficincy_loss, idq_lim_loss, idq_ss_loss, acts_norm_loss)
    """
    obs, acts = vmap_rollout_traj_env_policy(policy, init_obs, ref_obs, horizon_length, env, featurize)
    loss, ref_loss, efficincy_loss, idq_lim_loss, idq_ss_loss, acts_norm_loss = vmap_compute_loss(
        obs, ref_obs, acts, featurize, loss_fcns, t, loss_weights
    )
    return loss, (ref_loss, efficincy_loss, idq_lim_loss, idq_ss_loss, acts_norm_loss)


@eqx.filter_jit
def make_step(
    policy: eqx.Module,
    env,
    init_obs: jnp.ndarray,
    ref_obs: jnp.ndarray,
    horizon_length: int,
    featurize: Callable,
    loss_fcns: list,
    loss_weights: list,
    optim: optax._src.base.GradientTransformation,
    opt_state,
    t: jnp.ndarray,
) -> tuple:
    """Execute one gradient descent step for policy training.

    Computes loss and gradients through a full trajectory rollout, applies
    optimizer updates to the policy parameters, and returns the updated policy
    along with loss values for monitoring.

    Parameters
    ----------
    policy : eqx.Module
        Current policy network.
    env : PMSM
        Motor environment instance.
    init_obs : jnp.ndarray
        Initial observation, shape (batch_size, 8).
    ref_obs : jnp.ndarray
        Reference observation, shape (batch_size, 8).
    horizon_length : int
        Trajectory rollout horizon.
    featurize : Callable
        Feature extraction function.
    loss_fcns : list
        Loss function components.
    loss_weights : list
        Loss weighting factors.
    optim : optax optimizer
        Gradient transformation optimizer.
    opt_state : optax optimizer state
        Current optimizer state.
    t : jnp.ndarray
        Training iteration counter.

    Returns
    -------
    tuple
        (policy, opt_state, loss, ref_loss, efficincy_loss, idq_lim_loss,
         idq_ss_loss, acts_norm_loss)
    """
    (loss, aux_outputs), grads = grad_loss(
        policy,
        env,
        init_obs,
        ref_obs,
        horizon_length,
        featurize,
        loss_fcns,
        t,
        loss_weights,
    )
    updates, opt_state = optim.update(grads, opt_state)
    policy = eqx.apply_updates(policy, updates)
    return policy, opt_state, loss, aux_outputs[0], aux_outputs[1], aux_outputs[2], aux_outputs[3], aux_outputs[4]

@eqx.filter_jit
def compute_loss(
    obs: jnp.ndarray,
    ref_obs: jnp.ndarray,
    acts: jnp.ndarray,
    featurize: Callable,
    loss_fcns: list,
    t: jnp.ndarray,
    loss_weights: list = [1, 0.5, 1.5, 1.5, 5]
) -> tuple:
    """Compute loss for a single trajectory in the batch.

    Combines multiple loss components (torque tracking, efficiency, current limits,
    steady-state accuracy) with dynamic weighting based on operating conditions.

    Parameters
    ----------
    obs : jnp.ndarray
        Observed trajectory, shape (horizon_length + 1, 8).
    ref_obs : jnp.ndarray
        Reference observation, shape (8,).
    acts : jnp.ndarray
        Applied actions, shape (horizon_length, 2).
    featurize : Callable
        Feature extraction function.
    loss_fcns : list
        List of 6 loss function callables.
    t : jnp.ndarray
        Training iteration (scalar).
    loss_weights : list, optional
        Base weights for each loss component. Default is [1, 0.5, 1.5, 1.5, 5].

    Returns
    -------
    tuple
        (total_loss, ref_loss, efficincy_loss, idq_lim_loss, idq_ss_loss,
         acts_norm_loss) all as scalars.
    """
    # Compute individual loss components
    ref_loss, w_ref = loss_fcns[0](obs, ref_obs), loss_weights[0]
    efficincy_loss, w_effi = loss_fcns[1](obs), loss_weights[1]
    posit_id_loss, w_pos_id = loss_fcns[2](acts), loss_weights[2]
    idq_nom_loss, w_idq_nom = loss_fcns[3](obs, 0.8), loss_weights[3]
    idq_lim_loss, w_idq_lim = loss_fcns[4](obs, t), loss_weights[4]
    idq_SS_loss, w_idq_SS = loss_fcns[5](obs, ref_obs), loss_weights[5]

    # Dynamic weighting: reduce efficiency loss for low torque references
    msk = (ref_obs[3] < 0.12)
    w_dyn = jnp.where(msk, 0.001, w_effi)

    # Compute weighted total loss
    loss = (
        w_ref * ref_loss
        + w_dyn * efficincy_loss
        + w_idq_lim * idq_lim_loss
        + w_pos_id * posit_id_loss
        + w_idq_nom * idq_nom_loss
        + w_idq_SS * idq_SS_loss
    )
    loss = jnp.clip(loss, max=1e5)

    return loss, w_ref * ref_loss, w_dyn * efficincy_loss, w_idq_lim * idq_lim_loss, w_idq_SS * idq_SS_loss, w_pos_id * posit_id_loss


@eqx.filter_jit
def vmap_compute_loss(
    sim_obs: jnp.ndarray,
    ref_obs: jnp.ndarray,
    acts: jnp.ndarray,
    featurize: Callable,
    loss_fcns: list,
    t: jnp.ndarray,
    loss_weights: list = [1, 0.5, 1.5, 1.5, 5, 1]
) -> tuple:
    """Compute batch loss by vmapping over trajectories.

    Applies compute_loss to a batch of trajectories and returns averaged
    loss components across the batch.

    Parameters
    ----------
    sim_obs : jnp.ndarray
        Batch of observed trajectories, shape (batch_size, horizon_length + 1, 8).
    ref_obs : jnp.ndarray
        Batch of reference observations, shape (batch_size, 8).
    acts : jnp.ndarray
        Batch of action sequences, shape (batch_size, horizon_length, 2).
    featurize : Callable
        Feature extraction function.
    loss_fcns : list
        Loss function components.
    t : jnp.ndarray
        Training iteration.
    loss_weights : list, optional
        Loss weighting factors. Default is [1, 0.5, 1.5, 1.5, 5, 1].

    Returns
    -------
    tuple
        (loss, ref_loss, efficincy_loss, idq_lim_loss, idq_ss_loss,
         acts_norm_loss) all batch-averaged.
    """
    loss, ref_loss, efficincy_loss, idq_lim_loss, idq_ss_loss, acts_norm_loss = jax.vmap(
        compute_loss, in_axes=(0, 0, 0, None, None, None, None)
    )(sim_obs, ref_obs, acts, featurize, loss_fcns, t, loss_weights)

    # Compute batch means
    ref_loss = jnp.mean(ref_loss)
    efficincy_loss = jnp.mean(efficincy_loss)
    idq_lim_loss = jnp.mean(idq_lim_loss)
    loss = jnp.mean(loss)
    idq_ss_loss = jnp.mean(idq_ss_loss)

    return loss, ref_loss, efficincy_loss, idq_lim_loss, idq_ss_loss, acts_norm_loss


@eqx.filter_jit
def val_step(
    policy: eqx.Module,
    env,
    init_obs: jnp.ndarray,
    ref_obs: jnp.ndarray,
    feas_obs: jnp.ndarray,
    horizon_length: int,
    featurize: Callable
) -> tuple:
    """Compute validation losses for policy evaluation.

    Performs trajectory rollout and computes transient and steady-state
    tracking metrics for validation during training.

    Parameters
    ----------
    policy : eqx.Module
        Policy network.
    env : PMSM
        Motor environment.
    init_obs : jnp.ndarray
        Batch of initial observations, shape (batch_size, 8).
    ref_obs : jnp.ndarray
        Batch of reference observations, shape (batch_size, 8).
    feas_obs : jnp.ndarray
        Batch of feasible observations, shape (batch_size, 8).
    horizon_length : int
        Rollout length.
    featurize : Callable
        Feature extraction function.

    Returns
    -------
    tuple
        (transient_torque_loss, steady_state_torque_loss, steady_state_current_loss)
        all batch-averaged.
    """
    obs, acts = vmap_rollout_traj_env_policy(
        policy, init_obs, ref_obs, horizon_length, env, featurize
    )
    batch_trans_torque_loss, batch_ss_torque_loss, batch_ss_current_loss = vmap_val_loss(
        env, obs, feas_obs
    )
    return batch_trans_torque_loss, batch_ss_torque_loss, batch_ss_current_loss


@eqx.filter_jit
def val_loss(
    env,
    obs: jnp.ndarray,
    feas_obs: jnp.ndarray,
    threshold: float = 0.95
) -> tuple:
    """Compute transient and steady-state validation losses.

    Separates trajectory into transient (before reaching reference) and
    steady-state (after reaching reference) phases and computes error
    metrics for each phase.

    Parameters
    ----------
    env : PMSM
        Motor environment (for constraint scaling).
    obs : jnp.ndarray
        Observed trajectory, shape (horizon_length, 8).
    feas_obs : jnp.ndarray
        Feasible reference observation, shape (8,).
    threshold : float, optional
        Settling threshold (e.g., 0.95 = ±5% of reference). Default is 0.95.

    Returns
    -------
    tuple
        (transient_torque_loss, steady_state_torque_loss, steady_state_current_loss)
    """
    trq_lim = env.env_properties.physical_constraints.torque
    idq_lim = env.env_properties.physical_constraints.i_d
    idq_meas = jnp.sqrt((obs[:, 0] * idq_lim) ** 2 + (obs[:, 1] * idq_lim) ** 2)
    idq_ref = jnp.sqrt((feas_obs[0] * idq_lim) ** 2 + (feas_obs[1] * idq_lim) ** 2)
    trq_feas = feas_obs[3] * trq_lim
    trq_meas = obs[:, 3] * trq_lim

    T = obs.shape[0]
    lower = jnp.minimum(threshold * trq_feas, (2 - threshold) * trq_feas)
    upper = jnp.maximum(threshold * trq_feas, (2 - threshold) * trq_feas)
    within = (trq_meas >= lower) & (trq_meas <= upper)
    ss_index = jnp.argmax(within)

    # Handle edge case: if steady-state is never reached, use full horizon
    reached_threshold = jnp.any(within)
    ss_index = jnp.where(reached_threshold, ss_index, T)

    # Compute tracking errors
    torque_error = trq_meas - trq_feas
    indices = jnp.arange(T)
    current_error = idq_meas - idq_ref

    # Create phase masks
    transient_torque_mask = indices < ss_index
    ss_torque_mask = indices >= ss_index

    # Apply masks to extract phase-specific errors
    transient_torque_error = jnp.where(transient_torque_mask, torque_error, 0.0)
    ss_torque_error = jnp.where(ss_torque_mask, torque_error, 0.0)
    ss_current_error = jnp.where(ss_torque_mask, current_error, 0.0)

    # Count elements in each phase
    transient_len = jnp.sum(transient_torque_mask)
    ss_torque_len = jnp.sum(ss_torque_mask)

    # Compute phase-specific MSE
    trans_loss_sum = jnp.sum(transient_torque_error ** 2)
    tss_loss_sum = jnp.sum(ss_torque_error ** 2)
    css_loss_sum = jnp.sum(ss_current_error ** 2)

    trans_torque_loss = jnp.where(
        transient_len > 0, trans_loss_sum / transient_len, 0.0
    )
    ss_torque_loss = jnp.where(ss_torque_len > 0, tss_loss_sum / ss_torque_len, 0.0)
    ss_current_loss = jnp.where(
        ss_torque_len > 0, css_loss_sum / ss_torque_len, 0.0
    )

    return trans_torque_loss, ss_torque_loss, ss_current_loss


@eqx.filter_jit
def vmap_val_loss(
    env,
    sim_obs: jnp.ndarray,
    feas_obs: jnp.ndarray
) -> tuple:
    """Compute batch validation losses by vmapping over trajectories.

    Parameters
    ----------
    env : PMSM
        Motor environment.
    sim_obs : jnp.ndarray
        Batch of observed trajectories, shape (batch_size, horizon_length, 8).
    feas_obs : jnp.ndarray
        Batch of feasible observations, shape (batch_size, 8).

    Returns
    -------
    tuple
        (transient_loss, steady_state_torque_loss, steady_state_current_loss)
        all batch-averaged.
    """
    trans_torque_loss, ss_torque_loss, ss_current_loss = jax.vmap(
        val_loss, in_axes=(None, 0, 0)
    )(env, sim_obs, feas_obs)

    batch_trans_torque_loss = jnp.mean(trans_torque_loss)
    batch_ss_torque_loss = jnp.mean(ss_torque_loss)
    batch_ss_current_loss = jnp.mean(ss_current_loss)

    return batch_trans_torque_loss, batch_ss_torque_loss, batch_ss_current_loss


def get_batches(
    X: jnp.ndarray,
    y: jnp.ndarray,
    Z: jnp.ndarray,
    batch_size: int
):
    """Iterate over dataset in mini-batches.

    Parameters
    ----------
    X : jnp.ndarray
        First data array.
    y : jnp.ndarray
        Second data array.
    Z : jnp.ndarray
        Third data array.
    batch_size : int
        Number of samples per batch.

    Yields
    ------
    tuple
        (X_batch, y_batch, Z_batch) for each batch.
    """
    N = X.shape[0]
    for start in range(0, N, batch_size):
        end = start + batch_size
        yield X[start:end], y[start:end], Z[start:end]


@eqx.filter_jit
def data_generation(
    env,
    reset_env: Callable,
    data_gen_single: Callable,
    gen_feas: Callable,
    rng: jnp.ndarray,
    traj_len: int | None = None
) -> tuple:
    """Generate batch of initial and reference observations.

    Vectorizes data generation over a batch of random keys, allowing parallel
    generation of multiple training samples.

    Parameters
    ----------
    env : PMSM
        Motor environment.
    reset_env : Callable
        Environment reset function.
    data_gen_single : Callable
        Single-sample data generation function.
    gen_feas : Callable
        Feasibility generation function.
    rng : jnp.ndarray
        Batch of random keys, shape (batch_size, 2).
    traj_len : int or None, optional
        Trajectory length (currently unused). Default is None.

    Returns
    -------
    tuple
        (init_obs, ref_obs, key) where:
        - init_obs: shape (batch_size, 8)
        - ref_obs: shape (batch_size, 8)
        - key: updated random keys, shape (batch_size, 2)
    """
    init_obs, ref_obs, key = jax.vmap(data_gen_single, in_axes=(None, None, 0, None))(
        env, reset_env, rng, gen_feas
    )
    return init_obs, ref_obs, key


def fit_on_env_non_jit(
    policy: eqx.Module,
    train_steps: int,
    env,
    reset_env: Callable,
    data_gen_sin: Callable,
    gen_feas: Callable,
    rng: jnp.ndarray,
    horizon_length: int,
    featurize: Callable,
    loss_fcns: list,
    loss_weights: list,
    optim: optax._src.base.GradientTransformation,
    init_opt_state,
) -> tuple:
    """Training loop for policy learning.

    Iterates for the specified number of steps, generating data, computing losses,
    and updating policy parameters. Returns final policy and loss histories.

    Parameters
    ----------
    policy : eqx.Module
        Initial policy network.
    train_steps : int
        Number of training iterations.
    env : PMSM
        Motor environment.
    reset_env : Callable
        Environment reset function.
    data_gen_sin : Callable
        Data generation function.
    gen_feas : Callable
        Feasibility filtering function.
    rng : jnp.ndarray
        Initial random key.
    horizon_length : int
        Rollout horizon.
    featurize : Callable
        Feature extraction function.
    loss_fcns : list
        Loss function components.
    loss_weights : list
        Loss weighting factors.
    optim : optax optimizer
        Gradient transformation optimizer.
    init_opt_state : tuple
        Initial optimizer state.

    Returns
    -------
    tuple
        (final_policy, final_opt_state, final_key, losses, ref_losses,
         eff_losses, i_lim_losses, i_ss_losses, acts_norm_losses, train_data)
    """
    key = rng
    policy_state = policy
    opt_state = init_opt_state
    losses = []
    ref_losses = []
    eff_losses = []
    i_lim_losses = []
    i_ss_losses = []
    acts_norm_losses = []
    t = jnp.array([0])
    train_data = []

    disp = tqdm(range(train_steps))
    for i in disp:
        t += 1
        # Generate batch of training data
        init_obs, ref_obs, key = data_generation(
            env, reset_env, data_gen_sin, gen_feas, key
        )
        train_data.append(ref_obs[:, 2:4])

        # Single gradient descent step
        (
            policy_state,
            opt_state,
            loss,
            ref_loss,
            efficincy_loss,
            idq_lim_loss,
            idq_ss_loss,
            acts_norm_loss,
        ) = make_step(
            policy_state,
            env,
            init_obs,
            ref_obs,
            horizon_length,
            featurize,
            loss_fcns,
            loss_weights,
            optim,
            opt_state,
            t,
        )

        # Update progress bar and loss histories
        disp.set_postfix({"loss": loss})
        losses.append(loss)
        ref_losses.append(ref_loss)
        eff_losses.append(efficincy_loss)
        i_lim_losses.append(idq_lim_loss)
        i_ss_losses.append(idq_ss_loss)
        acts_norm_losses.append(acts_norm_loss)

    return (
        policy_state,
        opt_state,
        key,
        losses,
        ref_losses,
        eff_losses,
        i_lim_losses,
        i_ss_losses,
        acts_norm_losses,
        train_data,
    )


def data_slice(
    rng: jnp.ndarray,
    obs_long: jnp.ndarray,
    acts_long: jnp.ndarray,
    sequence_len: int
) -> tuple:
    """Extract a random contiguous slice from trajectory data.

    Samples a random starting index and returns the corresponding observation
    and action sequences.

    Parameters
    ----------
    rng : jnp.ndarray
        Random number generator key.
    obs_long : jnp.ndarray
        Full observation trajectory, shape (traj_len, 8).
    acts_long : jnp.ndarray
        Full action trajectory, shape (traj_len, 2).
    sequence_len : int
        Length of subsequence to extract.

    Returns
    -------
    tuple
        (obs, acts, rng) where:
        - obs: observation slice, shape (sequence_len + 1, 8)
        - acts: action slice, shape (sequence_len, 2)
        - rng: updated random key
    """
    rng, subkey = jax.random.split(rng)
    idx = jax.random.randint(
        subkey,
        shape=(1,),
        minval=0,
        maxval=(obs_long.shape[0] - sequence_len - 1),
    )

    slice_idx = jnp.linspace(
        start=idx, stop=idx + sequence_len, num=sequence_len + 1, dtype=int
    ).T
    act_slice_idx = jnp.linspace(
        start=idx, stop=idx + sequence_len - 1, num=sequence_len, dtype=int
    ).T

    obs = obs_long[slice_idx][0]
    acts = acts_long[act_slice_idx][0]
    return obs, acts, rng


class DPCTrainer(eqx.Module):
    """Neural network policy trainer for DPC motor control.

    Manages the complete training pipeline including data generation, loss
    computation, parameter updates, and validation. Supports both JIT-compiled
    and non-JIT training loops.

    Attributes
    ----------
    batch_size : int
        Number of samples per training batch.
    train_steps : int
        Total number of training iterations.
    horizon_length : int
        Trajectory rollout length per iteration.
    reset_env : Callable
        Environment reset function.
    data_gen_sin : Callable
        Data generation function (uses sinusoidal references).
    gen_feas : Callable
        Feasibility filtering function.
    featurize : Callable
        Feature extraction function for policy input.
    policy_optimizer : optax optimizer
        Gradient transformation optimizer.
    loss_fcns : list
        List of loss function callables.
    loss_weights : list
        Weighting factors for each loss component.
    """

    batch_size: jnp.int32
    train_steps: jnp.int32
    horizon_length: jnp.int32
    reset_env: Callable
    data_gen_sin: Callable
    gen_feas: Callable
    featurize: Callable
    policy_optimizer: optax._src.base.GradientTransformationExtraArgs
    loss_fcns: list
    loss_weights: list

    def fit_non_jit(
        self,
        policy: eqx.Module,
        env,
        key: jnp.ndarray,
        opt_state
    ) -> tuple:
        """Execute non-JIT training loop.

        Parameters
        ----------
        policy : eqx.Module
            Initial policy network.
        env : PMSM
            Motor environment.
        key : jnp.ndarray
            Random keys for training, shape (batch_size, 2).
        opt_state : tuple
            Initial optimizer state.

        Returns
        -------
        tuple
            (final_policy, final_opt_state, final_key, losses, ref_losses,
             eff_losses, i_lim_losses, i_ss_losses, acts_norm_losses, train_data)
        """
        assert self.batch_size == key.shape[0]
        (
            final_policy,
            final_opt_state,
            final_key,
            losses,
            ref_losses,
            eff_losses,
            i_lim_losses,
            i_ss_losses,
            acts_norm_losses,
            train_data,
        ) = fit_on_env_non_jit(
            policy=policy,
            train_steps=self.train_steps,
            env=env,
            reset_env=self.reset_env,
            data_gen_sin=self.data_gen_sin,
            gen_feas=self.gen_feas,
            rng=key,
            horizon_length=self.horizon_length,
            featurize=self.featurize,
            loss_fcns=self.loss_fcns,
            loss_weights=self.loss_weights,
            optim=self.policy_optimizer,
            init_opt_state=opt_state,
        )

        return (
            final_policy,
            final_opt_state,
            final_key,
            losses,
            ref_losses,
            eff_losses,
            i_lim_losses,
            i_ss_losses,
            acts_norm_losses,
            train_data,
        )

    def evaluate(
        self,
        policy: eqx.Module,
        env,
        init_obs: jnp.ndarray,
        ref_obs: jnp.ndarray,
        feas_obs: jnp.ndarray,
        batch_size: int
    ) -> tuple:
        """Evaluate policy on validation data.

        Computes transient and steady-state tracking losses over a batch of
        trajectories by processing them in mini-batches.

        Parameters
        ----------
        policy : eqx.Module
            Policy network to evaluate.
        env : PMSM
            Motor environment.
        init_obs : jnp.ndarray
            Batch of initial observations, shape (n_samples, 8).
        ref_obs : jnp.ndarray
            Batch of reference observations, shape (n_samples, 8).
        feas_obs : jnp.ndarray
            Batch of feasible observations, shape (n_samples, 8).
        batch_size : int
            Mini-batch size for evaluation.

        Returns
        -------
        tuple
            (mean_transient_loss, mean_ss_torque_loss, mean_ss_current_loss)
        """
        ss_torque_loss = 0
        transient_torque_loss = 0
        ss_current_loss = 0
        i = 0

        for obs_b, ref_b, feas_b in get_batches(
            init_obs, ref_obs, feas_obs, batch_size=batch_size
        ):
            losses = val_step(
                policy, env, obs_b, ref_b, feas_b, self.horizon_length, self.featurize
            )
            transient_torque_loss += losses[0]
            ss_torque_loss += losses[1]
            ss_current_loss += losses[2]
            i += 1

        mean_transient_loss = transient_torque_loss / i
        mean_ss_torque_loss = ss_torque_loss / i
        mean_ss_current_loss = ss_current_loss / i

        print(mean_transient_loss, mean_ss_torque_loss, mean_ss_current_loss)
        return mean_transient_loss, mean_ss_torque_loss, mean_ss_current_loss
    

