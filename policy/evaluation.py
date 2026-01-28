"""
Policy evaluation functions for training and performance assessment.

This module provides utilities for evaluating learned DPC policies against
reference trajectories. It includes:
- Trajectory evaluation with optional visualization
- Analytical reference generation
- Steady-state error computation and masking
- Fundamental voltage extraction from DPC control signals
- Reference generation via optimal control (SQP-based)

All functions support both deterministic and random reference generation.
"""

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from policy.eval_visualization import plot_i_dq_ref_tracking_time
from utils.signals import steps_ref_traj_manual
from utils.interactions import rollout_traj_env_policy

def steps_eval_manual(
    env, reset_fun, gen_feas_fun, policy, featurize, key, trq_list,
    init_obs_key=None, plot=True, step_lens=[100, 100], omega_el_coeff=1
):
    """Evaluate policy with manually specified torque references.
    
    Parameters
    ----------
    env : PMSM
        Motor environment instance.
    reset_fun : Callable
        Function to generate reference observations.
    gen_feas_fun : Callable
        Function to filter reference for feasibility.
    policy : eqx.Module
        Learned policy network.
    featurize : Callable
        Feature extraction function.
    key : jax.random.PRNGKey
        Random number generator key.
    trq_list : array-like
        List of normalized torque references to evaluate.
    init_obs_key : jax.random.PRNGKey, optional
        Key for initial observation sampling.
    plot : bool, optional
        Whether to generate visualization. Default is True.
    step_lens : list, optional
        [step_length, unused] for each torque reference. Default [100, 100].
    omega_el_coeff : float, optional
        Electrical angular velocity coefficient. Default 1.
    
    Returns
    -------
    tuple
        (obs, obs_ref, acts, obs_des, obs_ref_plot, acts_albet)
    """
    obs_ref, obs_des, obs_ref_plot = steps_ref_traj_manual(
        env, reset_fun, gen_feas_fun, key, trq_list, step_lens=step_lens, omega_el_coeff=omega_el_coeff
    )
    init_obs = reset_fun(env, init_obs_key, omega_el_coeff)
    obs, acts, acts_albet = rollout_traj_env_policy(
        policy, init_obs, obs_ref, len(trq_list) * step_lens[0], env, featurize
    )
    
    if plot:
        fig, axes = plt.subplots(4, 2, figsize=(20, 15), sharex=False)
        plot_i_dq_ref_tracking_time(obs, obs_ref_plot, acts, acts_albet, obs_des, axes, env, tau=env.tau)
    
    return obs, obs_ref, acts, obs_des, obs_ref_plot, acts_albet


def generate_mask(
    ref_data, sorted_trq_idx: np.ndarray, sorted_trq: np.ndarray,
    rated_trq, ss_start_idx: int = 50, tolerance: float = 1
):
    """Create mask for feasible operating points in evaluation grid.
    
    Identifies (speed, torque) points where reference achieves target torque
    in steady-state, used to mask unreliable error measurements.
    
    Parameters
    ----------
    ref_data : np.ndarray
        Reference data array, shape (n_speeds, n_steps, n_features).
    sorted_trq_idx : np.ndarray
        Indices for sorting torque references.
    sorted_trq : np.ndarray
        Sorted normalized torque values.
    rated_trq : float
        Rated torque for physical scaling.
    ss_start_idx : int, optional
        Time index where steady-state begins. Default 50.
    tolerance : float, optional
        Torque tolerance in N.m for feasibility. Default 1.
    
    Returns
    -------
    np.ndarray
        Boolean mask of shape (n_speeds, n_trqs, step_len - ss_start_idx),
        True for feasible points.
    """
    n_speeds = ref_data.shape[0]
    n_trqs = len(sorted_trq_idx)
    step_len = ref_data.shape[1] // n_trqs
    
    # Extract reference torques in steady-state
    ref_trq = (ref_data[:, :, 3] * rated_trq).reshape(n_speeds, n_trqs, step_len)[
        :, sorted_trq_idx, ss_start_idx:
    ].mean(axis=2)
    
    torque_target_grid = sorted_trq[np.newaxis, :]
    mask = (np.abs(ref_trq - torque_target_grid) < tolerance)
    mask_expanded = np.repeat(mask[:, :, np.newaxis], step_len - ss_start_idx, axis=2)
    
    return mask_expanded


def compute_steady_state_error(
    sim_data, ref_data, feature_idx: list, sorted_trq_idx: np.ndarray,
    sorted_trq: np.ndarray, Re_scale: list,
    mask_feasible: bool = True, tolerance: float = 1.0, ss_start_idx: int = 50
):
    """Compute mean absolute error in steady-state for selected features.
    
    Parameters
    ----------
    sim_data : np.ndarray
        Simulated data, shape (n_speeds, n_steps, n_features).
    ref_data : np.ndarray
        Reference data, shape (n_speeds, n_steps, n_features).
    feature_idx : list[int]
        Feature indices to compute (2 = idq magnitude, 3 = torque, etc.).
    sorted_trq_idx : np.ndarray
        Indices for sorting torque in evaluation grid.
    sorted_trq : np.ndarray
        Sorted normalized torque values.
    Re_scale : list[float]
        Physical scaling factors for each feature.
    mask_feasible : bool, optional
        Mask infeasible operating points. Default True.
    tolerance : float, optional
        Tolerance for feasibility masking. Default 1.0.
    ss_start_idx : int, optional
        Time index for steady-state start. Default 50.
    
    Returns
    -------
    dict
        Dictionary mapping feature indices to error arrays,
        shape (n_speeds, n_trqs) each, with NaN for infeasible points.
    """
    n_speeds = sim_data.shape[0]
    n_trqs = len(sorted_trq_idx)
    step_len = sim_data.shape[1] // n_trqs
    error_mean_dict = {}
    
    for feat in feature_idx:
        # Extract and scale features
        if feat == 2:
            # Compute dq-current magnitude
            sim_id = sim_data[:, :, 0] * Re_scale[0]
            ref_id = ref_data[:, :, 0] * Re_scale[0]
            sim_iq = sim_data[:, :, 1] * Re_scale[1]
            ref_iq = ref_data[:, :, 1] * Re_scale[1]
            sim_feat = jnp.sqrt(sim_id**2 + sim_iq**2)
            ref_feat = jnp.sqrt(ref_id**2 + ref_iq**2)
        else:
            # Extract scalar feature
            sim_feat = sim_data[:, :, feat] * Re_scale[feat]
            ref_feat = ref_data[:, :, feat] * Re_scale[feat]
        
        # Reshape into speed/torque/time blocks
        sim_blocks = sim_feat.reshape(n_speeds, n_trqs, step_len)
        ref_blocks = ref_feat.reshape(n_speeds, n_trqs, step_len)
        
        # Extract steady-state region
        sim_slice = sim_blocks[:, sorted_trq_idx, ss_start_idx:]
        ref_slice = ref_blocks[:, sorted_trq_idx, ss_start_idx:]
        
        # Apply feasibility mask if requested
        if mask_feasible:
            mask_expanded = generate_mask(ref_data, sorted_trq_idx, sorted_trq, Re_scale[3])
        else:
            mask_expanded = np.ones_like(sim_slice, dtype=bool)
        
        # Compute mean absolute error percentage
        normalization = jnp.ones_like(ref_slice) * Re_scale[feat]
        error_raw = np.abs(sim_slice - ref_slice) * 100 / normalization
        error_masked = np.where(mask_expanded, error_raw, np.nan)
        
        # Average over steady-state time region
        error_mean = np.nanmean(error_masked, axis=2)
        error_mean_dict[feat] = error_mean
    
    return error_mean_dict


def extract_fund_voltage_from_dpc(sorted_u_d, sorted_u_q, speed_scaled, Ts: float = 1e-4):
    """Extract fundamental (average) dq-voltage from DPC control signals.
    
    Computes the average dq-voltage magnitude for each (speed, torque) point,
    representing the fundamental component of the DPC modulation.
    
    Parameters
    ----------
    sorted_u_d : np.ndarray
        d-axis voltages, shape (n_speeds, n_torques, n_steps).
    sorted_u_q : np.ndarray
        q-axis voltages, shape (n_speeds, n_torques, n_steps).
    speed_scaled : np.ndarray
        Electrical speeds in rad/s, shape (n_speeds,).
    Ts : float, optional
        Sampling time in seconds. Default 1e-4.
    
    Returns
    -------
    np.ndarray
        Fundamental dq-voltage magnitude, shape (n_speeds, n_torques).
    """
    N_speed, N_torque, N_steps = sorted_u_d.shape
    
    Vd_fund = np.zeros((N_speed, N_torque))
    Vq_fund = np.zeros((N_speed, N_torque))
    Vfund_norm = np.zeros((N_speed, N_torque))
    
    for si in range(N_speed):
        for ti in range(N_torque):
            # Extract dq voltage segment
            vd_seg = sorted_u_d[si, ti, :N_steps]
            vq_seg = sorted_u_q[si, ti, :N_steps]
            
            # Compute fundamental (average) dq voltage
            Vdq_fund = np.mean(vd_seg + 1j * vq_seg)
            
            Vd_fund[si, ti] = np.real(Vdq_fund)
            Vq_fund[si, ti] = np.imag(Vdq_fund)
            Vfund_norm[si, ti] = np.abs(Vdq_fund)
    
    print("Fundamental dq extraction completed using omega_e input.")
    return Vfund_norm