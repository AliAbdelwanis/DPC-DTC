"""
Reference signal generation utilities for policy training and evaluation.

This module provides functions for generating reference trajectories used in
DPC policy training and evaluation. It includes:
- Random signal generation with constraints
- Trajectory generation with feasibility filtering
- Manual torque reference generation for grid-based studies

These utilities are designed to work with JAX for efficient computation
and JIT compilation.
"""

import jax
import jax.numpy as jnp


def steps_ref_traj_manual(
    env, reset_fun, gen_feas_fun, key, trq_list,
    step_lens: list = [100, 400], omega_el_coeff: float = 1
) -> tuple:
    """Generate reference trajectories with manually specified torques.
    
    Creates reference trajectories by concatenating steps with explicitly
    specified torque values. Each step has fixed length. Optionally, each 
    step is feasibility-filtered to ensure it is reachable (currently not 
    used; commented out).
    
    Parameters
    ----------
    env : PMSM
        Motor environment instance.
    reset_fun : Callable
        Function to reset environment and generate reference observations.
    gen_feas_fun : Callable
        Function to filter reference to ensure feasibility constraints.
    key : jax.random.PRNGKey
        Random number generator key.
    trq_list : array-like
        List of normalized torque reference values.
    step_lens : list, optional
        [step_length, unused] for consistent interface. Default [100, 400].
        Only first element is used.
    omega_el_coeff : float, optional
        Electrical angular velocity coefficient. Default 1.
    
    Returns
    -------
    tuple
        (ref, des, ref_plot) where:
        - ref: reference trajectory with desired torques, shape (ref_len, 8)
        - des: desired (unfiltered) trajectory, shape (ref_len, 8)
        - ref_plot: filtered reference for visualization, shape (ref_len, 8)
    """
    ref = []
    des = []
    ref_plot = []
    key, key2 = jax.random.split(key)
    ref_len = len(trq_list) * step_lens[0]
    
    for t in trq_list:
        key, subkey1, subkey2 = jax.random.split(key, num=3)
        ref_obs = reset_fun(env, subkey1)
        ref_obs = ref_obs.at[2].set(omega_el_coeff)
        ref_obs = ref_obs.at[3].set(t)
        
        # Fixed step length for each torque reference
        t_step = step_lens[0]
        des.append(jnp.repeat(ref_obs[:, None], t_step, axis=1))
        filtered_ref = gen_feas_fun(env.env_properties, ref_obs)
        #ref_obs = filtered_ref   #      make sure torque is feasible
        ref_plot.append(jnp.repeat(filtered_ref[:, None], t_step, axis=1)) 
        ref.append(jnp.repeat(ref_obs[:, None], t_step, axis=1))

    return jnp.hstack(ref)[:, :ref_len].T, jnp.hstack(des)[:, :ref_len].T, jnp.hstack(ref_plot)[:, :ref_len].T