
"""
This script defines loss functions used to train and update the parameters
of the policy.

The losses are designed to enforce desired control behavior and physical
constraints, and include:
- Torque tracking loss to penalize deviation from the reference torque
- d–q current magnitude loss to discourage excessive current usage (copper loss)
- d–q current limit loss to enforce hard current constraints

The loss functions are implemented using JAX to enable efficient,
differentiable, and JIT-compilable training pipelines. They are intended
to be weighted as part of an overall training objective for
PMSM torque-tracking control using DPC.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from utils.AnalyticalRG_Jax import *


# -------------------------
# Torque Tracking Loss
# -------------------------
@eqx.filter_jit
def ref_loss_fcn(obs, ref_obs):
    """
    Computes the torque deviation loss between observed and reference states.
    
    This loss measures how closely the torque component (4th) 
    of the observed trajectory matches the reference trajectory.
    
    Parameters:
        obs (jnp.ndarray): Observed state array of shape (N, 8), where N is 
                           the horizon length.
        ref_obs (jnp.ndarray): Reference state array, (8,).
        
    Returns:
        jnp.ndarray: Scalar loss representing summed squared torque deviation.
    """
    # Torque is assumed to be the 4th state variable (index 3)
    loss = jnp.sum((obs[:, 3] - ref_obs[3]) ** 2)
    return loss


# -------------------------
# Efficiency Loss
# -------------------------
@eqx.filter_jit
def efficincy_loss_fcn(obs):
    """
    Computes a copper loss approximated with current magnitude.
    
    This is the sum of magnitudes of the first two state components (
    representing i_d and i_q currents), promoting lower current usage.
    
    Parameters:
        obs (jnp.ndarray): Observed state array of shape (N, 8), where the 
                           first two columns correspond to i_d and i_q.
    
    Returns:
        jnp.ndarray: Scalar loss proportional to the total current magnitude.
    """
    loss_idq_normalized = jnp.sum(jnp.sqrt(obs[:, 0]**2 + obs[:, 1]**2))
    return loss_idq_normalized


# -------------------------
# Positive d-axis Current Penalty (Not in use)
# -------------------------
@eqx.filter_jit
def posit_id_loss_fcn(acts):
    """
    Penalizes positive d-axis current actions.
    
    This function computes the normalized magnitude of the first two action 
    components and returns the negative sum, effectively penalizing positive i_d.
    
    Parameters:
        acts (jnp.ndarray): Action array of shape (N, 2) representing i_d and i_q.
        
    Returns:
        jnp.ndarray: Negative normalized magnitude sum.
    """
    norm_acts = jnp.sum(jnp.sqrt(acts[:, 0]**2 + acts[:, 1]**2) / 230.94)
    return -norm_acts


# -------------------------
# Nominal Current Limit Penalty (currently not in use)
# -------------------------
@eqx.filter_jit
def idq_nom_loss_fcn(obs, i_nom):
    """
    Penalizes magnitude of dq-currents exceeding a nominal threshold.
    
    Uses ReLU to penalize only when the current magnitude exceeds `i_nom`.
    
    Parameters:
        obs (jnp.ndarray): Observed state array of shape (N, 8), where first 
                           two columns are i_d, i_q.
        i_nom (float): Nominal maximum current magnitude allowed.
    
    Returns:
        jnp.ndarray: Scalar loss penalizing currents above the nominal value.
    """
    idq = jnp.sqrt(obs[:, 0]**2 + obs[:, 1]**2)
    return jnp.sum(jax.nn.relu(idq - i_nom))


# -------------------------
# Current Limit Loss
# -------------------------
@eqx.filter_jit
def idq_lim_loss(obs, t):
    """
    Penalizes currents exceeding a time-dependent limit based on state conditions.
    
    If the normalized omega_el exceeds 0.33, a higher limit (1.02) is applied.
    Otherwise, the limit is 1.0. ReLU ensures penalties only occur when the 
    squared current exceeds the squared limit.
    
    Parameters:
        obs (jnp.ndarray): Observed state array (N, 8).
        t (float): Time step (unused in this version, included for API consistency).
    
    Returns:
        jnp.ndarray: Scalar loss penalizing currents above the adaptive limit.
    """
    idq = jnp.sqrt(obs[:, 0]**2 + obs[:, 1]**2)
    i_lim = jnp.where(obs[:, 2] > 0.33, 1.02, 1.0)
    return jnp.sum(jax.nn.relu(idq**2 - i_lim**2))


# -------------------------
# Current Limit Barrier Function
# -------------------------
@eqx.filter_jit
def idq_lim_loss_barrier(obs, t):
    """
    Implements a barrier-style loss to prevent exceeding normalized current limits.
    
    This smooth approximation penalizes current magnitudes approaching the limit 
    using a logarithmic barrier. Useful in optimization for safe exploration.
    
    Parameters:
        obs (jnp.ndarray): Observed state array (N, 8).
        t (float): Barrier scaling factor (larger t => steeper penalty near limit).
    
    Returns:
        jnp.ndarray: Barrier-based scalar loss.
    """
    idq = jnp.sqrt(obs[:, 0]**2 + obs[:, 1]**2)
    i_lim = 1.0
    z = idq**2 - i_lim**2
    
    # Smooth barrier function: piecewise for numerical stability
    return jnp.where(
        z <= -1 / t**2,
        -jnp.log(-z + 1e-8) / t,
        t * z - jnp.log(1 / t**2 + 1e-8) / t + 1 / t
    )


# -------------------------
# Steady-State Convergence Loss
# -------------------------
@eqx.filter_jit
def idq_SS_loss(obs, obs_ref):
    """
    Measures convergence of i_d/i_q currents to a reference steady-state trajectory.
    
    Computes the mean squared deviation of the first two state components 
    (i_d, i_q) from their reference values.
    
    Parameters:
        obs (jnp.ndarray): Observed state array (N, 8).
        obs_ref (jnp.ndarray): Reference state array (8,).
    
    Returns:
        jnp.ndarray: Scalar loss representing steady-state tracking error.
    """
    ach_loss = jnp.mean(jnp.sum((obs[:, 0:2] - obs_ref[0:2])**2, axis=0))
    return ach_loss
