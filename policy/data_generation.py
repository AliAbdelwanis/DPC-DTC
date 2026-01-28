"""
This script defines utility functions for environment initialization,
reference generation, feasibile reference generation, and feature construction
for a learning-based trajectory tracking setup.

The code is built using JAX and Equinox, with an emphasis on functional,
JIT-compiled components for efficient simulation and data generation.
It includes:
- Environment reset wrappers for reproducible stochastic initialization
- Generation of initial and reference observations for node-level data
- Feasibile reference trajectories
- Featurization logic for constructing policy inputs, including
  an integrated horizon error.

These utilities serve as modular building blocks for training, evaluation,
and rollout pipelines in DPC.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from utils.AnalyticalRG_Jax import *

# Generate init_obs
@eqx.filter_jit
def reset(env,rng, omega_el_coeff=1):
    """
    Reset the environment and return the initial observation.

    This function wraps the environment's `reset` method and is JIT-compiled
    using Equinox's `filter_jit` for performance. It resets the environment
    using the provided random number generator and optional electrical
    angular velocity scaling coefficient.

    Parameters
    ----------
    env : PMSM 
        PMSM environment instance.
    rng : jax.random.PRNGKey
        JAX random number generator key used for environment initialization.
    omega_el_coeff : float, optional
        Scaling coefficient for the electrical angular velocity.
        Defaults to 1.

    Returns
    -------
    obs : jax.numpy.ndarray 
        Initial observation of shape (8,) returned by the environment after reset.
    """
    obs, _ = env.reset(env.env_properties, rng, omega_el_coeff=omega_el_coeff) 
    return obs

@eqx.filter_jit
def generate_feasible(env_properties, ref_obs):
    """
    Ensure that reference torque are feasible given the operating omega_el and returns
    the theoretical optimal dq-currents

    If any reference torque is infeasible, it is projected into the
    feasible maximum torque according to the operating omega_el and return the 
    corresponding optimal dq-currents.

    Parameters
    ----------
    env_properties : jax pytree_dataclass
        Environment configuration or properties defining feasibility
        constraints (e.g., limits, bounds, physical parameters).
    ref_obs : jax.numpy.ndarray 
        Reference observation(s) of shape (8,).

    Returns
    -------
    ref_obs : jax.numpy.ndarray 
        Reference observation(s) of shape (8,) guaranteed to be 
        feasible for the environment.
    """
    omega_max = env_properties.physical_constraints.omega_el
    torque_max = env_properties.physical_constraints.torque
    omega_k = ref_obs[2]*omega_max
    m_ref =  ref_obs[3]*torque_max
    values = {"ld": env_properties.static_params.l_d,
        "lq": env_properties.static_params.l_q,
        "Rs": env_properties.static_params.r_s,
        "n_p": env_properties.static_params.p,
        "lambda_p": env_properties.static_params.psi_p, 
        "u_m": 254.65, # Fundamental component (six-step modulation)
        "i_m": env_properties.physical_constraints.i_d, 
        }
    i_dq, m = jnp_operation_management(values, m_ref, omega_k)
    ref_obs = ref_obs.at[3].set(m/torque_max)
    ref_obs = ref_obs.at[:2].set(i_dq/env_properties.physical_constraints.i_d)
    return ref_obs

@eqx.filter_jit
def node_dat_gen_sin(env,reset_env, rng, gen_feas):
    """
    Generate initial and reference observations for a single node.

    This function samples two independent environment resets using different
    random keys:
    - one for the reference observation
    - one for the initial observation

    The third state (omega_el) of the reference observation is overwritten
    with the corresponding value from the initial observation for consistency.

    Parameters
    ----------
    env : PMSM
        Environment instance passed to the reset function.
    reset_env : callable
        Function that resets the environment and returns an observation.
        Expected signature: reset_env(env, rng) -> obs
    rng : jax.random.PRNGKey
        JAX random number generator key.
    gen_feas : callable
        Feasibility generation function. Currently unused.

    Returns
    -------
    init_obs : jax.numpy.ndarray 
        Initial observation of shape (8,), sampled from the environment.
    ref_obs : jax.numpy.ndarray 
        Reference observation of shape (8,), sampled from the environment.
    rng : jax.random.PRNGKey
        Updated RNG key after splitting.
    """
    rng, subkey = jax.random.split(rng)
    ref_obs = reset_env(env,subkey)
    rng, subkey = jax.random.split(rng)
    init_obs = reset_env(env,subkey)
    ref_obs = ref_obs.at[2].set(init_obs[2])

    return init_obs, ref_obs, rng

# Create features for the policy network
@eqx.filter_jit
def featurize(obs, ref_obs, featurize_state=jnp.array([0])):
    """
    Generate policy input features from observations and references.

    This function constructs feature vectors for the policy based on
    the current observation, a reference observation, and an integrated
    horizon torck-tracking error.

    Parameters
    ----------
    obs : jax.numpy.ndarray 
        Current environment observation of shape (8,).
    ref_obs : jax.numpy.ndarray
        Reference observation of shape (8,).
    featurize_state : jax.numpy.ndarray,
        Integrated horizon torque-tracking error. This represents accumulated
        tracking error over the horizon.

    Returns
    -------
    policy_in : jax.numpy.ndarray
        Feature vector of shape (8,)
    featurize_state : jax.numpy.ndarray
        Updated integrated horizon torque-tracking error.
    """
    policy_in = jnp.concat([obs[0:3], ref_obs[3:4], ref_obs[3:4]-obs[3:4], 0.1*obs[4:6], featurize_state])
    featurize_state=jnp.clip(featurize_state + ref_obs[3:4]-obs[3:4],min=-1,max=1) * (jnp.sign(0.02-jnp.sum((ref_obs[3:4]-obs[3:4])**2))*0.5+0.5)
    return policy_in, featurize_state
