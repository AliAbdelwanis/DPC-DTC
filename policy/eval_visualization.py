"""
Visualization functions for policy evaluation and analysis.

This module provides plotting and visualization utilities for analyzing DPC
(Direct Predictive Control) policy performance, including:
- Reference tracking visualization (torques, currents, voltages)
- Settling time analysis and computation
- Boxplot generation for steady-state error analysis
- Filtering and signal processing utilities

All visualization functions are designed to work with JAX arrays and
matplotlib for comprehensive performance analysis.
"""

import jax
import jax.numpy as jnp
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter


def moving_average(x: jnp.ndarray, window_size: int) -> jnp.ndarray:
    """Compute moving average of a signal using convolution.
    
    Parameters
    ----------
    x : jnp.ndarray
        Input signal.
    window_size : int
        Size of the averaging window.
    
    Returns
    -------
    jnp.ndarray
        Smoothed signal with same shape as input.
    """
    kernel = jnp.ones(window_size) / window_size
    return jnp.convolve(x, kernel, mode='same')


def low_pass_filter(x: jnp.ndarray, alpha: float = 0.1) -> jnp.ndarray:
    """Apply first-order low-pass filter to a signal.
    
    Parameters
    ----------
    x : jnp.ndarray
        Input signal.
    alpha : float, optional
        Filter coefficient (0 < alpha <= 1). Smaller values yield more smoothing.
        Default is 0.1.
    
    Returns
    -------
    jnp.ndarray
        Filtered signal with same shape as input.
    """
    def step(carry, x_val):
        y_prev = carry
        y = alpha * x_val + (1 - alpha) * y_prev
        return y, y
    
    _, filtered = jax.lax.scan(step, x[0], x)
    return filtered


def plot_i_dq_ref_tracking_time(
    obs, obs_ref, acts, acts_albet, obs_des, axes, env, tau: float = 1e-4
):
    """Plot comprehensive reference tracking analysis with currents and voltages.
    
    Creates an 8-subplot figure showing torque tracking, current magnitude,
    dq-voltages, current components, and error metrics over time.
    
    Parameters
    ----------
    obs : jnp.ndarray
        Observed state trajectory, shape (N, 8).
    obs_ref : jnp.ndarray
        Reference state trajectory, shape (N, 8).
    acts : jnp.ndarray
        Applied dq-voltage actions, shape (N, 2).
    acts_albet : jnp.ndarray
        Applied alpha-beta voltage actions, shape (N, 2).
    obs_des : jnp.ndarray
        Desired state trajectory, shape (N, 8).
    axes : numpy.ndarray
        Matplotlib axes array of shape (4, 2).
    env : PMSM
        Motor environment instance.
    tau : float, optional
        Sampling time in seconds. Default is 1e-4.
    """
    env_prop = env.env_properties
    if obs.shape[0] > obs_ref.shape[0]:
        # cut off initial state
        obs = obs[1:]
    assert obs.shape[0] == obs_ref.shape[0]
   
    # Generate time array
    time = jnp.linspace(0, obs_ref.shape[0] - 1, obs_ref.shape[0]) * tau

    # Extract environment constraints
    T_max = env_prop.physical_constraints.torque
    I_max = env_prop.physical_constraints.i_d
    U_max = env_prop.action_constraints.u_d
    
    # Generate physical states from observations
    ref_state = env.vmap_generate_state_from_observation(obs_ref, env_prop).physical_state
    sim_state = env.vmap_generate_state_from_observation(obs, env_prop).physical_state
    des_state = env.vmap_generate_state_from_observation(obs_des, env_prop).physical_state
    id_iq_ana = jnp.concatenate((ref_state.i_d.reshape(-1, 1), ref_state.i_q.reshape(-1, 1)), axis=1)
    id_iq_sim = jnp.concatenate((sim_state.i_d.reshape(-1, 1), sim_state.i_q.reshape(-1, 1)), axis=1)
    trq_ach = ref_state.torque
    trq_des = des_state.torque
    trq_sim = sim_state.torque
    idq_ana = np.sqrt(id_iq_ana[:, 0]**2 + id_iq_ana[:, 1]**2)
    idq_sim = np.sqrt(id_iq_sim[:, 0]**2 + id_iq_sim[:, 1]**2)
    
    # Plot torques
    axes[0, 0].plot(time, trq_des, label=r"$T_\mathrm{des}$", color='orange')
    axes[0, 0].plot(time, trq_ach, label=r"$T_\mathrm{feas}$", color='black', linestyle='--')
    axes[0, 0].plot(time, trq_sim, label=r"$T$", color='b', linestyle='-.')
    axes[0, 0].set_ylabel(r"Torque in N.m")
    
    # Plot current magnitude error
    axes[0, 1].plot(
        time, (jnp.abs(idq_ana) - jnp.abs(idq_sim)) * 100 / I_max,
        label=r"${\|\boldsymbol{i}_\mathrm{dq}\|}_\mathrm{err}$", color='b'
    )
    error_idq = moving_average((jnp.abs(idq_ana) - jnp.abs(idq_sim)) * 100 / I_max, 5)
    axes[0, 1].plot(
        time, error_idq,
        label=r"${\|\boldsymbol{i}_\mathrm{dq}\|}_\mathrm{err}\,(MAV)$", color='darkorange'
    )
    axes[0, 1].set_ylabel(r"${\|\boldsymbol{i}_\mathrm{dq}\|}_\mathrm{err}$ in %")

    # Plot current magnitude comparison
    axes[1, 0].plot(time, jnp.ones(len(time)) * I_max, color='r', label=r"$i_\mathrm{dq,lim}$") 
    axes[1, 0].plot(time, idq_ana, label=r"${\|\boldsymbol{i}_\mathrm{dq}\|}_\mathrm{ana}$", color='black', linestyle='-')
    axes[1, 0].plot(time, idq_sim, label=r"${\|\boldsymbol{i}_\mathrm{dq}\|}$", color='b', linestyle='-.') 
    axes[1, 0].set_ylabel(r"${\|\boldsymbol{i}_\mathrm{dq}\|}$ in A")
    
    # Plot alpha-beta voltage plane
    hex_angle = jnp.linspace(0, 2 * jnp.pi, 7)
    hex_boundary = jnp.column_stack([
        (4 / 3) * jnp.cos(hex_angle),
        (4 / 3) * jnp.sin(hex_angle),
    ])
    theta = np.linspace(0, 2 * jnp.pi, 500)
    x_circle = (2 * np.sqrt(3) / 3) * jnp.cos(theta)
    y_circle = (2 * np.sqrt(3) / 3) * jnp.sin(theta)
    axes[1, 1].plot(hex_boundary[:, 0], hex_boundary[:, 1], "k-", lw=2, label=r"Hex-boundary (Over-modulation)")
    axes[1, 1].scatter(acts_albet[:, 0], acts_albet[:, 1], color='blue', label=r"$\boldsymbol{u}_\mathrm{\alpha\beta}$")
    axes[1, 1].plot(x_circle, y_circle, 'r--', linewidth=2, label=r'Circ-limit (Linear-modulation)')
    axes[1, 1].set_xlabel(r"$u_\mathrm{\alpha}$ in V")
    axes[1, 1].set_ylabel(r"$u_\mathrm{\beta}$ in V")
    
    # Plot torque error
    axes[2, 0].plot(time, (jnp.abs(trq_ach) - jnp.abs(trq_sim)) * 100 / T_max, label=r"$T_\mathrm{err}$", color='b')
    error_trq = moving_average((jnp.abs(trq_ach) - jnp.abs(trq_sim)) * 100 / T_max, 5)
    axes[2, 0].plot(time, error_trq, label=r"$T_\mathrm{err}\,(MAV)$", color='darkorange')
    axes[2, 0].set_ylabel(r"$T_\mathrm{err}$ in %")

    # Plot dq current components
    axes[2, 1].plot(time, id_iq_sim[:, 0], label=r"$i_\mathrm{d}$") 
    axes[2, 1].plot(time, id_iq_sim[:, 1], label=r"$i_\mathrm{q}$")  
    axes[2, 1].plot(time, id_iq_ana[:, 0], label=r"$i_\mathrm{d,ana}$") 
    axes[2, 1].plot(time, id_iq_ana[:, 1], label=r"$i_\mathrm{q,ana}$")
    axes[2, 1].set_ylabel(r"$i_\mathrm{d}$ and $i_\mathrm{q}$ in A")

    # Plot dq voltage magnitude
    axes[3, 0].plot(time, jnp.ones(len(time)) * U_max, color='r', label=r"Hex-constraint")
    axes[3, 0].plot(time, jnp.ones(len(time)) * 230.94, color='black', label=r"$u_\mathrm{inner-circ}$") 
    axes[3, 0].plot(time, jnp.sqrt(acts[:, 0]**2 + acts[:, 1]**2), label=r"$\boldsymbol{u}_\mathrm{dq}$", color='b', linestyle='-.') 
    axes[3, 0].set_ylim(0, U_max + 50)
    axes[3, 0].set_ylabel(r"$\boldsymbol{u}_\mathrm{dq}$ in V")
    axes[3, 0].set_xlabel(r"time in $s^\mathrm{-1}$")

    # Plot dq current component errors
    axes[3, 1].plot(time, jnp.abs(id_iq_ana[:, 0]) - jnp.abs(id_iq_sim[:, 0]), label=r"$i_\mathrm{d,error}$")  
    axes[3, 1].plot(time, jnp.abs(id_iq_ana[:, 1]) - jnp.abs(id_iq_sim[:, 1]), label=r"$i_\mathrm{q,error}$")  
    axes[3, 1].set_ylabel(r"$i_\mathrm{d,err}$ and $i_\mathrm{q,err}$ in %")
    axes[3, 1].set_xlabel(r"time in $s^\mathrm{-1}$")

    for i in range(4):
        for j in range(2):
            axes[i, j].legend()

def boxplot_error_by_bins(
    error_mean: dict,
    features: list,
    sorted_x: np.ndarray,
    sorted_x_idx: np.ndarray,
    bin_values: np.ndarray,
    features_indx,
    bin_type: str | None = None,
    figsize: tuple = (4.2, 3.8),
):
    """Create boxplots of steady-state error grouped by operating point bins.
    
    Visualizes mean absolute error for selected features grouped into bins
    along one axis (e.g., speed or torque), useful for grid-based analysis.
    
    Parameters
    ----------
    error_mean : dict
        Dictionary mapping feature indices to error arrays of shape (n_speeds, n_trqs).
    features : list[str]
        Feature names for labels.
    sorted_x : np.ndarray
        Operating point values (speeds or torques), shape (n_points,).
    sorted_x_idx : np.ndarray
        Indices for sorting.
    bin_values : np.ndarray
        Number of bins to create for each feature.
    features_indx : list[int]
        Indices of features to plot.
    bin_type : str, optional
        Type of binning ('Speed' or 'Torque'). Used for labeling.
    figsize : tuple, optional
        Figure size in inches. Default is (4.2, 3.8).
    """
    sorted_x_percent = sorted_x * 100
    n_speeds = error_mean[features_indx[0]].shape[0]
    n_trqs = error_mean[features_indx[0]].shape[1]
    
    for j in range(len(bin_values)):
        num_bins = bin_values[j]
        x_bins = np.linspace(sorted_x_percent.min(), sorted_x_percent.max(), num_bins + 1)
        bin_labels = [f"{int(x_bins[i])}–{int(x_bins[i+1])}" for i in range(num_bins)]
        
        # Create grid matching error shape
        if bin_type == 'Speed':
            x_matrix = np.tile(sorted_x_percent[:, np.newaxis], (1, n_trqs))
        else:
            x_matrix = np.tile(sorted_x_percent, (n_speeds, 1))
        
        x_flat = x_matrix.ravel()
        error_flat = error_mean[features_indx[j]].ravel()
        
        # Filter out NaN values
        valid = ~np.isnan(error_flat)
        x_valid = x_flat[valid]
        error_valid = error_flat[valid]
        
        # Assign to bins
        bin_indices = np.digitize(x_valid, x_bins) - 1
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)
        errors_by_range = [error_valid[bin_indices == b] for b in range(num_bins)]
        sample_counts = [len(e) for e in errors_by_range]
        bin_labels_with_n = [f"{bin_labels[i]}\nn={sample_counts[i]}" for i in range(num_bins)]
        
        plt.figure(figsize=figsize)
        plt.boxplot(
            errors_by_range,
            labels=bin_labels_with_n,
            showmeans=True,
            meanline=True,
            boxprops=dict(color='blue'),
            medianprops=dict(color='red'),
            meanprops=dict(color='green')
        )

        plt.xlabel(rf"{bin_type} reference (w.r.t. nominal {bin_type}) in \si{{\percent}}", fontsize=11)
        plt.ylabel(rf"{features[j]} MAE in \si{{\percent}}", fontsize=11)
        plt.title(rf"{features[j]} MAE across {bin_type.lower()} ranges", fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.5)

        # Legend
        legend_handles = [
            mlines.Line2D([], [], color='blue', linewidth=2, label=r'Box (data range)'),
            mlines.Line2D([], [], color='red', linewidth=2, label=r'Median'),
            mlines.Line2D([], [], color='green', linewidth=2, label=r'Mean'),
        ]
        plt.legend(handles=legend_handles, loc='upper left', frameon=False, fontsize=9)
        plt.tick_params(axis='x', labelsize=11)
        plt.tick_params(axis='y', labelsize=11)
        plt.tight_layout()
        plt.show()


def format_time_axis_ocp_comparison(ax, fontsize: int = 13, powerlimits: tuple = (-3, -3)):
    """Format time axis for OCP comparison plots with scientific notation.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to format.
    fontsize : int, optional
        Font size for labels and ticks. Default is 13.
    powerlimits : tuple, optional
        Power limits for scientific notation (min, max). Default is (-3, -3).
    """
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits(powerlimits)
    formatter.set_useOffset(False)

    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.get_offset_text().set_fontsize(fontsize)
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.margins(x=0, y=0)


def first_entry_time_3d(y: np.ndarray, r: np.ndarray, dt: float, tol: float) -> np.ndarray:
    """Compute first entry time into tolerance band for 3D trajectories.
    
    Finds the first time step where each trajectory enters the tolerance band
    around a reference trajectory.
    
    Parameters
    ----------
    y : np.ndarray
        Measured trajectory array, shape (n_speed, n_torque, n_steps).
    r : np.ndarray
        Reference trajectory array, shape (n_speed, n_torque, n_steps).
    dt : float
        Sampling time in seconds.
    tol : float
        Relative tolerance for band definition.
    
    Returns
    -------
    np.ndarray
        First entry times in seconds, shape (n_speed, n_torque). Returns np.nan
        if trajectory never enters band.
    """
    err = np.abs(y - r) / r
    inside = err <= tol

    ever_inside = np.any(inside, axis=-1)
    first_idx = np.argmax(inside, axis=-1)

    T_enter = first_idx.astype(float) * dt
    T_enter[~ever_inside] = np.nan

    return T_enter


def settling_time_env_3d(y: np.ndarray, r: np.ndarray, dt: float, tol: float) -> np.ndarray:
    """Compute settling time using envelope criterion for 3D trajectories.
    
    Identifies when each trajectory enters a tolerance envelope and remains
    within it for the rest of the time horizon.
    
    Parameters
    ----------
    y : np.ndarray
        Measured trajectory array, shape (n_speed, n_torque, n_steps).
    r : np.ndarray
        Reference trajectory array, shape (n_speed, n_torque, n_steps).
    dt : float
        Sampling time in seconds.
    tol : float
        Relative tolerance for band definition.
    
    Returns
    -------
    np.ndarray
        Settling times in seconds, shape (n_speed, n_torque). Returns np.nan
        if trajectory never settles within tolerance.
    """
    err = np.abs(y - r) / r

    future_max = np.maximum.accumulate(err[..., ::-1], axis=-1)[..., ::-1]
    settled = future_max <= tol

    ever_settled = np.any(settled, axis=-1)
    first_idx = np.argmax(settled, axis=-1)

    Ts = first_idx.astype(float) * dt
    Ts[~ever_settled] = np.nan

    return Ts


def calculate_settling_for_grid(
    opc_torque: np.ndarray, dpc_torque: np.ndarray, ref_dpc_trq: np.ndarray,
    tol: float = 0.1, dt: float = 1e-4
) -> tuple:
    """Calculate settling time statistics for grid of operating points.
    
    Computes mean and standard deviation of settling times for both OCP and DPC
    controllers across a grid of speed/torque operating points.
    
    Parameters
    ----------
    opc_torque : np.ndarray
        OCP torque trajectories, shape (n_speed, n_torque, n_steps).
    dpc_torque : np.ndarray
        DPC torque trajectories, shape (n_speed, n_torque, n_steps).
    ref_dpc_trq : np.ndarray
        Reference torque trajectories, shape (n_speed, n_torque, n_steps).
    tol : float, optional
        Relative tolerance for settling criterion. Default is 0.1 (10%).
    dt : float, optional
        Sampling time in seconds. Default is 1e-4.
    
    Returns
    -------
    tuple
        (T_OCP_avg, T_DPC_avg, T_OCP_std, T_DPC_std)
        Mean and standard deviation of settling times for each controller.
    """
    T_OCP = first_entry_time_3d(opc_torque, ref_dpc_trq, dt, tol)
    T_DPC = settling_time_env_3d(dpc_torque, ref_dpc_trq, dt, tol)
    
    T_OCP_avg = np.nanmean(T_OCP, axis=1)
    T_DPC_avg = np.nanmean(T_DPC, axis=1)
    T_OCP_std = np.nanstd(T_OCP, axis=1)
    T_DPC_std = np.nanstd(T_DPC, axis=1)

    return T_OCP_avg, T_DPC_avg, T_OCP_std, T_DPC_std