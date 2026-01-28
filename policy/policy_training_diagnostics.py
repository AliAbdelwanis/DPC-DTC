"""
Training Diagnostics
=========================

Utilities for checking convergence of training losses 
and coverage of training data used for policy learning.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple


def plot_training_losses(
    losses: list,
    ref_losses: list,
    eff_losses: list,
    i_lim_losses: list,
    i_ss_losses: list,
    acts_norm_losses: list
) -> None:
    """Plot training loss convergence over iterations.

    Creates a 3x2 subplot figure showing the evolution of different loss
    components during policy training. All losses are shown on a logarithmic
    y-axis for better visualization of convergence across orders of magnitude.

    Parameters
    ----------
    losses : list
        Total loss values over iterations.
    ref_losses : list
        Torque reference tracking loss over iterations.
    eff_losses : list
        Efficiency (copper) loss over iterations, based on current magnitude.
    i_lim_losses : list
        Current limit constraint loss over iterations.
    i_ss_losses : list
        Steady-state current tracking loss over iterations.
    acts_norm_losses : list
        Voltage action normalization loss over iterations.

    Returns
    -------
    None
        Displays the matplotlib figure.
    """
    fig, axes = plt.subplots(3, 2, sharex=True)

    # Total loss
    axes[0, 0].plot(losses)
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_ylabel("Total loss")
    axes[0, 0].set_title('Total loss')

    # Torque tracking loss
    axes[0, 1].plot(ref_losses)
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_ylabel("Torque loss")
    axes[0, 1].set_title('Torque tracking loss')

    # Efficiency (copper) loss
    axes[1, 0].plot(eff_losses)
    axes[1, 0].set_ylabel("idq magnitude")
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_title('Copper loss (efficiency)')

    # Current limit constraint loss
    axes[1, 1].plot(i_lim_losses)
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_ylabel("Current constraint")
    axes[1, 1].set_title('Current limit loss')

    # Voltage constraint loss
    axes[2, 0].plot(acts_norm_losses)
    axes[2, 0].set_yscale('log')
    axes[2, 0].set_xlabel("Iteration")
    axes[2, 0].set_ylabel("Voltage constraint")
    axes[2, 0].set_title('Voltage constraint loss')

    # Steady-state current loss
    axes[2, 1].plot(i_ss_losses)
    axes[2, 1].set_yscale('log')
    axes[2, 1].set_xlabel("Iteration")
    axes[2, 1].set_ylabel("Steady-state error")
    axes[2, 1].set_title('Steady-state current loss')

    fig.tight_layout()


def check_speed_torque_distribution(
    motor_env,
    data: Optional[np.ndarray] = None,
    data_path: Optional[str] = None,
    speed_range: Tuple[float, float]=[0,11000],
    bins: int = 10
):
    """
    Check the distribution of training data in speed–torque space.

    This function is intended to verify that the training data
    covers the desired operating region uniformly.

    The input data is assumed to be normalized and is scaled
    back to physical units before plotting.

    Parameters
    ----------
    data : np.ndarray or None, optional
        Training data array of shape (N, 2) containing
        normalized [speed, torque] samples.
    data_path : str or None, optional
        Path to `.npy` file containing the training data.
        Used if `data` is None.
    omega_el_base : float, optional
        Electrical angular speed base [rad/s] used for normalization.
    torque_base : float, optional
        Torque base [Nm] used for normalization.
    speed_range : tuple(float, float), optional
        Speed range [RPM] to include in the histogram.
    torque_range : tuple(float, float), optional
        Torque range [Nm] shown in the plot title.
    bins : int, optional
        Number of histogram bins.

    Notes
    -----
    - Exactly one of `data` or `data_path` must be provided.
    - Speed is converted from electrical rad/s to mechanical RPM
      assuming 3 pole pairs.
    """

    if (data is None) == (data_path is None):
        raise ValueError("Provide exactly one of `data` or `data_path`.")

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------
    if data is None:
        data = np.load(data_path)

    data = np.asarray(data).reshape(-1, 2)

    # --------------------------------------------------
    # Convert to physical units
    # --------------------------------------------------
    speed_norm = data[:, 0]
    torque_norm = data[:, 1]

    speed_rpm = speed_norm * motor_env.env_properties.physical_constraints.omega_el*60/(3 * 2 * np.pi)
    torque_nm = torque_norm * motor_env.env_properties.physical_constraints.torque

    df = pd.DataFrame(
        {
            "speed": speed_rpm,
            "torque": torque_nm,
        }
    )

    # --------------------------------------------------
    # Filter speed range
    # --------------------------------------------------
    speed_min, speed_max = speed_range

    df_torque = df[
        (df["speed"] >= speed_min) &
        (df["speed"] <= speed_max)
    ]

    # --------------------------------------------------
    # Plot histogram
    # --------------------------------------------------

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # -------------------------------------------------------
    # Speed histogram (full range)
    # -------------------------------------------------------
    axes[0].hist(df["speed"], bins=bins, alpha=0.7, color="green", edgecolor="black")
    axes[0].set_title("Speed Distribution (full range)")
    axes[0].set_xlabel("Speed [RPM]")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(True)

    # -------------------------------------------------------
    # Torque histogram (filtered by speed range)
    # -------------------------------------------------------
    axes[1].hist(df_torque["torque"], bins=bins, alpha=0.7, color="blue", edgecolor="black")
    axes[1].set_title(f"Torque Distribution (speed {speed_min}-{speed_max} RPM)")
    axes[1].set_xlabel("Torque [Nm]")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()
