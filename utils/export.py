"""
Model Export Utilities
======================

Functions for exporting trained models to external formats
(e.g. MATLAB) for deployment or analysis.
"""

from pathlib import Path
import numpy as np
import scipy.io as sio
from typing import Sequence


def export_eqx_model_to_matlab(
    model,
    save_path,
    activations: Sequence[str] | None = None,
    verbose: bool = True
):
    """
    Export a feedforward Equinox model to MATLAB `.mat` format.

    The model is assumed to have a `.layers` attribute containing
    linear layers with `.weight` and `.bias`.

    Weight matrices are transposed to match MATLAB's convention:
        (in_features, out_features)

    Parameters
    ----------
    model : equinox.Module
        Trained Equinox model with a `layers` attribute.
    save_path : str or pathlib.Path
        Destination `.mat` file.
    activations : sequence of str or None, optional
        Activation functions per layer.
        If None, defaults to:
            ["leakyrelu"] * (num_layers - 1) + ["tanh"]
    verbose : bool, optional
        If True, print export summary.

    Notes
    -----
    This function is intended for *deployment*, not reloading
    back into Python.
    """

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    weights_list = []
    biases_list = []
    layer_sizes = []

    # --------------------------------------------------
    # Extract parameters
    # --------------------------------------------------
    for layer in model.layers:
        W = np.asarray(layer.weight).T   # MATLAB convention
        b = np.asarray(layer.bias)

        weights_list.append(W)
        biases_list.append(b)
        layer_sizes.append((W.shape[0], W.shape[1]))

    num_layers = len(weights_list)

    # --------------------------------------------------
    # Activations
    # --------------------------------------------------
    if activations is None:
        activations = ["leakyrelu"] * (num_layers - 1) + ["tanh"]

    if len(activations) != num_layers:
        raise ValueError(
            "Length of activations must match number of layers."
        )

    activations = np.array(activations, dtype=object)

    # --------------------------------------------------
    # Save to MATLAB
    # --------------------------------------------------
    mat_dict = {
        "weights": weights_list,
        "biases": biases_list,
        "activations": activations,
        "num_layers": num_layers,
        "layer_sizes": np.array(layer_sizes),
    }

    sio.savemat(save_path, mat_dict)

    # --------------------------------------------------
    # Logging
    # --------------------------------------------------
    if verbose:
        print(f"[export] Saved Equinox model to: {save_path}")
        print(f"[export] Layers: {layer_sizes}")
        print(f"[export] Activations: {list(activations)}")

from pathlib import Path
import numpy as np
from scipy.io import savemat

def export_trajectories_to_matlab(
    U_dq,        # input trajectory array: time x 2
    i_dq,        # state trajectory array: time x 2
    trq,         # output torque trajectory: time x 1
    t,           # time vector: time x 1
    save_path,
):
    """
    Save simulated trajectories to MATLAB .mat file in a Simulink-compatible format.

    Parameters
    ----------
    U_dq : array-like
        Input trajectory (e.g., voltage) shape [time_steps, 2]
    i_dq : array-like
        State trajectory (e.g., currents) shape [time_steps, 2]
    trq : array-like
        Output trajectory (torque) shape [time_steps, ]
    t : array-like
        Time vector, shape [time_steps, 1]
    save_path : str or Path, optional
        Name of .mat file to save
    project_root : Path, optional
        If provided, save_path is relative to project_root
    """

    # Convert to numpy arrays and transpose if needed
    U_dq_np = np.array(U_dq).T
    i_dq_np = np.array(i_dq).T
    trq_np = np.array(trq)
    t_np = np.array(t).reshape(-1, 1)

    # MATLAB-compatible structures
    state_struct = {
        "time": t_np,
        "signals": {
            "values": i_dq_np,
            "dimensions": i_dq_np.shape,
        },
    }

    output_struct = {
        "time": t_np,
        "signals": {
            "values": trq_np,
            "dimensions": trq_np.shape,
        },
    }

    input_struct = {
        "time": t_np,
        "signals": {
            "values": U_dq_np,
            "dimensions": U_dq_np.shape,
        },
    }

    # Save .mat
    savemat(save_path, {"idq_JAX": state_struct,
                        "trq_JAX": output_struct,
                        "udq_JAX": input_struct})

    print(f"[save_trajectory_to_matlab] Saved trajectories to '{save_path}'")