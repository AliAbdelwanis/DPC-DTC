# DPC-DTC: Neural Network-Based Differentiable Predictive Control for PMSM Torque Tracking

A comprehensive [Jax](https://github.com/google/jax)-based implementation of Differentiable Predictive Control (DPC) for Permanent Magnet Synchronous Motors (PMSM), featuring neural network policy learning and comparison with optimal control solutions.

## Overview

This project implements a learning-based control framework for PMSM torque tracking using:

- **Differentiable Predictive Control (DPC):** Real-time predictive control without solving complex optimization at runtime
- **Neural Network Policy:** Multi-layer perceptron trained via gradient descent to learn near-optimal control behavior
- **JAX Framework:** High-performance automatic differentiation and JIT compilation for efficient training and inference
- **Analytical Reference Generation:** Constrained optimization using Lagrange multipliers for optimal current references
- **Open-Loop Optimal Control (OCP):** CasADi-based benchmark for comparing policy performance

## Installation
Download the current state of the [exciting-environments](https://github.com/AliAbdelwanis/exciting-environments) repository, e.g.:
```
git clone https://github.com/AliAbdelwanis/exciting-environments.git
```
and install it in your python environment by moving to the downloaded folder and running ```pip install .```. Afterwards, download the source code of this repo, e.g.:

```
git clone https://github.com/AliAbdelwanis/DPC-DTC.git
```
## Project Structure

```
meta-DPC-DTC/
├── policy/                          # Policy training and evaluation
│   ├── networks.py                  # MLP architecture (Equinox)
│   ├── policy_training.py           # Training loop and DPCTrainer class
│   ├── policy_training_diagnostics.py # Visualization and monitoring
│   ├── data_generation.py           # Reference generation and features
│   ├── losses.py                    # loss functions
│   ├── evaluation.py                # Policy evaluation utilities
│   ├── eval_visualization.py        # Result visualization
│   ├── data/                        # Pre-computed training data
│   └── models/                      # Trained policy checkpoints
│
├── utils/                           # Utility modules
│   ├── AnalyticalRG.py             # NumPy reference generation (Lagrange multipliers)
│   ├── AnalyticalRG_Jax.py         # JAX reference generation (JIT-compiled)
│   ├── OCP.py                       # CasADi-based optimal control solver
│   ├── interactions.py              # Environment interaction utilities
│   ├── signals.py                   # Trajectory generation (speed, torque)
│   ├── Approx_six_step.py          # PWM approximation analysis
│   ├── export.py                    # Model export utilities
│   └── generate_filenames.py        # Checkpoint path generation
│
├── visualization/                   # Plot styling
│   └── style.py                     # Matplotlib configuration
│
├── notebooks/                       # Interactive examples
│   ├── 1.training_example.ipynb    # End-to-end training tutorial
│   ├── 2.policy_evaluation.ipynb   # Evaluation and OCP comparison
│   └── 0.dpc_training_my_model.ipynb # Custom training notebook
│
├── train.py                         # Standalone training script
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

### Motor Environment

**PMSM Simulator: exciting-environments library**
- BRUSA electric vehicle motor


## JAX Integration:
- JIT compilation for speedup
- Automatic differentiation for gradients
- Vectorization (vmap) for batch operations
- GPU acceleration for training

## Notebooks Guide

### 1. `training_example.ipynb` - Training Tutorial

Learn how to:
1. Initialize PMSM motor environment
2. Configure neural network architecture
3. Define multi-objective loss functions
4. Execute training loop
5. Monitor convergence via loss plots

**Output:** Trained policy checkpoint saved to `policy/models/`

### 2. `policy_evaluation.ipynb` - Evaluation & Comparison

Includes:
1. Single-point evaluation at specific speed/torque
2. Analytical reference comparison
3. Steady-state grid evaluation 
4. Modulation index visualization
5. Error statistics and boxplots
6. Transient analysis 
7. Optional: OCP solving or loading cached solutions
8. OCP vs DPC trajectory comparison
9. Settling time analysis

**Output:** Comprehensive visualization plots and performance metrics

## File Format Reference

### Policy Checkpoint (`.eqx`)

Binary Equinox serialization format containing neural network weights and architecture info.

```bash
policy/models/policy_hex_layer_sz-128_num_layers-3_steps-500000_h-60_ieff-0.008_iL-0.4_iLOffset-1.02_lw_Speed_ieff-0.001_stage-final.eqx
```

### Simulation Data (`.npz`)

NumPy archive containing pre-computed OCP solutions:
```bash
pmsm_OCP_simulation_s-24_t-51_steps-40.npz
```

## Citation

If you use this codebase in your research, please cite:

```bibtex
@software{abdelwanis2026dpc,
  title={Enhanced Differentiable Predictive Torque Control for Permanent-Magnet Synchronous Motors
  Including Overmodulation Region},
  author={Abdelwanis, Ali and Schweins, Oliver and Meyer, Marvin and Wallscheid, Oliver},
  year={2026},
  url={https://github.com/AliAbdelwanis/DPC-DTC.git}
}
```

## Contact & Support

For questions or issues:
- Create a GitHub issue in the repository
- Contact: [ali.abdelwanis@uni-siegen.de]
- Documentation: See inline docstrings and notebook tutorials

## Acknowledgments

- PMSM motor model and environment: exciting-environments library
- JAX team for automatic differentiation infrastructure

---
