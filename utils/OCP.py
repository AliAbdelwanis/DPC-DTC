"""
Optimal Control Problem (OCP) solver for open-loop trajectory generation.

This module provides utilities for solving finite-horizon optimal control
problems using CasADi's NLP interface (IPOPT solver). It generates minimum-cost
voltage trajectories to track desired torque references while respecting
current and voltage constraints.

The OCP minimizes a cost function combining:
- Torque tracking error (squared)
- Copper losses (proportional to current magnitude squared)

Subject to:
- PMSM motor dynamics
- Current magnitude limits
- Voltage hexagon boundary constraints (two-level inverter)
"""

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt


def generate_OCP_traj(
    motor_env,
    speed: np.ndarray,
    trq: np.ndarray,
    sim_data: np.ndarray | None = None,
    h: int = 40
) -> list:
    """Generate optimal open-loop trajectories for speed-torque grid.

    Solves a finite-horizon OCP for each (speed, torque) pair using IPOPT.
    The OCP minimizes tracking error and copper losses subject to motor
    dynamics and constraints. Returns control and state trajectories.

    Parameters
    ----------
    motor_env : PMSM
        Motor environment with static parameters and constraints.
    speed : np.ndarray
        Normalized speed values [0, 1], shape (n_speeds,).
    trq : np.ndarray
        Normalized torque values [0, 1], shape (n_torques,).
    sim_data : np.ndarray or None, optional
        DPC simulation data for warm-starting initial currents.
        Shape (n_speeds, n_torques*step_len, 8). Default is None.
    h : int, optional
        OCP horizon length (number of control steps). Default is 40.

    Returns
    -------
    list of dict
        Results for each (speed, torque) pair. Each dict contains:
        - "i0": initial current [A], shape (2,)
        - "T_des": desired torque [Nm], scalar
        - "omega": operating speed [rad/s], scalar
        - "U_opt": optimal dq-voltage trajectory [V], shape (h, 2)
        - "i_traj": dq-current trajectory [A], shape (h+1, 2)
        - "T_opt": torque trajectory [Nm], shape (h+1,)
        - "u_alpha": alpha-voltage trajectory [V], shape (h,)
        - "u_beta": beta-voltage trajectory [V], shape (h,)
    """
    # Problem dimensions and parameters
    N = h
    nx = 2
    nu = 2
    Ts = 1e-4

    theta0 = 0.0
    w_cu = 0.001  # Copper loss weight
    i_max = motor_env.env_properties.physical_constraints.i_d
    Umax = motor_env.env_properties.action_constraints.u_d

    # Extract motor parameters from environment
    R = motor_env.env_properties.static_params.r_s
    Ld = motor_env.env_properties.static_params.l_d
    Lq = motor_env.env_properties.static_params.l_q
    psi_f = motor_env.env_properties.static_params.psi_p
    p = motor_env.env_properties.static_params.p

    # Define PMSM dynamics (CasADi version for symbolic computation)
    def f(x, u, omega_val):
        """Compute dq-current derivatives from PMSM motor model.
        
        Parameters
        ----------
        x : ca.MX
            State vector [id, iq].
        u : ca.MX
            Control input [ud, uq].
        omega_val : float
            Electrical angular velocity (scalar).
            
        Returns
        -------
        ca.MX
            State derivative [did/dt, diq/dt].
        """
        id_, iq_ = x[0], x[1]
        ud, uq = u[0], u[1]
        did = (ud + omega_val * Lq * iq_ - R * id_) / Ld
        diq = (uq - omega_val * (Ld * id_ + psi_f) - R * iq_) / Lq
        return ca.vertcat(did, diq)

    # Define torque function (CasADi version)
    def torque(x):
        """Compute torque from dq-currents.
        
        Parameters
        ----------
        x : ca.MX
            State vector [id, iq].
            
        Returns
        -------
        ca.MX
            Torque scalar.
        """
        id_, iq_ = x[0], x[1]
        return 1.5 * p * (psi_f * iq_ + (Ld - Lq) * id_ * iq_)

    # Define dynamics and torque (NumPy version for trajectory evaluation)
    def f_np(x, u, omega_val):
        """NumPy version of PMSM dynamics for post-solve trajectory computation.
        
        Parameters
        ----------
        x : np.ndarray
            State [id, iq].
        u : np.ndarray
            Control [ud, uq].
        omega_val : float
            Electrical speed.
            
        Returns
        -------
        np.ndarray
            State derivative.
        """
        id_, iq_ = x
        ud, uq = u
        did = (ud + omega_val * Lq * iq_ - R * id_) / Ld
        diq = (uq - omega_val * (Ld * id_ + psi_f) - R * iq_) / Lq
        return np.array([did, diq])

    def torque_np(id_, iq_):
        """NumPy version of torque function.
        
        Parameters
        ----------
        id_ : float
            d-axis current.
        iq_ : float
            q-axis current.
            
        Returns
        -------
        float
            Torque.
        """
        return 1.5 * p * (psi_f * iq_ + (Ld - Lq) * id_ * iq_)

    # Store results for all (speed, torque) pairs
    results = []

    # Solve OCP for each speed-torque combination
    for j, omega_val in enumerate(speed):
        for i, T_des in enumerate(trq):
            # Initialize current: use DPC data if available, else default
            if i == 0:
                i0 = np.array([-125.0, 0])
            elif sim_data is not None:
                i0 = np.array(
                    sim_data[:, :-1, :2].reshape(len(speed), len(trq), -1, 2)[j, i, 0, :]
                    * motor_env.env_properties.physical_constraints.i_d
                )
            else:
                i0 = i_traj[-1]

            T_des = T_des * motor_env.env_properties.physical_constraints.torque
            omega_val = 3 * 11000 * 2 * np.pi / 60 * speed[j]

            # ========== NLP FORMULATION ==========

            # Decision variables: flattened control trajectory
            z = ca.MX.sym("z", N * nu)
            U = ca.reshape(z, nu, N)

            # Build state trajectory via forward simulation
            X = [ca.MX(i0)]
            for k in range(N):
                X.append(X[k] + Ts * f(X[k], U[:, k], omega_val))

            # Cost function: tracking error + copper loss
            J = 0
            for k in range(N):
                J += (torque(X[k]) - T_des) ** 2 + w_cu * ca.dot(X[k], X[k])
            J += (torque(X[N]) - T_des) ** 2 + w_cu * ca.dot(X[N], X[N])

            # Constraints: current limits and voltage hexagon
            g_list = []
            theta_prev = theta0

            for k in range(N):
                xk = X[k + 1]
                idk = xk[0]
                iqk = xk[1]

                # Current magnitude constraint
                g_list.append(i_max - ca.sqrt(idk ** 2 + iqk ** 2))

                # Coordinate transformation and voltage constraints
                uk = U[:, k]
                theta_k = theta_prev + omega_val * 0.5 * Ts
                c = ca.cos(theta_k)
                s = ca.sin(theta_k)
                R_p = ca.vertcat(ca.hcat([c, -s]), ca.hcat([s, c]))
                u_ab = R_p @ uk
                u_alpha = u_ab[0]
                u_beta = u_ab[1]

                # Six hexagon constraint faces
                g_list.append(Umax - (u_alpha + (1 / np.sqrt(3)) * u_beta))
                g_list.append(Umax - ((2 / np.sqrt(3)) * u_beta))
                g_list.append(Umax - (-u_alpha + (1 / np.sqrt(3)) * u_beta))
                g_list.append(Umax - (-u_alpha - (1 / np.sqrt(3)) * u_beta))
                g_list.append(Umax - (-(2 / np.sqrt(3)) * u_beta))
                g_list.append(Umax - (u_alpha - (1 / np.sqrt(3)) * u_beta))

                theta_prev += omega_val * Ts

            g = ca.vertcat(*g_list)

            # ========== SOLVE NLP ==========

            nlp = {"x": z, "f": J, "g": g}
            solver = ca.nlpsol(
                "solver",
                "ipopt",
                nlp,
                {
                    "ipopt.print_level": 0,
                    "ipopt.tol": 1e-6,
                    "print_time": False,
                    "ipopt.sb": "yes",
                },
            )

            # Initial guess and bounds
            z0 = np.tile(np.array([0.0, 0.0]), N)
            lbg = np.zeros(g.shape[0])
            ubg = np.inf * np.ones(g.shape[0])

            # Solve
            sol = solver(x0=z0, lbg=lbg, ubg=ubg)
            z_opt = sol["x"].full().flatten()
            U_opt = z_opt.reshape(N, nu)

            # ========== POST-PROCESS SOLUTION ==========

            # Simulate with optimal control to get trajectories
            i_traj = np.zeros((N + 1, nx))
            i_traj[0] = i0
            for k in range(N):
                i_traj[k + 1] = i_traj[k] + Ts * f_np(i_traj[k], U_opt[k], omega_val)

            T_opt = np.array(
                [torque_np(i_traj[k, 0], i_traj[k, 1]) for k in range(N + 1)]
            )

            # Transform control to alpha-beta frame for analysis
            u_alpha = []
            u_beta = []
            theta_prev = theta0
            for k in range(N):
                theta_k = theta_prev + 0.5 * Ts * omega_val
                c = np.cos(theta_k)
                s = np.sin(theta_k)
                R_p = np.array([[c, -s], [s, c]])
                u_ab = R_p @ U_opt[k]
                u_alpha.append(u_ab[0])
                u_beta.append(u_ab[1])
                theta_prev += Ts * omega_val

            u_alpha = np.array(u_alpha)
            u_beta = np.array(u_beta)

            # ========== STORE RESULT ==========

            results.append(
                {
                    "i0": i0,
                    "T_des": T_des,
                    "omega": omega_val,
                    "U_opt": U_opt,
                    "i_traj": i_traj,
                    "T_opt": T_opt,
                    "u_alpha": u_alpha,
                    "u_beta": u_beta,
                }
            )
            print(f"s_{j}_t_{i}")

    return results