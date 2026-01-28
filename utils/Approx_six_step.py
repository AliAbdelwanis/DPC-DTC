"""
PWM / Space-Vector Sampling Utilities
=====================================

This module implements analytical and numerical tools for analyzing
space-vector PWM sampling on a hexagonal voltage boundary.

It contains:
- Geometry of the hexagon boundary r(theta)
- Mechanical → electrical speed conversion
- Sample-count computation for fixed switching frequency
- Fundamental voltage magnitude computation
- Visualization utilities for verification and analysis

Core numerical functions are implemented using JAX + Equinox
and are JIT-compilable where appropriate.
"""

# ============================================================
# Imports
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import equinox as eqx


# ============================================================
# HEXAGON GEOMETRY
# ============================================================

@eqx.filter_jit
def r_hexagon(theta, Vdc):
    """
    Hexagon boundary radius r(θ).

    Implements the analytical radius of the hexagonal voltage
    limit in the αβ-plane for a two-level inverter.

    Equation:
        r(θ) = (√3 · V_act) / [sin(θ_mod) + √3 cos(θ_mod)]

    where:
        V_act = 2/3 · Vdc
        θ_mod = θ mod (π/3)

    Parameters
    ----------
    theta : float or jax.Array
        Electrical angle(s) in radians.
    Vdc : float
        DC-link voltage [V].

    Returns
    -------
    r : jax.Array
        Radius of the hexagon boundary at angle θ [V].
    """
    Vact = (2.0 / 3.0) * Vdc
    theta_mod = jnp.mod(theta, jnp.pi / 3)
    denom = jnp.sin(theta_mod) + jnp.sqrt(3.0) * jnp.cos(theta_mod)
    return (jnp.sqrt(3.0) * Vact) / denom


# ============================================================
# SPEED CONVERSIONS
# ============================================================

@eqx.filter_jit
def omega_el_from_mech(rpm, pole_pairs):
    """
    Convert mechanical speed to electrical angular speed.

    Parameters
    ----------
    rpm : float
        Mechanical rotor speed [RPM].
    pole_pairs : int
        Number of pole pairs of the machine.

    Returns
    -------
    omega_el : jax.Array
        Electrical angular speed [rad/s].
    """
    omega_mech = rpm * 2 * jnp.pi / 60.0
    return pole_pairs * omega_mech


@eqx.filter_jit
def compute_n_desired(rpm, f_sw, pole_pairs):
    """
    Compute desired number of samples per electrical period.

    Definition:
        n_desired = f_sw / f_el

    where:
        f_el = ω_el / (2π)

    Parameters
    ----------
    rpm : float
        Mechanical speed [RPM].
    f_sw : float
        Switching frequency [Hz].
    pole_pairs : int
        Number of pole pairs.

    Returns
    -------
    n_desired : jax.Array
        Desired (non-integer) samples per electrical cycle.
    """
    omega_el = omega_el_from_mech(rpm, pole_pairs)
    f_el = omega_el / (2 * jnp.pi)
    return f_sw / f_el


@eqx.filter_jit
def round_n_multiple_6(n_desired):
    """
    Round sample count to nearest multiple of 6.

    This enforces alignment of sample points with hexagon vertices.

    Parameters
    ----------
    n_desired : float
        Desired samples per electrical period.

    Returns
    -------
    n : jax.Array (int32)
        Rounded sample count (minimum = 6).
    """
    n_desired = jnp.maximum(n_desired, 6.0)
    return (6 * jnp.round(n_desired / 6)).astype(jnp.int32)


# ============================================================
# FUNDAMENTAL VOLTAGE COMPUTATION
# ============================================================

@eqx.filter_jit
def fundamental_magnitude(Vdc, n, phi):
    """
    Compute fundamental voltage magnitude |V1|.

    Implements:
        |V1| = sin(π/n)/π · Σ r(θ_k)

    with:
        θ_k = 2πk/n + φ
        k = 0 … n−1

    Uses `jax.lax.fori_loop` to allow dynamic n
    inside JIT-compiled code.

    Parameters
    ----------
    Vdc : float
        DC-link voltage [V].
    n : int
        Number of samples per electrical period.
    phi : float
        Phase shift of sampling [rad].

    Returns
    -------
    V1 : jax.Array
        Fundamental voltage magnitude [V].
    """
    n = jnp.asarray(n, dtype=jnp.int32)

    def body(k, acc):
        theta_k = 2 * jnp.pi * k / n + phi
        return acc + r_hexagon(theta_k, Vdc)

    sum_r = jax.lax.fori_loop(0, n, body, 0.0)
    return (jnp.sin(jnp.pi / n) / jnp.pi) * sum_r


@eqx.filter_jit
def compute_fundamental_from_speed(
    rpm,
    Vdc=400,
    f_sw=10000,
    pole_pairs=3,
    phi=0.0,
    vertex_align=True
):
    """
    High-level fundamental voltage computation from speed.

    This function:
    1. Computes desired sample count from speed
    2. Rounds samples (optionally vertex-aligned)
    3. Computes the fundamental voltage magnitude

    Parameters
    ----------
    rpm : float
        Mechanical speed [RPM].
    Vdc : float, optional
        DC-link voltage [V].
    f_sw : float, optional
        Switching frequency [Hz].
    pole_pairs : int, optional
        Number of pole pairs.
    phi : float, optional
        Sampling phase [rad].
    vertex_align : bool, optional
        If True, enforce n multiple of 6.

    Returns
    -------
    V1 : jax.Array
        Fundamental voltage magnitude [V].
    n_desired : jax.Array
        Desired (non-integer) sample count.
    n : jax.Array (int)
        Final sample count used.
    """
    n_desired = compute_n_desired(rpm, f_sw, pole_pairs)

    n = jnp.where(
        vertex_align,
        round_n_multiple_6(n_desired),
        jnp.maximum(jnp.round(n_desired), 6).astype(jnp.int32),
    )

    V1 = fundamental_magnitude(Vdc, n, phi)
    return V1, n_desired, n


# ============================================================
# VISUALIZATION UTILITIES
# ============================================================

def plot_sample_points(
    rpm,
    Vdc=400,
    pole_pairs=3,
    f_s=10000,
    phi=0.0,
    vertex_align=True
):
    """
    Plot sampled voltage vectors on the hexagon.

    Displays:
    - Continuous hexagon boundary
    - Raw sample points (n_desired)
    - Rounded / vertex-aligned samples
    - Hexagon vertices

    Parameters
    ----------
    rpm : float
        Mechanical speed [RPM].
    Vdc : float, optional
        DC-link voltage [V].
    pole_pairs : int, optional
        Number of pole pairs.
    f_s : float, optional
        Switching frequency [Hz].
    phi : float, optional
        Sampling phase [rad].
    vertex_align : bool, optional
        If True, enforce vertex-aligned sampling.
    """

    # 1. Compute n
    n_desired = compute_n_desired(rpm, f_s, pole_pairs)
    n_desired_float = float(n_desired)

    if vertex_align:
        n_rounded = round_n_multiple_6(n_desired)
    else:
        n_rounded = jnp.maximum(jnp.round(n_desired), 6).astype(jnp.int32)

    n_rounded_int = int(n_rounded)

    # 2. Raw samples
    k_raw = np.arange(int(np.floor(n_desired_float)))
    theta_raw = 2 * np.pi * k_raw / n_desired_float + phi
    r_raw = np.array([float(r_hexagon(th, Vdc)) for th in theta_raw])

    x_raw = r_raw * np.cos(theta_raw)
    y_raw = r_raw * np.sin(theta_raw)

    # 3. Rounded samples
    k_round = np.arange(n_rounded_int)
    theta_round = 2 * np.pi * k_round / n_rounded_int + phi
    r_round = np.array([float(r_hexagon(th, Vdc)) for th in theta_round])

    x_round = r_round * np.cos(theta_round)
    y_round = r_round * np.sin(theta_round)

    # 4. Hexagon boundary
    theta_dense = np.linspace(0, 2 * np.pi, 2000)
    r_dense = np.array([float(r_hexagon(th, Vdc)) for th in theta_dense])

    x_hex = r_dense * np.cos(theta_dense)
    y_hex = r_dense * np.sin(theta_dense)

    # 5. Hexagon vertices
    vertex_angles = np.arange(6) * np.pi / 3
    r_vert = np.array([float(r_hexagon(th, Vdc)) for th in vertex_angles])

    x_vert = r_vert * np.cos(vertex_angles)
    y_vert = r_vert * np.sin(vertex_angles)

    # 6. Plot
    plt.figure(figsize=(8, 8))

    plt.plot(x_hex, y_hex, 'k-', lw=1.2, label="Hexagon Boundary")
    plt.scatter(x_raw, y_raw, c='orange', s=45,
                label=f"Raw samples (n_desired={n_desired_float:.2f})")
    plt.scatter(x_round, y_round, c='red', s=60, marker='o',
                label=f"Rounded samples (n={n_rounded_int})")
    plt.scatter(x_vert, y_vert, s=90, marker='x', c='blue',
                label="Hexagon Vertices")

    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("alpha-axis")
    plt.ylabel("beta-axis")
    plt.title(
        f"Sampled Points at RPM={rpm}\n"
        f"n_desired={n_desired_float:.2f}, n_rounded={n_rounded_int}"
    )
    plt.legend()
    plt.show()


def plot_fundamental_vs_speed(
    rpm_max=11000,
    Vdc=400,
    f_s=10000,
    pole_pairs=3
):
    """
    Plot fundamental voltage magnitude and sample count vs speed.

    Produces:
    - |V1| vs mechanical speed
    - n_desired and n_rounded vs speed

    Parameters
    ----------
    rpm_max : float, optional
        Maximum mechanical speed [RPM].
    Vdc : float, optional
        DC-link voltage [V].
    f_s : float, optional
        Switching frequency [Hz].
    pole_pairs : int, optional
        Number of pole pairs.
    """
    rpm_list = np.linspace(10, rpm_max, 300)

    V1_list = []
    n_desired_list = []
    n_rounded_list = []

    for rpm in rpm_list:
        V1, n_desired, n = compute_fundamental_from_speed(
            rpm, Vdc=Vdc, f_sw=f_s, pole_pairs=pole_pairs
        )
        V1_list.append(float(V1))
        n_desired_list.append(float(n_desired))
        n_rounded_list.append(int(n))

    # Plot |V1|
    plt.figure(figsize=(10, 5))
    plt.plot(rpm_list, V1_list, lw=2, label="|V1| (rounded n)")

    Vact = 2 / 3 * Vdc
    V_linear = np.sqrt(3) / 2 * Vact
    plt.axhline(V_linear, color='green', linestyle='--',
                label=f"Linear modulation limit = {V_linear:.1f} V")

    r0 = float(r_hexagon(0, Vdc))
    V_six = 3 / np.pi * r0
    plt.axhline(V_six, color='red', linestyle='--',
                label=f"Six-step limit = {V_six:.1f} V")

    plt.xlabel("Mechanical speed [RPM]")
    plt.ylabel("|V1| [V]")
    plt.title("Fundamental Voltage vs Speed")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot n vs speed
    plt.figure(figsize=(10, 5))
    plt.plot(rpm_list, n_desired_list, lw=2, label="n_desired (fs/fel)")
    plt.plot(rpm_list, n_rounded_list, lw=2, label="n_rounded (multiple of 6)")
    plt.ylim((0, 100))
    plt.xlabel("Mechanical speed [RPM]")
    plt.ylabel("Samples per electrical cycle")
    plt.title("Sample Count vs Speed")
    plt.grid(True)
    plt.legend()
    plt.show()
