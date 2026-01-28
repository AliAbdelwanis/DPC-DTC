"""
Analytical reference generation using Lagrange multiplier optimization.

This module implements analytical solutions for finding optimal dq-current
references in PMSM control. It solves constrained optimization problems
arising from the intersection of multiple quadratic constraints (torque,
current limit, voltage limit, etc.) using Lagrange multiplier theory.

The core algorithm:
1. Formulates constraint surfaces as quadratic equations
2. Computes Lagrange multiplier equations (up to quartic polynomials)
3. Solves polynomial equations analytically (quadratic, cubic, quartic)
4. Selects feasible solutions based on motor constraints

This approach enables real-time optimal reference generation with guaranteed
convergence and is JIT-compilable with JAX.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from jax import lax

jax.config.update("jax_enable_x64", True)


# ============================================================
# Utility Functions for Numerical Stability
# ============================================================


@eqx.filter_jit
def jnp_general_cbrt(x: jnp.ndarray) -> jnp.ndarray:
    """Compute cube root handling both real and complex numbers.

    For real numbers, uses JAX's cbrt function. For complex numbers,
    uses complex exponentiation.

    Parameters
    ----------
    x : jnp.ndarray
        Input scalar or array.

    Returns
    -------
    jnp.ndarray
        Cube root of x, preserving input shape and dtype.
    """
    out = jnp.where(jnp.isreal(x), jnp.cbrt(jnp.real(x)), x ** (1 / 3))
    return out


@eqx.filter_jit
def jnp_sqrt(x: jnp.ndarray) -> jnp.ndarray:
    """Compute square root with support for negative real inputs.

    For positive inputs, returns real square root. For negative inputs,
    uses complex arithmetic to return the principal square root.

    Parameters
    ----------
    x : jnp.ndarray
        Input scalar or array.

    Returns
    -------
    jnp.ndarray
        Square root of x (possibly complex).
    """
    x = jnp.asarray(x)
    return jnp.where(x >= 0, jnp.sqrt(x), jnp.sqrt(x + 0j))


@eqx.filter_jit
def jnp_real_if_close(
    x: jnp.ndarray, tol: float = 100
) -> jnp.ndarray:
    """Extract real part if imaginary part is negligible.

    Handles numerical errors in polynomial solvers where mathematically
    real roots are returned with tiny spurious imaginary components.

    Parameters
    ----------
    x : jnp.ndarray
        Complex or real input.
    tol : float, optional
        Relative tolerance threshold (multiple of machine epsilon).
        Default is 100.

    Returns
    -------
    jnp.ndarray
        Real part if imaginary part is below threshold, else original value.
    """
    x = jnp.asarray(x)
    real_part = jnp.real(x)
    imag_part = jnp.imag(x)
    eps = jnp.finfo(real_part.dtype).eps
    threshold = tol * eps

    result = jnp.where(jnp.abs(imag_part) <= threshold, real_part, x)
    return result


# ============================================================
# Quadratic Surface Definitions
# ============================================================


@eqx.filter_jit
def jnp_generate_quads(
    ld: float,
    lq: float,
    Rs: float,
    n_p: int,
    lambda_p: float,
    u_m: float,
    i_m: float,
    omega_k: float,
    m_ref: float,
) -> dict:
    """Generate quadratic matrices for constraint surfaces.

    Computes the quadratic form (x^T D x + 2 d^T x) for each constraint surface:
    - Torque constraint (T = m_ref)
    - Current magnitude constraint (||i_dq|| <= i_m)
    - Voltage constraint (||u_dq|| <= u_m)
    - MTPC (Maximum Torque Per Current) operating boundary
    - MTPV (Maximum Torque Per Voltage) operating boundary

    Parameters
    ----------
    ld, lq : float
        d-axis and q-axis inductances [H].
    Rs : float
        Stator resistance [Ohm].
    n_p : int
        Number of pole pairs.
    lambda_p : float
        Permanent magnet flux linkage [Wb].
    u_m : float
        Voltage limit [V].
    i_m : float
        Current limit [A].
    omega_k : float
        Electrical angular velocity [rad/s].
    m_ref : float
        Desired torque [Nm].

    Returns
    -------
    dict
        Dictionary of 11 matrix/scalar entries for constraint surfaces:
        - "T", "t", "tau": torque constraint matrices and constant
        - "V", "v", "v_scalar": voltage constraint matrices
        - "Mc", "mc": MTPC matrices
        - "MV", "mV", "muV": MTPV matrices
    """
    Ls = jnp.array([[ld,0],[0,lq]])
    J= jnp.array([[0,-1],[1,0]])
    psi_p_v = jnp.array([lambda_p,0])
    Iden = jnp.array([[1,0],[0,1]]) 

    #Torque hyper 
    T = (3/4)*n_p*(J@Ls + Ls@J.T)
    t = (3/2)*n_p*jnp.array([0,lambda_p/2])
    tau = -m_ref


    #Voltage hyper 
    V = Rs**2 * Iden + Rs*omega_k*(J@Ls + Ls@J.T) + omega_k**2 * Ls**2
    v = (omega_k*psi_p_v.T@(omega_k*Ls + Rs*J.T)).T
    v_scalar = (omega_k**2 * psi_p_v.T @ J.T @ J @ psi_p_v - jnp.array([u_m**2]))[0]
    

    #MTPC hyper 
    Mc= (3/2)*n_p*jnp.array([[(ld-lq)/2,0],[0,-(ld-lq)/2]])
    mc=(3/2)*n_p*jnp.array([lambda_p/4,0])


    #MTPV hyper
    MV = (3/2)*n_p*jnp.array([[(ld-lq)*(Rs**2 + omega_k**2 * ld**2)/2,0],[0,-(ld-lq)*(Rs**2 + omega_k**2 * lq**2)/2]])
    mV = (3/2)*n_p*jnp.array([(Rs**2 + 2*omega_k**2 *ld**2 - omega_k**2 *ld*lq)*lambda_p /4,0])
    muV = (3/4)*n_p*omega_k**2 *ld *lambda_p**2
    
    matrices = {
        "T":T,
        "t":t,
        "tau":tau,
        "V":V,
        "v":v,
        "v_scalar": v_scalar,
        "Mc": Mc,
        "mc":mc,
        "MV":MV,
        "mV":mV,
        "muV":muV,

    }
    return matrices

#Generate quads' matrices for a given operating mode in order to calculate their intersection
def jnp_generate_DdMm(
    quad_name: str, matrices: dict, values: dict
) -> tuple:
    """Extract constraint matrices for a specific operating mode.

    Given a mode (MTPC, MC, FW, MTPV), returns the quadratic matrix D,
    linear vector d, and associated terms (M, m, mu) for the Lagrange
    multiplier formulation.

    Parameters
    ----------
    quad_name : str
        Operating mode name: 'MTPC', 'MC' (constant power), 'FW' (field weakening),
        or 'MTPV'.
    matrices : dict
        Dictionary of pre-computed constraint matrices from jnp_generate_quads.
    values : dict
        Dictionary with motor parameters including m_ref (desired torque).

    Returns
    -------
    tuple
        (D, d, M, m, mu) - quadratic matrices and constants for the
        Lagrange formulation.
    """
    match quad_name:
        case 'MTPC':
            D = matrices["Mc"]
            d = matrices["mc"]
            M = matrices["T"]
            m = matrices["t"]
            mu = jnp.array([-values["m_ref"]])
        case 'MC':
            D = ((jnp.eye(2)/(-values["i_m"]**2))-(matrices["V"]/matrices["v_scalar"]))
            d = (jnp.zeros((2,))-(matrices["v"]/matrices["v_scalar"]))
            
            M = jnp.eye(2)
            m = jnp.zeros((2,1))
            mu = jnp.array([-values["i_m"]**2])
        case 'FW':
            D = ((matrices["V"]/matrices["v_scalar"])-(matrices["T"]/matrices["tau"]))
            d = ((matrices["v"]/matrices["v_scalar"])-(matrices["t"]/matrices["tau"]))
            M = matrices["T"]
            m = matrices["t"]
            mu = jnp.array([-values["m_ref"]])
        case 'MTPV':
            D = ((matrices["MV"]/matrices["muV"])-(matrices["V"]/matrices["v_scalar"]))
            d = ((matrices["mV"]/matrices["muV"])-(matrices["v"]/matrices["v_scalar"]))
            M = matrices["MV"]
            m = matrices["mV"]
            mu = jnp.array([matrices["muV"]])
    return D,d,M,m,mu

#Generate lagrangian coefficients
@eqx.filter_jit
def jnp_gen_lagrang_coeff(
    D: jnp.ndarray,
    d: jnp.ndarray,
    M: jnp.ndarray,
    m: jnp.ndarray,
    mu: jnp.ndarray,
) -> jnp.ndarray:
    """Generate coefficients of the Lagrange multiplier quartic equation.

    From the Lagrange condition (D + lambda*M) is singular, derives a
    quartic equation in lambda whose roots are the Lagrange multipliers.

    Parameters
    ----------
    D, M : jnp.ndarray
        2x2 constraint matrices.
    d, m : jnp.ndarray
        2x1 linear coefficient vectors.
    mu : jnp.ndarray
        Scalar constants from constraints.

    Returns
    -------
    jnp.ndarray
        Coefficients [c4, c3, c2, c1, c0] of the quartic polynomial
        (c4*lambda^4 + c3*lambda^3 + ... + c0 = 0).
    """
    eps_4 = mu.reshape(1,)
    eps_3 = (4 * (m[0]*d[1]- m[1]*d[0])).reshape(1,)
    eps_2 = (4*M[1,1]*d[0]**2 - 8*M[0,1]*d[0]*d[1] + 4*m[1]*d[0]*D[0,1] - 4*m[0]*D[1,1]*d[0]\
            + 4*M[0,0]*d[1]**2 + 4*m[0]*d[1]*D[0,1] - 4*m[1]*D[0,0]*d[1] - 2*mu*D[0,1]**2 + 2*mu*D[0,0]*D[1,1]).reshape(1,)
    eps_1 = (4*m[1]*d[0]*D[0,1]**2 - 4*m[0]*d[1]*D[0,1]**2 + 8*M[0,0]*(d[1]**2)*D[0,1] - 8*M[0,1]*(d[1]**2)*D[0,0]\
            + 8*M[0,1]*(d[0]**2)*D[1,1] - 8*M[1,1]*(d[0]**2)*D[0,1] + 4*m[0]*d[1]*D[0,0]*D[1,1] - 4*m[1]*d[0]*D[0,0]*D[1,1]\
            - 8*M[0,0]*d[0]*d[1]*D[1,1] + 8*M[1,1]*d[0]*d[1]*D[0,0]).reshape(1,)
    eps_0 = (4*M[1,1]*(d[0]**2)*D[0,1]**2 - 8*M[0,1]*(d[0]**2)*D[0,1]*D[1,1] + 4*M[0,0]*(d[0]**2)*D[1,1]**2\
            - 8*M[1,1]*d[0]*d[1]*D[0,0]*D[0,1] + 8*M[0,1]*d[0]*d[1]*D[0,0]*D[1,1] + 8*M[0,1]*d[0]*d[1]*D[0,1]**2\
            - 8*M[0,0]*d[0]*d[1]*D[0,1]*D[1,1] + 4*m[1]*d[0]*D[0,0]*D[0,1]*D[1,1] - 4*m[0]*d[0]*D[0,0]*D[1,1]**2\
            - 4*m[1]*d[0]*D[0,1]**3 + 4*m[0]*d[0]*(D[0,1]**2)*D[1,1] + 4*M[1,1]*(d[1]**2)*D[0,0]**2\
            - 8*M[0,1]*(d[1]**2)*D[0,0]*D[0,1] + 4*M[0,0]*(d[1]**2)*D[0,1]**2 - 4*m[1]*d[1]*(D[0,0]**2)*D[1,1]\
            + 4*m[1]*d[1]*D[0,0]*D[0,1]**2 + 4*m[0]*d[1]*D[0,0]*D[0,1]*D[1,1] - 4*m[0]*d[1]*D[0,1]**3 + mu*(D[0,0]**2)*D[1,1]**2\
            - 2*mu*D[0,0]*(D[0,1]**2)*D[1,1] + mu*D[0,1]**4).reshape(1,)
    
    coeff = jnp.concatenate((eps_4, eps_3, eps_2, eps_1, eps_0))
    return coeff

# Solve quadratic polynomial analytically
@eqx.filter_jit
def solve_quad(
    b: float, c: float, a: float = 1, check: bool = False
) -> jnp.ndarray:
    """Solve quadratic equation a*x^2 + b*x + c = 0.

    Uses the standard quadratic formula with careful handling of
    numerical precision.

    Parameters
    ----------
    b : float
        Linear coefficient.
    c : float
        Constant term.
    a : float, optional
        Quadratic coefficient. Default is 1.
    check : bool, optional
        Unused, kept for interface compatibility.

    Returns
    -------
    jnp.ndarray
        Array of two roots [x1, x2].
    """
    x1 = -(b / (2 * a)) + jnp_sqrt(b ** 2 - 4 * a * c) / (2 * a)
    x2 = -(b / (2 * a)) - jnp_sqrt(b ** 2 - 4 * a * c) / (2 * a)

    return jnp.array([x1, x2])


#Solve cubic polynomial analytically
@eqx.filter_jit
def solve_cubic(
    d2: float, d1: float, d0: float, check: bool = False
) -> tuple:
    """Solve cubic equation z^3 + d2*z^2 + d1*z + d0 = 0.

    Uses Cardano's formula with complex number support for all root cases.

    Parameters
    ----------
    d2, d1, d0 : float
        Coefficients of the monic cubic.
    check : bool, optional
        Unused, kept for interface compatibility.

    Returns
    -------
    tuple
        Three roots (z1, z2, z3), may be complex.
    """
    q = d1 / 3 - d2 ** 2 / 9
    r = (d1 * d2 - 3 * d0) / 6 - d2 ** 3 / 27
    in_sqrt = jnp_sqrt(q ** 3 + r ** 2)
    s1 = jnp_general_cbrt(r + in_sqrt)
    s2 = jnp_general_cbrt(r - in_sqrt)
    z1 = (s1 + s2) - d2 / 3
    z2 = -0.5 * (s1 + s2) - d2 / 3 + 1j * (jnp.sqrt(3) / 2) * (s1 - s2)
    z3 = -0.5 * (s1 + s2) - d2 / 3 - 1j * (jnp.sqrt(3) / 2) * (s1 - s2)

    return z1, z2, z3


# Solve quartics analytically
@eqx.filter_jit
def solve_quartic(
    c4: float, c3: float, c2: float, c1: float, c0: float, check: bool = False
) -> jnp.ndarray:
    """Solve quartic equation c4*x^4 + c3*x^3 + c2*x^2 + c1*x + c0 = 0.

    Uses Ferrari's method with a reduced cubic resolution. Includes
    adaptive sign selection for numerical stability.

    Parameters
    ----------
    c4, c3, c2, c1, c0 : float
        Quartic polynomial coefficients.
    check : bool, optional
        Unused, kept for interface compatibility.

    Returns
    -------
    jnp.ndarray
        Array of four roots (may be complex).
    """
    #coeffs of the quartics with coeff of g**4 is set to 1 -> g**4 + a3*g**3 + a1*g**2 + a1*g + a0 = 0 
    a4 = 1
    a3 = c3/c4
    a2 = c2/c4
    a1 = c1/c4
    a0 = c0/c4

    #calculate the coeffs of the cubic polynomial z**3 + d2*z**2 + d1*z + d0 = 0
    p = (1/a4**2)*((a2*a4) - ((3*a3**2)/8))
    q = (1/a4**3)*(((a3**3)/8) - (a2*a3*a4/2) + (a1*a4**2))
    r = (1/a4**4)*(((-3*a3**4)/256) + ((a4**3) * a0) - ((a4**2) * a3 * a1/4)  + (a4 * (a3**2) * a2/16))
    d2 = 2*p
    d1 = (p**2) - 4*r
    d0 = -q**2

    #solve the cubic polynomial
    z1, z2, z3= solve_cubic(d2,d1,d0, check=check)
    
    #choose the signs corresponding to the solution g1, g2, g3, g4
    l=0
    g1 = ((-1)**l /2)*(jnp_sqrt(z1)+jnp_sqrt(z2)+jnp_sqrt(z3)) - (a3/(4*a4))
    g2 = ((-1)**l /2)*(jnp_sqrt(z1)-jnp_sqrt(z2)-jnp_sqrt(z3)) - (a3/(4*a4))
    g3 = ((-1)**l /2)*(-jnp_sqrt(z1)+jnp_sqrt(z2)-jnp_sqrt(z3)) - (a3/(4*a4))
    g4 = ((-1)**l /2)*(-jnp_sqrt(z1)-jnp_sqrt(z2)+jnp_sqrt(z3)) - (a3/(4*a4))

    cond = ~(jnp.isclose(-a3*10**15, (g1 + g2 + g3 + g4)*10**15, atol=1e-6) &
            jnp.isclose(a0*10**15,jnp_real_if_close((g1*g2*g3*g4)*10**15), atol=1e-6) &
            jnp.isclose(-a1*10**15, jnp_real_if_close((g1*g2*g3 + g1*g2*g4 + g1*g3*g4 + g2*g3*g4))*10**15,atol=1e-6) & 
            jnp.isclose(a2*10**15,jnp_real_if_close((g1*g2 + g1*g3 + g1*g4 + g2*g3 + g2*g4 + g3*g4))*10**15, atol=1e-6))
   # print("c1", jnp.isclose(-a3*10**15, (g1 + g2 + g3 + g4)*10**15, atol=1e-6),-a3*10**15, (g1 + g2 + g3 + g4)*10**15)
   # print("c2", jnp.isclose(a0*10**15,jnp_real_if_close((g1*g2*g3*g4))*10**15, atol=1e-6), a0*10**15,jnp_real_if_close((g1*g2*g3*g4))*10**15)
   # print("c3", jnp.isclose(-a1*10**15, jnp_real_if_close((g1*g2*g3 + g1*g2*g4 + g1*g3*g4 + g2*g3*g4))*10**15,atol=1e-6), -a1*10**15, jnp_real_if_close((g1*g2*g3 + g1*g2*g4 + g1*g3*g4 + g2*g3*g4))*10**15)
   # print("c4", jnp.isclose(a2*10**15,jnp_real_if_close((g1*g2 + g1*g3 + g1*g4 + g2*g3 + g2*g4 + g3*g4))*10**15, atol=1e-6), a2*10**15,jnp_real_if_close((g1*g2 + g1*g3 + g1*g4 + g2*g3 + g2*g4 + g3*g4))*10**15)
    def true_fn(_):
        l=1
        g1 = ((-1)**l /2)*(jnp_sqrt(z1)+jnp_sqrt(z2)+jnp_sqrt(z3)) - (a3/(4*a4))
        g2 = ((-1)**l /2)*(jnp_sqrt(z1)-jnp_sqrt(z2)-jnp_sqrt(z3)) - (a3/(4*a4))
        g3 = ((-1)**l /2)*(-jnp_sqrt(z1)+jnp_sqrt(z2)-jnp_sqrt(z3)) - (a3/(4*a4))
        g4 = ((-1)**l /2)*(-jnp_sqrt(z1)-jnp_sqrt(z2)+jnp_sqrt(z3)) - (a3/(4*a4))
        return g1, g2, g3, g4
    
    def false_fn(_):
        return g1, g2, g3, g4
    
    #Calculate the solution for the correct signs
    g1, g2, g3, g4 = lax.cond(cond, true_fn, false_fn, operand = None)
    
    return jnp.array([g1, g2, g3, g4])

# Calculate lagrangian multipliers analytically
@eqx.filter_jit
def jnp_calc_gamma_analy(
    c: jnp.ndarray, check: bool = False
) -> jnp.ndarray:
    """Compute Lagrange multipliers by solving the quartic equation.

    Parameters
    ----------
    c : jnp.ndarray
        Quartic coefficients [c4, c3, c2, c1, c0] from jnp_gen_lagrang_coeff.
    check : bool, optional
        Unused, kept for interface compatibility.

    Returns
    -------
    jnp.ndarray
        Array of four Lagrange multiplier values (may be complex).
    """
    g = solve_quartic(c[0], c[1], c[2], c[3], c[4], check=check)
    g = jnp.array([jnp_real_if_close(x) for x in g])
    return g


#Calculate intersection of two quads, given all roots (gamma)
@eqx.filter_jit
def jnp_calc_intersection(
    D: jnp.ndarray, d: jnp.ndarray, gamma: jnp.ndarray, values: dict
) -> tuple:
    """Find feasible current reference from constraint intersections.

    For each real Lagrange multiplier gamma, solves for the corresponding
    dq-current and checks feasibility (torque sign, current limit, maximum
    torque). Returns the solution with maximum torque magnitude.

    Parameters
    ----------
    D : jnp.ndarray
        2x2 quadratic matrix.
    d : jnp.ndarray
        2x1 linear vector.
    gamma : jnp.ndarray
        Array of Lagrange multipliers.
    values : dict
        Motor parameters including i_m (current limit), n_p (pole pairs),
        lambda_p, ld, lq, m_ref.

    Returns
    -------
    tuple
        (idq_best, torque_best) where idq_best is the optimal dq-current
        and torque_best is the achieved torque.
    """
    i_max = values["i_m"] + 1e-4
    t_best = jnp.complex128(0)
    idq_best = jnp.ones((2,), dtype=jnp.complex128)*10000   #10000 is an arbitrary value
  
    def true_fn(args):
        t_best, idq_best, gam, values= args
        mat = jnp.array([[D[1,1],-D[0,1]-gam],
                [gam-D[0,1], D[0,0]]])
        inverse_matrix = (1/(gam**2 + D[0,0]*D[1,1] - D[0,1]**2)) * mat
        i_dq = jnp.array(-2 * inverse_matrix @  d)
        torque = 1.5 * values["n_p"] * (values["lambda_p"] * i_dq[1] + (values["ld"] - values["lq"]) * i_dq[0] * i_dq[1])
        cond = ((values["m_ref"]*torque > 0) &
                (jnp.sqrt(i_dq[0]**2 + i_dq[1]**2)<=i_max) &
                (jnp.sqrt(i_dq[0]**2 + i_dq[1]**2)<=jnp.sqrt(idq_best[0]**2 + idq_best[1]**2)) &
                (jnp.abs(torque) > jnp.abs(t_best)))
        
        t_best, idq_best=lax.cond(cond, lambda args: (args[2], args[3]), lambda args: (args[0], args[1]), operand =(t_best, idq_best, torque, i_dq))
        return t_best, idq_best
    for gam in gamma:
        cond_gam = jnp.isreal(gam) 
        t_best, idq_best = lax.cond(cond_gam, true_fn, lambda args: (args[0], args[1]), operand=(t_best, idq_best,gam, values))
    return idq_best, t_best 

#For the given operation mode, calculate the intersection of the corresponding quads 
@eqx.filter_jit
def jnp_ref_values(values: dict, op_mode: str) -> tuple:
    """Compute optimal dq-current reference for a given operating mode.

    Solves the constrained optimization problem for a single operating mode
    and returns the optimal current and feasible torque.

    Parameters
    ----------
    values : dict
        Motor parameters and operating point specifications.
    op_mode : str
        Operating mode: 'MTPC', 'MC', 'FW', or 'MTPV'.

    Returns
    -------
    tuple
        (idq_optimal, torque_feasible) - optimal dq-current and achievable torque.
    """
    matrices = jnp_generate_quads(**values)
    D, d, M, m, mu = jnp_generate_DdMm(op_mode, matrices, values)
    coeff = jnp_gen_lagrang_coeff(D, d, M, m, mu)
    gamma = jnp_calc_gamma_analy(coeff)
    idq, m_feasible = jnp_calc_intersection(D, d, gamma, values)
    return idq, m_feasible


def jnp_ref_values_num(values: dict, op_mode: str) -> tuple:
    """Numerical version of jnp_ref_values using NumPy root finding.

    Used for verification and in cases where analytical solution is unstable.

    Parameters
    ----------
    values : dict
        Motor parameters.
    op_mode : str
        Operating mode name.

    Returns
    -------
    tuple
        (idq_optimal, torque_feasible).
    """
    matrices = jnp_generate_quads(**values)
    D, d, M, m, mu = jnp_generate_DdMm(op_mode, matrices, values)
    coeff = jnp_gen_lagrang_coeff(D, d, M, m, mu)
    gamma = jnp.roots(coeff)
    idq, m_feasible = jnp_calc_intersection(D, d, gamma[jnp.isreal(gamma)], values)
    return idq, m_feasible


#calculate the nominal torque at low speeds, MTPC
@eqx.filter_jit
def jnp_calc_m_nom(values: dict) -> float:
    """Compute nominal (maximum continuous) torque.

    Calculates the maximum torque achievable at low speeds under current
    limit constraint, used as a threshold for operating mode selection.

    Parameters
    ----------
    values : dict
        Motor parameters.

    Returns
    -------
    float
        Nominal torque magnitude.
    """
    matrices = jnp_generate_quads(**values)
    D = matrices["Mc"]
    d = matrices["mc"]
    M = jnp.eye(2)
    m = jnp.zeros((2, 1))
    mu = jnp.array([-values["i_m"] ** 2])
    coeff = jnp_gen_lagrang_coeff(D, d, M, m, mu)
    gamma = jnp_calc_gamma_analy(coeff)
    _, m_nom = jnp_calc_intersection(D, d, gamma, values)
    return m_nom


#calculate the cut-off omega for MTPC 
@eqx.filter_jit
def jnp_calc_omega_MTPC_feas(
    values: dict,
    ld: float,
    lq: float,
    Rs: float,
    n_p: int,
    lambda_p: float,
    u_m: float,
    i_m: float,
    omega_k: float,
    m_ref: float,
) -> tuple:
    """Compute speed limit above which MTPC mode becomes infeasible.

    Determines the maximum speed for operating under constant torque
    constraint due to voltage limits. Above this speed, field weakening
    is necessary.

    Parameters
    ----------
    values : dict
        Motor parameters dictionary.
    ld, lq : float
        Inductances.
    Rs : float
        Stator resistance.
    n_p : int
        Pole pairs.
    lambda_p : float
        Permanent magnet flux.
    u_m : float
        Voltage limit.
    i_m : float
        Current limit.
    omega_k : float
        Current electrical speed (unused for calculation).
    m_ref : float
        Reference torque (unused for calculation).

    Returns
    -------
    tuple
        (idq_mtpc, m_mtpc, omega_MTPC_feas) - MTPC current and torque,
        and array of speed limits.
    """
    idq_mtpc, m_mtpc = jnp_ref_values(values, "MTPC")
    coeff2 = (
        2 * idq_mtpc[0] * ld * lambda_p
        + idq_mtpc[0] ** 2 * ld ** 2
        + idq_mtpc[1] ** 2 * lq ** 2
        + lambda_p ** 2
    )
    coeff1 = (
        Rs ** 2 * idq_mtpc[0] ** 2 * idq_mtpc[1] ** 2 * (ld - lq) ** 2
        + 2 * Rs * idq_mtpc[1] * lambda_p
    )
    coeff0 = Rs ** 2 * idq_mtpc[0] ** 2 + Rs ** 2 * idq_mtpc[1] ** 2 - u_m ** 2
    omega_MTPC_feas = solve_quad(coeff1 / coeff2, coeff0 / coeff2)
    return idq_mtpc, m_mtpc, omega_MTPC_feas


def jnp_calc_omega_MTPC_feas_num(
    values: dict,
    ld: float,
    lq: float,
    Rs: float,
    n_p: int,
    lambda_p: float,
    u_m: float,
    i_m: float,
    omega_k: float,
    m_ref: float,
) -> tuple:
    """Numerical version of jnp_calc_omega_MTPC_feas using NumPy.

    Parameters
    ----------
    values : dict
        Motor parameters.
    ld, lq : float
        Inductances.
    Rs : float
        Resistance.
    n_p : int
        Pole pairs.
    lambda_p : float
        Flux linkage.
    u_m : float
        Voltage limit.
    i_m : float
        Current limit.
    omega_k : float
        Electrical speed.
    m_ref : float
        Reference torque.

    Returns
    -------
    tuple
        (idq_mtpc, m_mtpc, omega_MTPC_feas_roots).
    """
    idq_mtpc, m_mtpc = jnp_ref_values_num(values, "MTPC")
    coeff2 = 2 * idq_mtpc[0] * ld * lambda_p + idq_mtpc[0] ** 2 * ld ** 2 + idq_mtpc[1] ** 2 * lq ** 2 + lambda_p ** 2
    coeff1 = Rs ** 2 * idq_mtpc[0] ** 2 * idq_mtpc[1] ** 2 * (ld - lq) ** 2 + 2 * Rs * idq_mtpc[1] * lambda_p
    coeff0 = Rs ** 2 * idq_mtpc[0] ** 2 + Rs ** 2 * idq_mtpc[1] ** 2 - u_m ** 2
    omega_MTPC_feas = jnp.roots(jnp.array([coeff2, coeff1, coeff0]))
    return idq_mtpc, m_mtpc, omega_MTPC_feas


# Manage the operation online, determine the operating mode, given the speed and desired torque, and calculate the ref. current and torque
@eqx.filter_jit
def jnp_operation_management(
    values: dict,
    m_ref: float,
    omega_k: float,
    omega_cutin_MTPV: float = 3754.03,
    m_nom: float = 171.52579,
) -> tuple:
    """Select optimal operating mode and compute reference current.

    Implements mode selection logic: MTPC (low speed, constant torque) →
    Constant power → Field weakening (high speed).

    Parameters
    ----------
    values : dict
        Motor parameters dictionary.
    m_ref : float
        Desired torque [Nm].
    omega_k : float
        Electrical angular velocity [rad/s].
    omega_cutin_MTPV : float, optional
        Speed threshold for MTPV mode. Default 3754.03 rad/s.
    m_nom : float, optional
        Nominal torque for limiting references. Default 171.52579 Nm.

    Returns
    -------
    tuple
        (idq_reference, m_feasible) - optimal dq-current reference and
        the torque that can be achieved with that current.
    """
    mref = m_ref
    mref = jnp.where(jnp.abs(mref)>m_nom,jnp.sign(mref)*m_nom,mref)
    values.update({"m_ref": mref, "omega_k": omega_k})
    #m_nom = jnp.abs(jnp_calc_m_nom(values))
    idq_mtpc, m_mtpc, omega_MTPC_feas= jnp_calc_omega_MTPC_feas(values,**values)
    omega_MTPC_feas = jnp.min(jnp.where( omega_MTPC_feas>0,  omega_MTPC_feas, jnp.inf))
    #omega_cut_in_mtpv = calc_omega_MTPV_cut_in(values_omega)
    #print('omega_MTPV_cut_in:',omega_cut_in_mtpv)
   
    
    idq_MC, m_fw_feas = jnp_ref_values(values, 'MC')
    idq_mtpv, m_mtpv_cut_in = jnp_ref_values(values, 'MTPV')
    idq_fw, m_fw = jnp_ref_values(values, 'FW')

    i_dq_all = {
        "idq_mtpc": idq_mtpc,
        "idq_MC": idq_MC,
        "idq_mtpv": idq_mtpv,
        "idq_fw": idq_fw
    }
    m_all = {
        "mref": mref,
        "m_mtpc": m_mtpc,
        "m_fw_feas": m_fw_feas,
        "m_mtpv_cut_in": m_mtpv_cut_in,
        "m_fw": m_fw
    }
 
   
    def cutin_omega_MTPV(args):
        return lax.cond(jnp.abs(args[1]["mref"])<=jnp.abs(args[1]["m_fw_feas"]), lambda args: (args[0]["idq_fw"],args[1]["m_fw"]),\
                        lambda args: (args[0]["idq_MC"],args[1]["m_fw_feas"]), operand=args)
    
   
    return lax.cond((values["omega_k"]<=omega_MTPC_feas), lambda args: (args[0]["idq_mtpc"],args[1]["m_mtpc"]),
                    lambda _: lax.cond(values["omega_k"]<= omega_cutin_MTPV,cutin_omega_MTPV,
                    lambda _: lax.cond(mref<m_mtpv_cut_in, lambda args: (args[0]["idq_fw"],args[1]["m_fw"]), \
                                       lambda args: (args[0]["idq_mtpv"],args[1]["m_mtpv_cut_in"]), operand=(i_dq_all, m_all)),\
                                        operand=(i_dq_all, m_all)), operand = (i_dq_all, m_all))