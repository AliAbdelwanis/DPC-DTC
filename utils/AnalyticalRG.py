import sympy as sp
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
#Motor parameters
ld,lq, n_p, lambda_p, m_ref, Rs, omega_k, u_m, i_m = sp.symbols('l_d l_q n_p psi_p m_ref R_s omega_k u_max i_max')

#States
id,iq = sp.symbols('i_d i_q', real=True)
X = sp.Matrix([id,iq])

#common matrices and vectors
Ls = sp.Matrix([[ld,0],[0,lq]])
J= sp.Matrix([[0,-1],[1,0]])
psi_p_v = sp.Matrix([lambda_p,0])
I = sp.Matrix([[1,0],[0,1]]) 

#Torque hyper 
T = (3/4)*n_p*(J*Ls + Ls*J.T)
t = (3/2)*n_p*sp.Matrix([0,lambda_p/2])
tau = -m_ref
trq_quad = X.T *T*X + 2*t.T *X + sp.Matrix([[-m_ref]])
torque = sp.simplify(trq_quad[0,0])  # extract scalar from 1x1 matrix

#current hyper
current_quad = X.T *I*X + sp.Matrix([[-i_m**2]])
current_circ = sp.simplify(current_quad[0,0])

#Voltage hyper 
V = Rs**2 * I + Rs*omega_k*(J*Ls + Ls*J.T) + omega_k**2 * Ls**2
v = (omega_k*psi_p_v.T*(omega_k*Ls + Rs*J.T)).T
v_scalar = (omega_k**2 * psi_p_v.T * J.T * J * psi_p_v - sp.Matrix([u_m**2]))[0]
voltage_quad = X.T *V*X + 2*v.T *X + sp.Matrix([[v_scalar]])
volt_ellipse = sp.simplify(voltage_quad[0,0])

#MTPC hyper 
Mc= (3/2)*n_p*sp.Matrix([[(ld-lq)/2,0],[0,-(ld-lq)/2]])
mc=(3/2)*n_p*sp.Matrix([lambda_p/4,0])
MTPC_quad = X.T *Mc*X + 2*mc.T *X
MTPC = sp.simplify(MTPC_quad[0,0])

#MTPV hyper
MV = (3/2)*n_p*sp.Matrix([[(ld-lq)*(Rs**2 + omega_k**2 * ld**2)/2,0],[0,-(ld-lq)*(Rs**2 + omega_k**2 * lq**2)/2]])
mV = (3/2)*n_p*sp.Matrix([(Rs**2 + 2*omega_k**2 *ld**2 - omega_k**2 *ld*lq)*lambda_p /4,0])
muV = (3/4)*n_p*omega_k**2 *ld *lambda_p**2
MTPV_quad = X.T *MV*X + 2*mV.T *X + sp.Matrix([[muV]])
MTPV = sp.simplify(MTPV_quad[0,0])

def plot_intersection(values, mref, mdes, i_dq, Ref=True, ranges=[350, 250], figsize=(8.5, 5.6)):
    # Labels / styles
    labels = [
        r'MTPC',
        r'$u^\mathrm{approx.}_\mathrm{six-step}$',
        r'Current circle',
        r'MTPV'
    ]
    colors = ['steelblue', 'lightgreen', 'black', 'brown']
    markers = [None, None, None, None]

    # Base implicit functions
    funcs_sym = [
        MTPC.subs(values),
        volt_ellipse.subs(values),
        current_circ.subs(values),
        MTPV.subs(values)
    ]

    # Reference torque curves
    cmap = cm.get_cmap('cool', len(mref))
    for i, tref in enumerate(mref):
        values_t = values.copy()
        values_t[m_ref] = tref    

        funcs_sym.append(torque.subs(values_t))
        labels.append(r'$T_\mathrm{feas}^*$')
        colors.append(mcolors.to_hex(cmap(i)))
        markers.append(None)
    if mdes > mref:
        # Desired Torque curves
        cmap = cm.get_cmap('plasma', len(mdes))
        for i, tref in enumerate(mdes):
            values_t = values.copy()
            values_t[m_ref] = tref    

            funcs_sym.append(torque.subs(values_t))
            labels.append(r'$T_\mathrm{infeas}^*$')
            colors.append(mcolors.to_hex(cmap(i)))
            markers.append(None)

    # ------------------------------------------------------------------
    # Six-step voltage
    values_over_mod = values.copy()
    values_over_mod[u_m] = (4/sp.pi)*(400/2)

    funcs_sym.append(volt_ellipse.subs(values_over_mod))
    labels.append(r'$u_\mathrm{six-step}$')
    colors.append('darkgreen')
    markers.append(None)

    # ------------------------------------------------------------------
    # Meshgrid
    XX, YY = np.meshgrid(
        np.linspace(-ranges[0], ranges[0], 1000),
        np.linspace(-ranges[1], ranges[1], 1000)
    )

    # Convert symbolic → numeric
    numeric_funcs = []
    for f_sym, color in zip(funcs_sym, colors):
        if isinstance(f_sym, sp.Matrix):
            f_sym = f_sym[0, 0]

        f_num = sp.lambdify((id, iq), f_sym, modules='numpy')
        Z = f_num(XX, YY)

        if np.isscalar(Z):
            Z = np.full_like(XX, Z)

        numeric_funcs.append((Z, color))

    # Plot contours FIRST
    fig, ax = plt.subplots(figsize=figsize)

    for Z, color in numeric_funcs:
        ax.contour(XX, YY, Z, levels=[0], colors=color, linewidths=2)

   
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Draw full frame
    frame = Rectangle(
        (0, 0), 1, 1,
        transform=ax.transAxes,
        fill=False,
        linewidth=1.0,
        edgecolor='black',
        clip_on=False,     
        zorder=10          
    )
    ax.add_patch(frame)

    ax.set_aspect('equal', 'box')
    ax.grid(True)

    # Axis labels
    ax.set_xlabel(r"$i_\mathrm{q}$ in A")
    ax.set_ylabel(r"$i_\mathrm{d}$ in A")

    # Move x-label slightly below zero
    ax.xaxis.set_label_coords(0.5, 1.1)  

    # Move y-label slightly left of zero
    ax.yaxis.set_label_coords(1.1, 0.5)

    # Plot SS-points & trajectory
    extra_labels = []
    extra_colors = []
    extra_markers = []

    ax.scatter(
        i_dq[::100, 0], i_dq[::100, 1],
        marker='x', color='gray', s=30, zorder=10
    )
   # extra_labels.append('SS-points')
   # extra_colors.append('gray')
   # extra_markers.append('x')

    if not Ref:
        ax.scatter(i_dq[0, 0], i_dq[0, 1],
                   marker='x', color='black', s=30, zorder=11)
        extra_labels.append('start')
        extra_colors.append('black')
        extra_markers.append('x')

        ax.plot(i_dq[:, 0], i_dq[:, 1],
                color='blue', linewidth=2, zorder=9)
        extra_labels.append(r'$\boldsymbol{i}_\mathrm{dq,traj}$')
        extra_colors.append('blue')
        extra_markers.append(None)

        ax.scatter(i_dq[-1, 0], i_dq[-1, 1],
                   marker='x', color='red', s=30, zorder=12)
        extra_labels.append('finish')
        extra_colors.append('red')
        extra_markers.append('x')

    # Legend 
    legend_handles = []

    for i in range(len(funcs_sym)):
        legend_handles.append(
            Line2D([0], [0], color=colors[i], lw=2, label=labels[i])
        )

    for lbl, col, m in zip(extra_labels, extra_colors, extra_markers):
        legend_handles.append(
            Line2D([0], [0], color=col, marker=m,
                   linestyle='None' if m else '-',
                   markersize=10, label=lbl)
        )

    ax.legend(handles=legend_handles, loc='lower right', bbox_to_anchor=(1, 0),frameon=True, framealpha=1)

    #fig.savefig("04_Hackle_at_n_11000rpm_Tdes_170N.m.pdf")
    plt.show()

def check_analy_opt(motor_env, trq_ach, trq_des, sim_state, id_iq_ana, id_iq_sim, step_len, figsize=(5, 4)):
    values = {ld: motor_env.env_properties.static_params.l_d,
          lq: motor_env.env_properties.static_params.l_q,
          Rs: motor_env.env_properties.static_params.r_s,
          n_p: motor_env.env_properties.static_params.p,
          lambda_p: motor_env.env_properties.static_params.psi_p, 
          #u_m: env_properties.action_constraints.u_d,
          #u_m: (4/jnp.pi)*(400/2), # Fundamental component modulation (six$
          #u_m: 400/jnp.sqrt(3), # Linear modulation region (inner circle)
          u_m: 251, # Linear modulation region (inner circle)
          i_m: motor_env.env_properties.physical_constraints.i_d,
          omega_k: sim_state.omega_el[0], 
          m_ref: trq_ach[0]
                    }
    torques = trq_ach[::step_len]               #np.ones_like(trq_ach[::step_len])*175
    torques_des = trq_des[::step_len]
    start_indx = 0
    stop_indx = len(trq_ach)
    trq_start_indx = int(start_indx/step_len)
    trq_stop_indx = int(stop_indx/step_len)

    plot_intersection(values, torques[trq_start_indx:trq_stop_indx], torques_des[trq_start_indx:trq_stop_indx], id_iq_ana[start_indx:stop_indx], figsize=figsize)
    plot_intersection(values, torques[trq_start_indx:trq_stop_indx], torques_des[trq_start_indx:trq_stop_indx], id_iq_sim[start_indx:stop_indx], Ref=False, figsize=figsize)