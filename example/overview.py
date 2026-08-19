import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import simutils as su


# =============================================================================
# Input
# =============================================================================

input_file = Path("output.csv")
#output_file = None
output_file = "overview.pdf"

plots = [
    "trajectory",
    "position",
    "omega",
    "energy"
]

# Available:
# "trajectory"
# "position"
# "velocity"
# "omega"
# "quaternion"
# "kinetic"
# "potential"
# "energy"
# "energy_drift"
# "E_world"
# "E_body"
# "B_world"
# "B_body"
# "mu"
# "numerics"

ncols = 2


# =============================================================================
# Plot style
# =============================================================================

plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{siunitx} \usepackage{bm}",
    "font.size": 18,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
})


# =============================================================================
# Data
# =============================================================================

datasets = su.extract(input_file)
data = datasets[input_file.stem]

t = np.asarray(data.t)

energy = np.asarray(data.E_total)
energy_scale = max(abs(float(energy[0])), 1.0e-30)
energy_drift = (energy - energy[0]) / energy_scale

constraint_residual = np.maximum(
    np.abs(np.asarray(data.constraint_residual)),
    1.0e-18,
)

quaternion_error = np.maximum(
    np.abs(np.asarray(data.quaternion_norm) - 1.0),
    1.0e-18,
)


# =============================================================================
# Helpers
# =============================================================================

def components(ax, names, labels, ylabel=None):
    for name, label in zip(names, labels):
        ax.plot(
            t,
            np.asarray(getattr(data, name)),
            label=label,
        )

    ax.set_xlabel(r"$t\, [\unit{s}]$")

    if ylabel is not None:
        ax.set_ylabel(ylabel)


# =============================================================================
# Plots
# =============================================================================

nrows = int(np.ceil(len(plots) / ncols))

fig, axs = plt.subplots(
    nrows,
    ncols,
    figsize=(7.0 * ncols, 5.0 * nrows),
    layout="constrained",
)

axs = np.atleast_1d(axs).ravel()

for ax, plot in zip(axs, plots):

    if plot == "trajectory":
        ax.plot(data.x, data.y)
        ax.set_xlabel(r"$x\, [\unit{m}]$")
        ax.set_ylabel(r"$y\, [\unit{m}]$")
        ax.set_title(r"$\textrm{trajectory}$")
        ax.axis("equal")

    elif plot == "position":
        components(
            ax,
            ("x", "y", "z"),
            (r"$x$", r"$y$", r"$z$"),
            r"$[\unit{m}]$",
        )
        ax.set_title(r"$\textrm{position}$")

    elif plot == "velocity":
        components(
            ax,
            ("vx", "vy", "vz"),
            (r"$v_x$", r"$v_y$", r"$v_z$"),
            r"$[\unit{m.s^{-1}}]$",
        )
        ax.set_title(r"$\textrm{velocity}$")

    elif plot == "omega":
        components(
            ax,
            ("Ox", "Oy", "Oz"),
            (r"$\Omega_x$", r"$\Omega_y$", r"$\Omega_z$"),
            r"$[\unit{s^{-1}}]$",
        )
        ax.set_title(r"$\textrm{angular velocity}$")

    elif plot == "quaternion":
        components(
            ax,
            ("qw", "qx", "qy", "qz"),
            (r"$q_w$", r"$q_x$", r"$q_y$", r"$q_z$"),
        )
        ax.set_title(r"$\textrm{quaternion}$")

    elif plot == "kinetic":
        components(
            ax,
            ("T_trans", "T_rot"),
            (r"$T_{\mathrm{trans}}$", r"$T_{\mathrm{rot}}$"),
            r"$[\unit{J}]$",
        )
        ax.set_title(r"$\textrm{kinetic energy}$")

    elif plot == "potential":
        components(
            ax,
            ("U_generic", "U_gr", "U_em"),
            (r"$U_{\mathrm{gen}}$", r"$U_{\mathrm{gr}}$", r"$U_{\mathrm{em}}$"),
            r"$[\unit{J}]$",
        )
        ax.set_title(r"$\textrm{potential energy}$")

    elif plot == "energy":
        ax.plot(t, energy, label=r"$E$")
        ax.set_xlabel(r"$t\, [\unit{s}]$")
        ax.set_ylabel(r"$E\, [\unit{J}]$")
        ax.set_title(r"$\textrm{total energy}$")

    elif plot == "energy_drift":
        ax.plot(
            t,
            energy_drift,
            label=r"$(E-E_0)/|E_0|$",
        )
        ax.set_xlabel(r"$t\, [\unit{s}]$")
        ax.set_title(r"$\textrm{relative energy drift}$")

    elif plot == "E_world":
        components(
            ax,
            ("Ex_world", "Ey_world", "Ez_world"),
            (r"$E_x$", r"$E_y$", r"$E_z$"),
            r"$[\unit{V.m^{-1}}]$",
        )
        ax.set_title(r"$\bm{E}\ \textrm{world}$")

    elif plot == "E_body":
        components(
            ax,
            ("Ex_body", "Ey_body", "Ez_body"),
            (r"$E_x^b$", r"$E_y^b$", r"$E_z^b$"),
            r"$[\unit{V.m^{-1}}]$",
        )
        ax.set_title(r"$\bm{E}\ \textrm{body}$")

    elif plot == "B_world":
        components(
            ax,
            ("Bx_world", "By_world", "Bz_world"),
            (r"$B_x$", r"$B_y$", r"$B_z$"),
            r"$[\unit{T}]$",
        )
        ax.set_title(r"$\bm{B}\ \textrm{world}$")

    elif plot == "B_body":
        components(
            ax,
            ("Bx_body", "By_body", "Bz_body"),
            (r"$B_x^b$", r"$B_y^b$", r"$B_z^b$"),
            r"$[\unit{T}]$",
        )
        ax.set_title(r"$\bm{B}\ \textrm{body}$")

    elif plot == "mu":
        components(
            ax,
            (
                "mu_world_x",
                "mu_world_y",
                "mu_world_z",
                "mu_body_x",
                "mu_body_y",
                "mu_body_z",
            ),
            (
                r"$\mu_x$",
                r"$\mu_y$",
                r"$\mu_z$",
                r"$\mu_x^b$",
                r"$\mu_y^b$",
                r"$\mu_z^b$",
            ),
            r"$[\unit{A.m^2}]$",
        )
        ax.set_title(r"$\bm{\mu}$")

    elif plot == "numerics":
        ax.semilogy(
            t,
            constraint_residual,
            label=r"$\|A\nu-b\|$",
        )
        ax.semilogy(
            t,
            quaternion_error,
            label=r"$|\|q\|-1|$",
        )
        ax.set_xlabel(r"$t\, [\unit{s}]$")
        ax.set_title(r"$\textrm{numerics}$")

    else:
        raise ValueError(
            f"Unknown plot: {plot}"
        )

    ax.grid(True)

    handles, labels = ax.get_legend_handles_labels()

    if handles:
        ax.legend()


for ax in axs[len(plots):]:
    ax.remove()


# =============================================================================
# Output
# =============================================================================

if output_file is None:
    plt.show()

else:
    plt.savefig(
        output_file,
        dpi=300,
    )
