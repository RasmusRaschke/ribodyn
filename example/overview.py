#!/usr/bin/env python3
"""Overview plots for ribodyn CSV output.
Made with help of generative AI.
Usage
-----
python3 overview_full.py output.csv
python3 overview_full.py output.csv --save overview.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import extractor as su


def components(ax, t, data, names, labels, title, ylabel=None):
    for name, label in zip(names, labels):
        ax.plot(t, np.asarray(getattr(data, name)), label=label)
    ax.set_title(title)
    ax.set_xlabel(r"$t\,[\mathrm{s}]$")
    if ylabel:
        ax.set_ylabel(ylabel)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ribodyn simulation diagnostics.")
    parser.add_argument("csv", nargs="?", default="output.csv", help="CSV output file")
    parser.add_argument("--save", type=Path, help="Save the figure instead of only displaying it")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    datasets = su.extract(csv_path)
    try:
        data = datasets[csv_path.stem]
    except KeyError as exc:
        raise RuntimeError(f"Could not load dataset '{csv_path.stem}' from {csv_path}") from exc

    t = np.asarray(data.t)
    energy = np.asarray(data.E_total)
    energy_scale = max(abs(float(energy[0])), 1.0e-30)
    relative_energy_drift = (energy - energy[0]) / energy_scale

    constraint_residual = np.maximum(
        np.abs(np.asarray(data.constraint_residual)), 1.0e-18
    )
    quaternion_error = np.maximum(
        np.abs(np.asarray(data.quaternion_norm) - 1.0), 1.0e-18
    )

    plt.style.use("seaborn-v0_8-paper")
    fig, axs = plt.subplots(5, 3, figsize=(16, 18), constrained_layout=True)

    components(
        axs[0, 0], t, data,
        ("x", "y", "z"),
        (r"$x$", r"$y$", r"$z$"),
        "COM position",
        r"$[\mathrm{m}]$",
    )

    axs[0, 1].plot(data.x, data.y)
    axs[0, 1].set_title("Planar Projection")
    axs[0, 1].set_xlabel(r"$x\,[\mathrm{m}]$")
    axs[0, 1].set_ylabel(r"$y\,[\mathrm{m}]$")
    axs[0, 1].axis("equal")

    components(
        axs[0, 2], t, data,
        ("vx", "vy", "vz"),
        (r"$v_x$", r"$v_y$", r"$v_z$"),
        "COM velocity",
        r"$[\mathrm{m\,s^{-1}}]$",
    )

    components(
        axs[1, 0], t, data,
        ("Ox", "Oy", "Oz"),
        (r"$\Omega_x$", r"$\Omega_y$", r"$\Omega_z$"),
        "Body angular velocity",
        r"$[\mathrm{s^{-1}}]$",
    )

    components(
        axs[1, 1], t, data,
        ("qw", "qx", "qy", "qz"),
        (r"$q_w$", r"$q_x$", r"$q_y$", r"$q_z$"),
        "Orientation quaternion",
    )

    components(
        axs[1, 2], t, data,
        ("T_trans", "T_rot"),
        (r"$T_{\mathrm{trans}}$", r"$T_{\mathrm{rot}}$"),
        "Kinetic energies",
        r"$[\mathrm{J}]$",
    )

    components(
        axs[2, 0], t, data,
        ("U_generic", "U_gr", "U_em"),
        (r"$U_{\mathrm{gen}}$", r"$U_{\mathrm{gr}}$", r"$U_{\mathrm{em}}$"),
        "Potential-energy contributions",
        r"$[\mathrm{J}]$",
    )

    axs[2, 1].plot(t, energy, label=r"$E_{\mathrm{tot}}$")
    axs[2, 1].set_title("Total mechanical energy")
    axs[2, 1].set_xlabel(r"$t\,[\mathrm{s}]$")
    axs[2, 1].set_ylabel(r"$[\mathrm{J}]$")

    axs[2, 2].plot(t, relative_energy_drift, label=r"$\frac{E-E_0}{|E_0|}$")
    axs[2, 2].set_title("Relative energy drift")
    axs[2, 2].set_xlabel(r"$t\,[\mathrm{s}]$")

    components(
        axs[3, 0], t, data,
        ("Ex_world", "Ey_world", "Ez_world"),
        (r"$E_x$", r"$E_y$", r"$E_z$"),
        "Electric field — inertial frame",
        r"$[\mathrm{V\,m^{-1}}]$",
    )

    components(
        axs[3, 1], t, data,
        ("Ex_body", "Ey_body", "Ez_body"),
        (r"$E_x^b$", r"$E_y^b$", r"$E_z^b$"),
        "Electric field — body frame",
        r"$[\mathrm{V\,m^{-1}}]$",
    )

    components(
        axs[3, 2], t, data,
        ("Bx_world", "By_world", "Bz_world"),
        (r"$B_x$", r"$B_y$", r"$B_z$"),
        "Magnetic field — inertial frame",
        r"$[\mathrm{T}]$",
    )

    components(
        axs[4, 0], t, data,
        ("Bx_body", "By_body", "Bz_body"),
        (r"$B_x^b$", r"$B_y^b$", r"$B_z^b$"),
        "Magnetic field — body frame",
        r"$[\mathrm{T}]$",
    )

    components(
        axs[4, 1], t, data,
        (
            "mu_world_x", "mu_world_y", "mu_world_z",
            "mu_body_x", "mu_body_y", "mu_body_z",
        ),
        (
            r"$\mu_x$", r"$\mu_y$", r"$\mu_z$",
            r"$\mu_x^b$", r"$\mu_y^b$", r"$\mu_z^b$",
        ),
        "Magnetic moment",
        r"$[\mathrm{A\,m^2}]$",
    )

    axs[4, 2].semilogy(
        t, constraint_residual, label=r"$\|A\nu-b\|$"
    )
    axs[4, 2].semilogy(
        t, quaternion_error, label=r"$|\|q\|-1|$"
    )
    axs[4, 2].set_title("Numerical constraints")
    axs[4, 2].set_xlabel(r"$t\,[\mathrm{s}]$")

    for ax in axs.flat:
        ax.grid(True)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize="small", ncols=2)

    fig.suptitle(csv_path.name)

    if args.save:
        fig.savefig(args.save, dpi=200)
        print(f"Saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
