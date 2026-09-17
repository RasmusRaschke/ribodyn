import numpy as np 
import matplotlib.pyplot as plt 
from pathlib import Path
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{siunitx} \usepackage{bm}",
    "font.size": 24,
    "axes.titlesize": 24,
    "axes.labelsize": 24,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 24,
})

##############################################################################
input_file = Path(
    "trajectories"
    "/combination_0000"
    "/point_000000"
    "/output.csv"
)
transient_fraction = 0.25
# 2D phase-space projection.
phase_2d_x = "x"
phase_2d_y = "vx"

# 3D phase-space projection.
phase_3d_x = "x"
phase_3d_y = "y"
phase_3d_z = "Oz"

# Poincare section:
# section_variable crosses section_value in section_direction.
# section_direction = +1, -1 or 0 for both.
section_variable = "y"
section_value = 0.0
section_direction = +1

poincare_x = "x"
poincare_y = "vx"

output_2d = "attractor_2d.pdf"
output_3d = "attractor_3d.pdf"
output_poincare = "poincare_section.pdf"
##############################################################################


labels = {
    "t": r"$t \, [\unit{s}]$",
    "x": r"$x \, [\unit{m}]$",
    "y": r"$y \, [\unit{m}]$",
    "z": r"$z \, [\unit{m}]$",
    "vx": r"$v_x \, [\unit{m\,s^{-1}}]$",
    "vy": r"$v_y \, [\unit{m\,s^{-1}}]$",
    "vz": r"$v_z \, [\unit{m\,s^{-1}}]$",
    "Ox": r"$\Omega_x \, [\unit{s^{-1}}]$",
    "Oy": r"$\Omega_y \, [\unit{s^{-1}}]$",
    "Oz": r"$\Omega_z \, [\unit{s^{-1}}]$",
    "qw": r"$q_w$",
    "qx": r"$q_x$",
    "qy": r"$q_y$",
    "qz": r"$q_z$",
}


def load_data(filename):
    data = np.genfromtxt(
        filename,
        delimiter=",",
        names=True,
        dtype=float,
        encoding=None,
    )

    return data


def discard_transient(data):
    start = int(
        transient_fraction
        * len(data)
    )

    return data[
        start:
    ]


def section_crossings(data):
    s = (
        data[section_variable]
        - section_value
    )

    crossings = []

    for i in range(
        len(data) - 1
    ):
        crossed_up = (
            s[i] <= 0.0
            and s[i + 1] > 0.0
        )

        crossed_down = (
            s[i] >= 0.0
            and s[i + 1] < 0.0
        )

        if section_direction > 0:
            crossed = crossed_up
        elif section_direction < 0:
            crossed = crossed_down
        else:
            crossed = (
                crossed_up
                or crossed_down
            )

        if not crossed:
            continue

        denominator = (
            s[i + 1]
            - s[i]
        )

        if denominator == 0.0:
            alpha = 0.0
        else:
            alpha = (
                -s[i]
                / denominator
            )

        x_value = (
            data[poincare_x][i]
            + alpha
            * (
                data[poincare_x][i + 1]
                - data[poincare_x][i]
            )
        )

        y_value = (
            data[poincare_y][i]
            + alpha
            * (
                data[poincare_y][i + 1]
                - data[poincare_y][i]
            )
        )

        crossings.append([
            x_value,
            y_value,
        ])

    return np.asarray(
        crossings
    )


def plot_2d(data):
    fig, ax = plt.subplots(
        figsize=(9, 8),
        layout="constrained",
    )

    ax.plot(
        data[phase_2d_x],
        data[phase_2d_y],
        linewidth=0.8,
    )

    ax.set_xlabel(
        labels.get(
            phase_2d_x,
            phase_2d_x,
        )
    )

    ax.set_ylabel(
        labels.get(
            phase_2d_y,
            phase_2d_y,
        )
    )

    ax.tick_params(
        direction="in",
        top=True,
        right=True,
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    plt.savefig(
        output_2d,
        dpi=300,
    )


def plot_3d(data):
    fig = plt.figure(
        figsize=(10, 9),
        layout="constrained",
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    ax.plot(
        data[phase_3d_x],
        data[phase_3d_y],
        data[phase_3d_z],
        linewidth=0.7,
    )

    ax.set_xlabel(
        labels.get(
            phase_3d_x,
            phase_3d_x,
        )
    )

    ax.set_ylabel(
        labels.get(
            phase_3d_y,
            phase_3d_y,
        )
    )

    ax.set_zlabel(
        labels.get(
            phase_3d_z,
            phase_3d_z,
        )
    )

    plt.savefig(
        output_3d,
        dpi=300,
    )


def plot_poincare(data):
    points = section_crossings(
        data
    )

    print(
        f"Poincare points = "
        f"{len(points)}"
    )

    if len(points) == 0:
        return

    fig, ax = plt.subplots(
        figsize=(9, 8),
        layout="constrained",
    )

    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=12,
    )

    ax.set_xlabel(
        labels.get(
            poincare_x,
            poincare_x,
        )
    )

    ax.set_ylabel(
        labels.get(
            poincare_y,
            poincare_y,
        )
    )

    ax.tick_params(
        direction="in",
        top=True,
        right=True,
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    plt.savefig(
        output_poincare,
        dpi=300,
    )


def main():
    data = load_data(
        input_file
    )

    data = discard_transient(
        data
    )

    print(
        f"Points after transient = "
        f"{len(data)}"
    )

    plot_2d(
        data
    )

    plot_3d(
        data
    )

    plot_poincare(
        data
    )


if __name__ == "__main__":
    main()
