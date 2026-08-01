#!/usr/bin/env python3
"""Run and plot the two friction examples."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent


def load(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        header = [item.strip() for item in next(reader)]
        values = np.asarray(
            [
                [float(item.strip()) for item in row]
                for row in reader
                if row
            ]
        )

    return {
        name: values[:, index]
        for index, name in enumerate(header)
    }


def run(solver: Path, input_name: str, result_name: str) -> Path:
    result_dir = ROOT / "results" / result_name

    if result_dir.exists():
        shutil.rmtree(result_dir)

    result_dir.mkdir(parents=True)

    process = subprocess.run(
        [
            str(solver),
            str((ROOT / input_name).resolve()),
        ],
        cwd=result_dir,
        text=True,
        capture_output=True,
    )

    if process.returncode != 0:
        raise RuntimeError(
            f"Solver failed\n"
            f"stdout:\n{process.stdout}\n"
            f"stderr:\n{process.stderr}"
        )

    output = result_dir / "output.csv"

    if not output.is_file():
        raise RuntimeError(f"No output produced: {output}")

    return output


def positive_linear_quadratic_solution(
    t: np.ndarray,
    initial: float,
    linear_rate: float,
    quadratic_rate: float,
) -> np.ndarray:
    """Solve ydot = -linear_rate*y - quadratic_rate*y^2, y(0)>0."""
    if linear_rate == 0.0:
        return initial / (
            1.0 + quadratic_rate * initial * t
        )

    exponential = np.exp(-linear_rate * t)

    return (
        linear_rate
        * initial
        * exponential
        / (
            linear_rate
            + quadratic_rate
            * initial
            * (1.0 - exponential)
        )
    )


def plot_air(data: dict[str, np.ndarray], output: Path) -> None:
    t = data["t"]

    mass = 1.0
    radius = 0.1
    inertia = 0.004
    rho = 1.225
    eta = 1.81e-5
    cd = 0.47
    crot = 0.20
    area = np.pi * radius**2

    translational_linear_rate = (
        6.0 * np.pi * eta * radius / mass
    )
    translational_quadratic_rate = (
        0.5 * rho * cd * area / mass
    )

    rotational_linear_rate = (
        8.0 * np.pi * eta * radius**3 / inertia
    )
    rotational_quadratic_rate = (
        0.5
        * rho
        * crot
        * area
        * radius**3
        / inertia
    )

    vx_exact = positive_linear_quadratic_solution(
        t,
        8.0,
        translational_linear_rate,
        translational_quadratic_rate,
    )

    omega_exact = positive_linear_quadratic_solution(
        t,
        40.0,
        rotational_linear_rate,
        rotational_quadratic_rate,
    )

    fig, axs = plt.subplots(
        1,
        2,
        figsize=(9, 3.7),
        constrained_layout=True,
    )

    axs[0].plot(t, data["vx"], label="C++")
    axs[0].plot(t, vx_exact, "--", label="analytical")
    axs[0].set_title("Translational air drag")
    axs[0].set_xlabel("t [s]")
    axs[0].set_ylabel("v_x [m/s]")

    axs[1].plot(t, data["Oz"], label="C++")
    axs[1].plot(t, omega_exact, "--", label="analytical")
    axs[1].set_title("Rotational air drag")
    axs[1].set_xlabel("t [s]")
    axs[1].set_ylabel("Omega_z [1/s]")

    for ax in axs:
        ax.grid(True)
        ax.legend()

    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_coulomb(
    data: dict[str, np.ndarray],
    output: Path,
) -> None:
    t = data["t"]

    initial_speed = 2.0
    coefficient = 0.03
    gravity = 9.81

    # A force at the COM is coupled to the rotation by the exact rolling
    # constraint. For a solid sphere I/(m R^2)=2/5:
    #
    #     a = F / (m + I/R^2) = -(5/7) mu g.
    deceleration = (5.0 / 7.0) * coefficient * gravity

    speed_exact = np.maximum(
        initial_speed - deceleration * t,
        0.0,
    )

    fig, axs = plt.subplots(
        1,
        2,
        figsize=(9, 3.7),
        constrained_layout=True,
    )

    axs[0].plot(t, data["vx"], label="C++ regularized")
    axs[0].plot(
        t,
        speed_exact,
        "--",
        label="ideal Coulomb limit",
    )
    axs[0].set_title("Rolling speed")
    axs[0].set_xlabel("t [s]")
    axs[0].set_ylabel("v_x [m/s]")

    axs[1].plot(t, data["E_total"], label="mechanical energy")
    axs[1].set_title("Dissipative energy decay")
    axs[1].set_xlabel("t [s]")
    axs[1].set_ylabel("energy")

    for ax in axs:
        ax.grid(True)
        ax.legend()

    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("solver", type=Path)
    args = parser.parse_args()

    solver = args.solver.expanduser().resolve()

    if not solver.is_file():
        raise SystemExit(f"Solver not found: {solver}")

    plots = ROOT / "plots"
    plots.mkdir(exist_ok=True)

    air = load(
        run(
            solver,
            "air_resistance.in",
            "air_resistance",
        )
    )
    coulomb = load(
        run(
            solver,
            "coulomb_rolling_resistance.in",
            "coulomb_rolling_resistance",
        )
    )

    plot_air(air, plots / "air_resistance.png")
    plot_coulomb(
        coulomb,
        plots / "coulomb_rolling_resistance.png",
    )

    print(f"Plots written to {plots.resolve()}")


if __name__ == "__main__":
    main()
