import csv
import subprocess
import tempfile
from itertools import product
from multiprocessing import Pool
import numpy as np 
from pathlib import Path

base_directory = Path(__file__).resolve().parent
solver = (
    base_directory
    / "../../build/solver"
).resolve()
cores = 40
n_x = 200
n_y = 200
output_file = "magDipole_sweep.npz"
plate_size_x = 0.40
plate_size_y = 0.40
mass = 0.02
radius = 0.01
magnetic_moment = np.array([0.0, 0.0, 1.00])
g = 9.80665
dt = 1e-4
t_end = 3.0

initial_velocity = np.array([0.0, 0.0, 0.0])
initial_omega = np.array([0.0, 0.0, 0.0])

rolling_resistance_type = "none"
rolling_friction_coefficient = 0.0
rolling_friction_smoothing_speed = 0.02
surface_velocity = np.array([0.0, 0.0, 0.0])

# Every entry is varied independently. The complete sweep is the Cartesian
# product of all arrays in this dictionary.
#
# Vector inputs can be varied component-wise with _x, _y and _z.
#
# Examples:
#
# sweep_inputs = {
#     "omega_z": np.array([-20.0, 0.0, 20.0]),
# }
#
# sweep_inputs = {
#     "omega_x": np.array([-5.0, 0.0, 5.0]),
#     "omega_y": np.array([-5.0, 0.0, 5.0]),
#     "omega_z": np.array([-20.0, 0.0, 20.0]),
#     "rollingFrictionCoefficient": np.array([0.0, 0.005, 0.010]),
#     "rollingFrictionSmoothingSpeed": np.array([0.01, 0.02]),
# }
#
sweep_inputs = {
    "omega_z": np.array([
        0.0,
    ]),
}

progress_every = 100
keep_trajectories = False
trajectory_directory = (
    base_directory
    / "trajectories"
)

normal = np.array([0.0, 0.0, 1.0])
dipole_field_scale = 1.0e-7
dipole_minimum_distance = 0.001
magnet_exclusion_radius = 0.005
dipole_positions = np.array([
    [0.1050, 0.0000, -0.0050],
    [-0.0525, 0.0909326674, -0.0050],
    [-0.0525, -0.0909326674, -0.0050],
])
dipole_moments = np.array([
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0],
])


def make_sweep_combinations():
    sweep_names = list(sweep_inputs.keys())
    sweep_arrays = [np.asarray(sweep_inputs[name], dtype=float,) for name in sweep_names]
    for name, values in zip(sweep_names, sweep_arrays,):
        if values.ndim != 1:
            raise ValueError(
                f"Sweep input {name} must be one-dimensional."
            )
        if len(values) == 0:
            raise ValueError(
                f"Sweep input {name} is empty."
            )
    if len(sweep_names) == 0:
        sweep_combinations = np.empty(
            (
                1,
                0,
            )
        )
    else:
        sweep_combinations = np.array(list(product(*sweep_arrays)), dtype=float,)
    return (sweep_names, sweep_arrays, sweep_combinations,)


sweep_names, sweep_arrays, sweep_combinations = (
    make_sweep_combinations()
)


def apply_sweep_values(parameter_values):
    values = {
        "charge": 0.0,
        "magneticMoment": magnetic_moment.copy(),
        "velocity": initial_velocity.copy(),
        "omega": initial_omega.copy(),
        "normal": normal.copy(),
        "gravity": np.array([
            0.0,
            0.0,
            -g,
        ]),
        "surfaceVelocity": surface_velocity.copy(),
        "rollingFrictionCoefficient": rolling_friction_coefficient,
        "rollingFrictionSmoothingSpeed": rolling_friction_smoothing_speed,
        "dipoleFieldScale": dipole_field_scale,
        "dipoleMinimumDistance": dipole_minimum_distance,
        "dt": dt,
        "tEnd": t_end,
    }

    for name, value in zip(sweep_names, parameter_values,):
        if (name.endswith("_x") or name.endswith("_y") or name.endswith("_z")):
            input_name = name[:-2]
            if input_name not in values:
                raise ValueError(
                    f"Unknown vector sweep input: {name}"
                )
            if not isinstance(
                values[input_name],
                np.ndarray,
            ):
                raise ValueError(
                    f"Sweep input {name} is not a vector input."
                )
            component = {"x": 0, "y": 1, "z": 2,}[name[-1]]
            values[input_name][component] = value
        else:
            if name not in values:
                raise ValueError(
                    f"Unknown scalar sweep input: {name}"
                )
            if isinstance(
                values[name],
                np.ndarray,
            ):
                raise ValueError(
                    f"Vector sweep input {name} requires _x, _y or _z."
                )

            values[name] = value

    return values


def format_sweep_values(parameter_values):
    if len(sweep_names) == 0:
        return "baseline"

    return ", ".join(
        (
            f"{name} = {value:.6e}"
        )
        for name, value in zip(
            sweep_names,
            parameter_values,
        )
    )


def make_input(x_0, y_0, parameter_values):
    inertia = 2.0 / 5.0 * mass * radius**2
    values = apply_sweep_values(parameter_values)
    current_rolling_resistance_type = (rolling_resistance_type)
    if (
        "rollingFrictionCoefficient"
        in sweep_names
        or "rollingFrictionSmoothingSpeed"
        in sweep_names
    ):
        current_rolling_resistance_type = "coulomb"

    lines = [
        f"mass {mass:.17g}",
        f"radius {radius:.17g}",
        f"charge {values['charge']:.17g}",
        (
            "magneticMoment "
            f"{values['magneticMoment'][0]:.17g} "
            f"{values['magneticMoment'][1]:.17g} "
            f"{values['magneticMoment'][2]:.17g}"
        ),
        "magneticPolarizability 0 0 0  0 0 0  0 0 0",
        (
            "inertia "
            f"{inertia:.17g} 0 0  "
            f"0 {inertia:.17g} 0  "
            f"0 0 {inertia:.17g}"
        ),
        "",
        f"position {x_0:.17g} {y_0:.17g} {radius:.17g}",
        (
            "velocity "
            f"{values['velocity'][0]:.17g} "
            f"{values['velocity'][1]:.17g} "
            f"{values['velocity'][2]:.17g}"
        ),
        (
            "omega "
            f"{values['omega'][0]:.17g} "
            f"{values['omega'][1]:.17g} "
            f"{values['omega'][2]:.17g}"
        ),
        "quaternion 1.0 0.0 0.0 0.0",
        "",
        "solverMode dAlembert",
        "",
        "constraint rolling",
        (
            "normal "
            f"{values['normal'][0]:.17g} "
            f"{values['normal'][1]:.17g} "
            f"{values['normal'][2]:.17g}"
        ),
        "",
        "gravityType uniform",
        (
            "gravity "
            f"{values['gravity'][0]:.17g} "
            f"{values['gravity'][1]:.17g} "
            f"{values['gravity'][2]:.17g}"
        ),
        "",
        "emType magDipole",
        f"dipoleCount {len(dipole_positions)}",
    ]

    for index, (position, moment) in enumerate(
        zip(dipole_positions, dipole_moments),
        start=1,
    ):
        lines.append(
            f"dipolePosition{index} "
            f"{position[0]:.17g} "
            f"{position[1]:.17g} "
            f"{position[2]:.17g}"
        )
        lines.append(
            f"dipoleMoment{index} "
            f"{moment[0]:.17g} "
            f"{moment[1]:.17g} "
            f"{moment[2]:.17g}"
        )

    lines.extend([
        (
            "dipoleFieldScale "
            f"{values['dipoleFieldScale']:.17g}"
        ),
        (
            "dipoleMinimumDistance "
            f"{values['dipoleMinimumDistance']:.17g}"
        ),
        "",
        "airType none",
        (
            "rollingResistanceType "
            f"{current_rolling_resistance_type}"
        ),
    ])

    if current_rolling_resistance_type == "coulomb":
        lines.extend([
            (
                "rollingFrictionCoefficient "
                f"{values['rollingFrictionCoefficient']:.17g}"
            ),
            (
                "rollingFrictionSmoothingSpeed "
                f"{values['rollingFrictionSmoothingSpeed']:.17g}"
            ),
            (
                "surfaceVelocity "
                f"{values['surfaceVelocity'][0]:.17g} "
                f"{values['surfaceVelocity'][1]:.17g} "
                f"{values['surfaceVelocity'][2]:.17g}"
            ),
        ])

    lines.extend([
        "",
        f"dt {values['dt']:.17g}",
        f"tEnd {values['tEnd']:.17g}",
        "",
    ])

    return "\n".join(lines)


def read_final_state(filename):
    with open(filename, "r", newline="") as file:
        reader = csv.DictReader(file)
        final_row = None

        for row in reader:
            final_row = row

    if final_row is None:
        raise RuntimeError("The solver produced no output rows.")

    return (
        float(final_row["x"]),
        float(final_row["y"]),
        float(final_row["z"]),
        float(final_row["vx"]),
        float(final_row["vy"]),
        float(final_row["vz"]),
        float(final_row["constraint_residual"]),
    )


def point_is_inside_magnet(x_0, y_0):
    distance = np.sqrt(
        (x_0 - dipole_positions[:, 0])**2
        + (y_0 - dipole_positions[:, 1])**2
    )

    return np.any(distance < magnet_exclusion_radius)


def run_in_directory(
    workdir,
    combination_index,
    index,
    x_0,
    y_0,
    parameter_values,
):
    workdir = Path(workdir)
    workdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_name = (
        workdir
        / "input.in"
    )

    output_name = (
        workdir
        / "output.csv"
    )

    if output_name.exists():
        output_name.unlink()

    with open(input_name, "w") as file:
        file.write(
            make_input(
                x_0,
                y_0,
                parameter_values,
            )
        )

    process = subprocess.run(
        [
            solver,
            input_name,
        ],
        cwd=workdir,
        text=True,
        capture_output=True,
    )

    if process.returncode != 0:
        print(
            f"\nSimulation failed at "
            f"x = {x_0:.6e}, y = {y_0:.6e}, "
            f"{format_sweep_values(parameter_values)}"
        )
        print(process.stderr)

        if keep_trajectories:
            with open(
                workdir / "error.log",
                "w",
            ) as file:
                file.write(process.stderr)

        return (
            combination_index,
            index,
            -1,
            np.full(7, np.nan),
        )

    final_state = np.array(
        read_final_state(output_name)
    )

    return (
        combination_index,
        index,
        1,
        final_state,
    )


def run_simulation(point):
    (
        combination_index,
        index,
        x_0,
        y_0,
        parameter_values,
    ) = point

    if point_is_inside_magnet(x_0, y_0):
        return (
            combination_index,
            index,
            0,
            np.full(7, np.nan),
        )

    if keep_trajectories:
        workdir = (
            trajectory_directory
            / f"combination_{combination_index:04d}"
            / f"point_{index:06d}"
        )

        return run_in_directory(
            workdir,
            combination_index,
            index,
            x_0,
            y_0,
            parameter_values,
        )

    with tempfile.TemporaryDirectory(
        prefix="ribodyn_sweep_"
    ) as workdir:
        return run_in_directory(
            workdir,
            combination_index,
            index,
            x_0,
            y_0,
            parameter_values,
        )


# =============================================================================
# Sweep
# =============================================================================

def main():
    x = np.linspace(
        -plate_size_x / 2.0 + radius,
        plate_size_x / 2.0 - radius,
        n_x,
    )

    y = np.linspace(
        -plate_size_y / 2.0 + radius,
        plate_size_y / 2.0 - radius,
        n_y,
    )

    xx, yy = np.meshgrid(x, y)

    points = [
        (
            combination_index,
            index,
            xx.flat[index],
            yy.flat[index],
            parameter_values,
        )
        for combination_index, parameter_values
        in enumerate(sweep_combinations)
        for index in range(xx.size)
    ]

    print(f"\nGrid points      = {xx.size}")
    print(f"Sweep inputs     = {len(sweep_names)}")
    print(f"Combinations     = {len(sweep_combinations)}")
    print(f"Calculations     = {len(points)}")
    print(f"Cores            = {cores}")
    print(f"Plate            = {plate_size_x:.3f} m x {plate_size_y:.3f} m")
    print(f"t_end            = {t_end:.3f} s")

    for name, values in zip(
        sweep_names,
        sweep_arrays,
    ):
        print(
            f"{name:<28} = {values}"
        )

    if keep_trajectories:
        for combination_index, parameter_values in enumerate(
            sweep_combinations
        ):
            combination_directory = (
                trajectory_directory
                / f"combination_{combination_index:04d}"
            )

            combination_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            with open(
                combination_directory
                / "parameters.txt",
                "w",
            ) as file:
                for name, value in zip(
                    sweep_names,
                    parameter_values,
                ):
                    file.write(
                        f"{name} {value:.17g}\n"
                    )

    status = np.zeros(
        (
            len(sweep_combinations),
            xx.size,
        ),
        dtype=int,
    )

    final_state = np.full(
        (
            len(sweep_combinations),
            xx.size,
            7,
        ),
        np.nan,
    )

    successful = 0
    skipped = 0
    failed = 0

    with Pool(processes=cores) as pool:
        results = pool.imap_unordered(
            run_simulation,
            points,
            chunksize=1,
        )

        for calculation, result in enumerate(
            results,
            start=1,
        ):
            (
                combination_index,
                index,
                point_status,
                point_state,
            ) = result

            status[
                combination_index,
                index,
            ] = point_status

            final_state[
                combination_index,
                index,
            ] = point_state

            if point_status == 1:
                successful += 1
            elif point_status == 0:
                skipped += 1
            else:
                failed += 1

            if (
                calculation % progress_every == 0
                or calculation == len(points)
            ):
                print(
                    f"Finished {calculation} / {len(points)} "
                    f"({100.0 * calculation / len(points):.1f} %), "
                    f"successful = {successful}, "
                    f"skipped = {skipped}, "
                    f"failed = {failed}",
                    flush=True,
                )

    status = status.reshape(
        (
            len(sweep_combinations),
            n_y,
            n_x,
        )
    )

    final_state = final_state.reshape(
        (
            len(sweep_combinations),
            n_y,
            n_x,
            7,
        )
    )

    sweep_output = {
        (
            "sweep_values_"
            + name
        ): values
        for name, values in zip(
            sweep_names,
            sweep_arrays,
        )
    }

    np.savez(
        output_file,
        x=x,
        y=y,
        sweep_names=np.array(
            sweep_names,
            dtype=str,
        ),
        sweep_combinations=sweep_combinations,
        status=status,
        x_final=final_state[:, :, :, 0],
        y_final=final_state[:, :, :, 1],
        z_final=final_state[:, :, :, 2],
        vx_final=final_state[:, :, :, 3],
        vy_final=final_state[:, :, :, 4],
        vz_final=final_state[:, :, :, 5],
        constraint_residual=final_state[:, :, :, 6],
        plate_size_x=plate_size_x,
        plate_size_y=plate_size_y,
        sphere_radius=radius,
        magnet_exclusion_radius=magnet_exclusion_radius,
        dipole_positions=dipole_positions,
        dipole_moments=dipole_moments,
        dipole_field_scale=dipole_field_scale,
        t_end=t_end,
        rolling_resistance_type=rolling_resistance_type,
        rolling_friction_coefficient=rolling_friction_coefficient,
        rolling_friction_smoothing_speed=rolling_friction_smoothing_speed,
        **sweep_output,
    )

    print(f"\nSuccessful  = {successful}")
    print(f"Skipped     = {skipped}")
    print(f"Failed      = {failed}")
    print(f"Output      = {output_file}")

    if keep_trajectories:
        print(
            f"Trajectories = "
            f"{trajectory_directory}"
        )


if __name__ == "__main__":
    main()
