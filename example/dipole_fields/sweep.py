import csv
import subprocess
import tempfile
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

dipole_field_scale = 1.0e-7
dipole_minimum_distance = 0.001

# Initial COM positions inside one of these circles are skipped.
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


def make_input(x_0, y_0):
    inertia = 2.0 / 5.0 * mass * radius**2

    lines = [
        f"mass {mass:.17g}",
        f"radius {radius:.17g}",
        "charge 0.0",
        (
            "magneticMoment "
            f"{magnetic_moment[0]:.17g} "
            f"{magnetic_moment[1]:.17g} "
            f"{magnetic_moment[2]:.17g}"
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
        "velocity 0.0 0.0 0.0",
        "omega 0.0 0.0 0.0",
        "quaternion 1.0 0.0 0.0 0.0",
        "",
        "solverMode dAlembert",
        "",
        "constraint rolling",
        "normal 0.0 0.0 1.0",
        "",
        "gravityType uniform",
        f"gravity 0.0 0.0 {-g:.17g}",
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
        f"dipoleFieldScale {dipole_field_scale:.17g}",
        f"dipoleMinimumDistance {dipole_minimum_distance:.17g}",
        "",
        "airType none",
        "rollingResistanceType none",
        "",
        f"dt {dt:.17g}",
        f"tEnd {t_end:.17g}",
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


def run_simulation(point):
    index, x_0, y_0 = point

    if point_is_inside_magnet(x_0, y_0):
        return index, 0, np.full(7, np.nan)

    with tempfile.TemporaryDirectory(prefix="ribodyn_sweep_") as workdir:
        input_name = f"{workdir}/input.in"
        output_name = f"{workdir}/output.csv"

        with open(input_name, "w") as file:
            file.write(make_input(x_0, y_0))

        process = subprocess.run(
            [solver, input_name],
            cwd=workdir,
            text=True,
            capture_output=True,
        )

        if process.returncode != 0:
            print(
                f"\nSimulation failed at "
                f"x = {x_0:.6e}, y = {y_0:.6e}"
            )
            print(process.stderr)

            return index, -1, np.full(7, np.nan)

        final_state = np.array(
            read_final_state(output_name)
        )

    return index, 1, final_state


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
        (index, xx.flat[index], yy.flat[index])
        for index in range(xx.size)
    ]

    print(f"\nGrid points = {len(points)}")
    print(f"Cores       = {cores}")
    print(f"Plate       = {plate_size_x:.3f} m x {plate_size_y:.3f} m")
    print(f"t_end       = {t_end:.3f} s")

    with Pool(processes=cores) as pool:
        results = pool.map(
            run_simulation,
            points,
            chunksize=1,
        )

    status = np.zeros(xx.size, dtype=int)
    final_state = np.full((xx.size, 7), np.nan)

    for index, point_status, point_state in results:
        status[index] = point_status
        final_state[index] = point_state

    status = status.reshape(xx.shape)
    final_state = final_state.reshape(
        xx.shape + (7,)
    )

    np.savez(
        output_file,
        x=x,
        y=y,
        status=status,
        x_final=final_state[:, :, 0],
        y_final=final_state[:, :, 1],
        z_final=final_state[:, :, 2],
        vx_final=final_state[:, :, 3],
        vy_final=final_state[:, :, 4],
        vz_final=final_state[:, :, 5],
        constraint_residual=final_state[:, :, 6],
        plate_size_x=plate_size_x,
        plate_size_y=plate_size_y,
        sphere_radius=radius,
        magnet_exclusion_radius=magnet_exclusion_radius,
        dipole_positions=dipole_positions,
        dipole_moments=dipole_moments,
        dipole_field_scale=dipole_field_scale,
        t_end=t_end,
    )

    successful = np.sum(status == 1)
    skipped = np.sum(status == 0)
    failed = np.sum(status == -1)

    print(f"\nSuccessful  = {successful}")
    print(f"Skipped     = {skipped}")
    print(f"Failed      = {failed}")
    print(f"Output      = {output_file}")


if __name__ == "__main__":
    main()
