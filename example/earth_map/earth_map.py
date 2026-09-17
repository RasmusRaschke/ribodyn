from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import shutil
import subprocess
import tempfile
import numpy as np
import pandas as pd
from wmm import wmm_calc
import warnings
warnings.filterwarnings("ignore")

###########################################################################
YEAR = 2026
MONTH = 8
DAY = 20
ALTITUDE_M = 0.0
DLAT = 20.0          # latitude spacing [deg]
DLON = 20.0          # longitude spacing [deg]
SOLVER = Path("../../build/solver").resolve()
INPUT_TEMPLATE = Path("input.in")
OUTPUT_DIR = Path("results")
OUTFILE = OUTPUT_DIR / "earth_results.npz"
MAX_CORES = 40
CITY_CASES = {
    "hamburg": (53.550556, 9.993682),
    "jakarta": (-6.200000, 106.826944),
    "tokyo": (35.689444, 139.691667),
    "australia": (-35.279722, 149.128998),
}
###########################################################################

def create_grid(dlat, dlon):
    lats = np.arange(-90.0, 90.0 + dlat, dlat)
    lons = np.arange(-180.0, 180.0 + dlon, dlon)
    Lon, Lat = np.meshgrid(lons, lats)
    return Lat, Lon


def create_wmm():
    model = wmm_calc()
    model.setup_time(YEAR, MONTH, DAY)
    return model


def field_at(model, lat, lon):
    model.setup_env(
        lat=lat,
        lon=lon,
        alt=ALTITUDE_M,
        unit="m",
        msl=True,
    )
    try:
        B = model.get_all()
        east = float(np.asarray(B["y"]).squeeze())
        north = float(np.asarray(B["x"]).squeeze())
        up = -float(np.asarray(B["z"]).squeeze())
    except Exception:
        east = float(np.asarray(model.get_By()).squeeze())
        north = float(np.asarray(model.get_Bx()).squeeze())
        up = -float(np.asarray(model.get_Bz()).squeeze())

    # Convert nT -> Tesla.
    east *= 1e-9
    north *= 1e-9
    up *= 1e-9

    # Solver world coordinates:
    # +x = west, +y = south, +z = up.
    bx = -east
    by = -north
    bz = up

    return bx, by, bz


def build_cases(model, Lat, Lon):
    cases = []
    nlat, nlon = Lat.shape
    for i in range(nlat):
        for j in range(nlon):
            lat = float(Lat[i, j])
            lon = float(Lon[i, j])
            bx, by, bz = field_at(model, lat, lon)
            cases.append(
                (
                    lat,
                    lon,
                    bx,
                    by,
                    bz,
                )
            )

    return cases


def read_template_value(key):
    with open(INPUT_TEMPLATE, "r") as file:
        for line in file:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if parts[0] == key:
                return parts[1:]
    raise RuntimeError(f"Missing '{key}' in {INPUT_TEMPLATE}")


def initial_rotation_matrix():
    q = np.asarray(
        read_template_value("quaternion"),
        dtype=float,
    )

    if q.size != 4:
        raise RuntimeError(
            "Initial quaternion must contain four components."
        )

    qnorm = np.linalg.norm(q)

    if qnorm < 1.0e-14:
        raise RuntimeError(
            "Initial quaternion cannot be zero."
        )

    qw, qx, qy, qz = q / qnorm

    # Body -> world rotation, matching Eigen::Quaterniond::toRotationMatrix().
    return np.array([
        [
            1.0 - 2.0 * (qy*qy + qz*qz),
            2.0 * (qx*qy - qw*qz),
            2.0 * (qx*qz + qw*qy),
        ],
        [
            2.0 * (qx*qy + qw*qz),
            1.0 - 2.0 * (qx*qx + qz*qz),
            2.0 * (qy*qz - qw*qx),
        ],
        [
            2.0 * (qx*qz - qw*qy),
            2.0 * (qy*qz + qw*qx),
            1.0 - 2.0 * (qx*qx + qy*qy),
        ],
    ])


def magnetic_moment_body(bx, by, bz):
    B_world = np.array(
        [bx, by, bz],
        dtype=float,
    )

    Bnorm = np.linalg.norm(B_world)

    if Bnorm < 1.0e-20:
        raise RuntimeError(
            "Cannot align magnetic moment with a zero magnetic field."
        )

    B_direction_world = B_world / Bnorm
    R0 = initial_rotation_matrix()

    # magneticMoment is stored in the body frame, while WMM B is in world.
    # Choose |mu|=1 and R0*mu_body = B_world/|B_world|.
    mu_body = R0.T @ B_direction_world

    return mu_body


def validate_template():
    if read_template_value("constraint")[0] != "pointContact":
        raise RuntimeError(
            "Earth sweep requires 'constraint pointContact' in input_flat_drypoint_mu_aligned.in."
        )

    if read_template_value("contactFrictionType")[0] != "dryPoint":
        raise RuntimeError(
            "Earth sweep requires 'contactFrictionType dryPoint' in input_flat_drypoint_mu_aligned.in."
        )

    normal = np.asarray(
        read_template_value("normal"),
        dtype=float,
    )

    if not np.allclose(
        normal,
        np.array([0.0, 0.0, 1.0]),
        atol=1.0e-12,
    ):
        raise RuntimeError(
            "Earth dry-point sweep expects a flat surface: "
            "'normal 0 0 1'."
        )

    read_template_value("contactPointBody")
    read_template_value("omega")

    R0 = initial_rotation_matrix()

    if not np.allclose(
        R0.T @ R0,
        np.eye(3),
        atol=1.0e-12,
    ):
        raise RuntimeError(
            "Initial quaternion did not produce an orthogonal rotation."
        )


def make_input(bx, by, bz):
    base = INPUT_TEMPLATE.read_text()

    mu_body = magnetic_moment_body(
        bx,
        by,
        bz,
    )

    # The flat surface, body tilt, contact geometry and dry friction remain
    # fixed.  For every Earth location:
    #
    #   B_world = local WMM field
    #   |mu| = 1
    #   mu_world(t=0) = B_world / |B_world|
    #
    # Since magneticMoment is a body-frame quantity:
    #
    #   mu_body = R(q0)^T B_world / |B_world|.
    overrides = f"""
# earth_grid.py overrides
magneticField {bx:.17g} {by:.17g} {bz:.17g}
magneticMoment {mu_body[0]:.17g} {mu_body[1]:.17g} {mu_body[2]:.17g}
"""

    return base.rstrip() + "\n\n" + overrides.lstrip()


def read_last_values(csv_file):
    df = pd.read_csv(csv_file)
    final = df.iloc[-1]

    state = np.array([
        float(final["x"]),
        float(final["y"]),
        float(final["z"]),
        float(final["qw"]),
        float(final["qx"]),
        float(final["qy"]),
        float(final["qz"]),
        float(final["vx"]),
        float(final["vy"]),
        float(final["vz"]),
        float(final["Ox"]),
        float(final["Oy"]),
        float(final["Oz"]),
        float(final["constraint_residual"]),
    ])

    t_last = float(final["t"])

    return state, t_last


def run_case(case):
    index, lat, lon, bx, by, bz = case

    with tempfile.TemporaryDirectory(prefix="ribodyn_earth_") as workdir:
        workdir = Path(workdir)
        input_file = workdir / "input.in"
        output_file = workdir / "output.csv"

        input_file.write_text(
            make_input(bx, by, bz)
        )

        process = subprocess.run(
            [
                str(SOLVER),
                str(input_file),
            ],
            cwd=workdir,
            text=True,
            capture_output=True,
        )

        if process.returncode != 0:
            return (
                index,
                -1,
                np.full(14, np.nan),
                np.nan,
                process.stderr,
            )

        final_state, t_last = read_last_values(output_file)

    return (
        index,
        1,
        final_state,
        t_last,
        "",
    )


def run_city_case(name, lat, lon, model):
    bx, by, bz = field_at(model, lat, lon)

    with tempfile.TemporaryDirectory(prefix=f"ribodyn_{name}_") as workdir:
        workdir = Path(workdir)
        input_file = workdir / "input.in"
        output_file = workdir / "output.csv"

        input_file.write_text(
            make_input(bx, by, bz)
        )

        process = subprocess.run(
            [
                str(SOLVER),
                str(input_file),
            ],
            cwd=workdir,
            text=True,
            capture_output=True,
        )

        if process.returncode != 0:
            raise RuntimeError(
                f"Simulation failed for {name}.\n\n"
                f"stdout:\n{process.stdout}\n\n"
                f"stderr:\n{process.stderr}"
            )

        # Keep the old combined.py workflow: four named CSV files.
        shutil.copy2(
            output_file,
            Path(f"{name}.csv"),
        )


def main():
    if not SOLVER.is_file():
        raise FileNotFoundError(
            f"Solver not found:\n{SOLVER}"
        )

    if not INPUT_TEMPLATE.is_file():
        raise FileNotFoundError(
            f"Input template not found:\n{INPUT_TEMPLATE}"
        )

    validate_template()

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Creating latitude-longitude grid...")
    Lat, Lon = create_grid(DLAT, DLON)

    print("Initialising WMM...")
    model = create_wmm()

    print("Computing magnetic field...")
    cases = build_cases(model, Lat, Lon)
    print(f"{len(cases)} grid points")

    indexed_cases = [
        (index, *case)
        for index, case in enumerate(cases)
    ]

    print("Running simulations...")
    with ThreadPoolExecutor(max_workers=MAX_CORES) as executor:
        results = list(
            executor.map(
                run_case,
                indexed_cases,
            )
        )

    nlat, nlon = Lat.shape

    status_flat = np.zeros(
        len(cases),
        dtype=int,
    )

    final_state_flat = np.full(
        (
            len(cases),
            14,
        ),
        np.nan,
    )

    t_flat = np.full(
        len(cases),
        np.nan,
    )

    failures = 0

    for index, point_status, final_state, t_last, error in results:
        status_flat[index] = point_status
        final_state_flat[index] = final_state
        t_flat[index] = t_last

        if error:
            failures += 1
            if failures <= 10:
                print(
                    f"Simulation failed at "
                    f"lat={cases[index][0]:.3f}, "
                    f"lon={cases[index][1]:.3f}"
                )
                print(error)

    if failures:
        print(f"{failures} grid simulations failed.")

    status = status_flat.reshape(
        nlat,
        nlon,
    )

    final_state = final_state_flat.reshape(
        nlat,
        nlon,
        14,
    )

    t_last = t_flat.reshape(
        nlat,
        nlon,
    )

    Bx = np.empty((nlat, nlon))
    By = np.empty((nlat, nlon))
    Bz = np.empty((nlat, nlon))

    mu_body_x = np.empty((nlat, nlon))
    mu_body_y = np.empty((nlat, nlon))
    mu_body_z = np.empty((nlat, nlon))

    k = 0

    for i in range(nlat):
        for j in range(nlon):
            bx, by, bz = cases[k][2:]

            Bx[i, j] = bx
            By[i, j] = by
            Bz[i, j] = bz

            mu_body = magnetic_moment_body(
                bx,
                by,
                bz,
            )

            mu_body_x[i, j] = mu_body[0]
            mu_body_y[i, j] = mu_body[1]
            mu_body_z[i, j] = mu_body[2]

            k += 1

    # The compact state layout now matches the chaos sweep output:
    #
    #   (r, q, v, Omega)
    #
    # Longitude and latitude are the two parameter-grid coordinates.
    # They are not phase-space coordinates; a chaos overview made from this
    # file therefore measures sensitivity to Earth-field/location parameters.
    longitude = Lon[0, :]
    latitude = Lat[:, 0]

    np.savez(
        OUTFILE,
        x=longitude,
        y=latitude,
        lat=Lat,
        lon=Lon,
        Bx=Bx,
        By=By,
        Bz=Bz,
        mu_body_x=mu_body_x,
        mu_body_y=mu_body_y,
        mu_body_z=mu_body_z,
        magnetic_moment_norm=1.0,
        sweep_names=np.array([], dtype=str),
        sweep_combinations=np.empty((1, 0)),
        status=status[np.newaxis, :, :],
        x_final=final_state[np.newaxis, :, :, 0],
        y_final=final_state[np.newaxis, :, :, 1],
        z_final=final_state[np.newaxis, :, :, 2],
        qw_final=final_state[np.newaxis, :, :, 3],
        qx_final=final_state[np.newaxis, :, :, 4],
        qy_final=final_state[np.newaxis, :, :, 5],
        qz_final=final_state[np.newaxis, :, :, 6],
        vx_final=final_state[np.newaxis, :, :, 7],
        vy_final=final_state[np.newaxis, :, :, 8],
        vz_final=final_state[np.newaxis, :, :, 9],
        Ox_final=final_state[np.newaxis, :, :, 10],
        Oy_final=final_state[np.newaxis, :, :, 11],
        Oz_final=final_state[np.newaxis, :, :, 12],
        constraint_residual=final_state[np.newaxis, :, :, 13],
        t_final=t_last[np.newaxis, :, :],
        t_end=np.nanmax(t_last),
        normal=np.asarray(
            read_template_value("normal"),
            dtype=float,
        ),
        contact_point_body=np.asarray(
            read_template_value("contactPointBody"),
            dtype=float,
        ),
        omega_initial=np.asarray(
            read_template_value("omega"),
            dtype=float,
        ),
        quaternion_initial=np.asarray(
            read_template_value("quaternion"),
            dtype=float,
        ),
        contact_friction_coefficient=float(
            read_template_value("contactFrictionCoefficient")[0]
        ),
        contact_normal_load=float(
            read_template_value("contactNormalLoad")[0]
        ),
        contact_friction_smoothing_speed=float(
            read_template_value("contactFrictionSmoothingSpeed")[0]
        ),
        parameter_coordinate_names=np.array([
            "longitude",
            "latitude",
        ]),
        parameter_coordinate_units=np.array([
            "degree",
            "degree",
        ]),
    )

    print("Running four detailed city trajectories...")

    for name, (lat, lon) in CITY_CASES.items():
        run_city_case(
            name,
            lat,
            lon,
            model,
        )

    print("Saved")
    print(OUTFILE)


if __name__ == "__main__":
    main()
