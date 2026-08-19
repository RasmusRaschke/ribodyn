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
# PARAMETERS
YEAR = 2026
MONTH = 6
DAY = 29
ALTITUDE_M = 0.0
DLAT = 1.0          # latitude spacing [deg]
DLON = 1.0          # longitude spacing [deg]

SOLVER = Path("../../build/solver").resolve()
INPUT_TEMPLATE = Path("input.base")

OUTPUT_DIR = Path("results")
OUTFILE = OUTPUT_DIR / "earth_results.npz"
MAX_CORES = 40

# Surface inclination:
# n = (0, -sin(phi), cos(phi))
PHI_DEG = 0.0

# Initial angular velocity around the surface normal [rad/s].
OMEGA_NORMAL = 0.0

# Exact positions used for the four trajectories in combined.py.
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


def make_input(bx, by, bz):
    base = INPUT_TEMPLATE.read_text()

    radius = float(read_template_value("radius")[0])

    phi = np.deg2rad(PHI_DEG)
    normal = np.array([
        0.0,
        -np.sin(phi),
        np.cos(phi),
    ])

    # Contact point initially at the origin.
    position = radius * normal

    # State::Omega is body-frame. With the identity initial quaternion,
    # body and world axes coincide initially.
    omega = OMEGA_NORMAL * normal

    overrides = f"""
# earth_grid.py overrides
position {position[0]:.17g} {position[1]:.17g} {position[2]:.17g}
velocity 0.0 0.0 0.0
omega {omega[0]:.17g} {omega[1]:.17g} {omega[2]:.17g}
quaternion 1.0 0.0 0.0 0.0

constraint rolling
normal {normal[0]:.17g} {normal[1]:.17g} {normal[2]:.17g}

emType homogeneousMagnetic
magneticField {bx:.17g} {by:.17g} {bz:.17g}
"""

    return base.rstrip() + "\n\n" + overrides.lstrip()


def read_last_values(csv_file):
    df = pd.read_csv(csv_file)
    x_last = float(df["x"].iloc[-1])
    t_last = float(df["t"].iloc[-1])
    return x_last, t_last


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
                np.nan,
                np.nan,
                process.stderr,
            )

        x_last, t_last = read_last_values(output_file)

    return (
        index,
        x_last,
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

    x_flat = np.full(len(cases), np.nan)
    t_flat = np.full(len(cases), np.nan)

    failures = 0

    for index, x_last, t_last, error in results:
        x_flat[index] = x_last
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

    x_last = x_flat.reshape(nlat, nlon)
    t_last = t_flat.reshape(nlat, nlon)

    Bx = np.empty((nlat, nlon))
    By = np.empty((nlat, nlon))
    Bz = np.empty((nlat, nlon))

    k = 0

    for i in range(nlat):
        for j in range(nlon):
            bx, by, bz = cases[k][2:]
            Bx[i, j] = bx
            By[i, j] = by
            Bz[i, j] = bz
            k += 1

    np.savez(
        OUTFILE,
        lat=Lat,
        lon=Lon,
        Bx=Bx,
        By=By,
        Bz=Bz,
        x=x_last,
        t=t_last,
        phi_deg=PHI_DEG,
        omega_normal=OMEGA_NORMAL,
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
