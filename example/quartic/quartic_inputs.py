from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def solve_radial_energy(E=None, rho=None, rho_dot=None, L=None, a=None, branch=1, root="upper"):
    """Solve the quartic radial energy equation for exactly one missing variable.

    E = 0.5*rho_dot**2 + L**2/(2*rho**2) + 0.5*a*rho**2 + rho**4/8.

    Parameters
    ----------
    branch : int
        Sign used when solving for rho_dot or L.
    root : str
        When rho is missing, choose 'lower', 'middle', or 'upper' positive root.
    """

    values = {"E": E, "rho": rho, "rho_dot": rho_dot, "L": L, "a": a}
    missing = [name for name, value in values.items() if value is None]
    if len(missing) != 1:
        raise ValueError("Exactly one of E, rho, rho_dot, L, a must be None.")
    missing = missing[0]
    sign = 1.0 if branch >= 0 else -1.0

    if missing == "E":
        E = 0.5*rho_dot**2 + L**2/(2.0*rho**2) + 0.5*a*rho**2 + rho**4/8.0
    elif missing == "rho_dot":
        radicand = 2.0*E - L**2/rho**2 - a*rho**2 - rho**4/4.0
        if radicand < -1e-12:
            raise ValueError("No real rho_dot satisfies the requested energy.")
        rho_dot = sign*np.sqrt(max(0.0, radicand))
    elif missing == "L":
        radicand = rho**2*(2.0*E - rho_dot**2 - a*rho**2 - rho**4/4.0)
        if radicand < -1e-12:
            raise ValueError("No real L satisfies the requested energy.")
        L = sign*np.sqrt(max(0.0, radicand))
    elif missing == "a":
        a = (2.0*E - rho_dot**2 - L**2/rho**2 - rho**4/4.0)/rho**2
    elif missing == "rho":
        q_roots = np.roots([1.0, 4.0*a, -8.0*E, 4.0*L**2])
        positive = sorted(float(r.real) for r in q_roots if abs(r.imag) < 1e-9 and r.real > 0.0)
        if not positive:
            raise ValueError("No positive real radial turning point exists.")
        index = {"lower": 0, "middle": len(positive)//2, "upper": -1}[root]
        rho = np.sqrt(positive[index])
        rho_dot = 0.0

    return {"E": float(E), "rho": float(rho), "rho_dot": float(rho_dot), "L": float(L), "a": float(a)}


def solve_physical_constraint(E=None, L=None, C=None, kappa=None, spin=None, branch=1):
    """Solve 2E + spin*L + C^2/4 = kappa^2 for one missing variable."""

    values = {"E": E, "L": L, "C": C, "kappa": kappa, "spin": spin}
    missing = [name for name, value in values.items() if value is None]
    if len(missing) != 1:
        raise ValueError("Exactly one of E, L, C, kappa, spin must be None.")
    missing = missing[0]
    sign = 1.0 if branch >= 0 else -1.0

    if missing == "E":
        E = 0.5*(kappa**2 - spin*L - C**2/4.0)
    elif missing == "L":
        if abs(spin) < 1e-14:
            raise ValueError("Cannot solve for L when spin=0; the constraint is independent of L.")
        L = (kappa**2 - 2.0*E - C**2/4.0)/spin
    elif missing == "C":
        radicand = kappa**2 - 2.0*E - spin*L
        if radicand < -1e-12:
            raise ValueError("No real C satisfies the physical constraint.")
        C = sign*2.0*np.sqrt(max(0.0, radicand))
    elif missing == "kappa":
        radicand = 2.0*E + spin*L + C**2/4.0
        if radicand < -1e-12:
            raise ValueError("No real kappa satisfies the physical constraint.")
        kappa = sign*np.sqrt(max(0.0, radicand))
    elif missing == "spin":
        if abs(L) < 1e-14:
            raise ValueError("Cannot solve for spin when L=0; the constraint is independent of spin.")
        spin = (kappa**2 - 2.0*E - C**2/4.0)/L

    a = spin**2/4.0 - C/2.0
    return {
        "E": float(E), "L": float(L), "C": float(C),
        "kappa": float(kappa), "spin": float(spin), "a": float(a),
    }


def solve_magnetic_coupling(M, R, I, mu_norm=None, B=None, kappa=None):
    """Solve kappa = mu_norm*B/(M*R**2+I) for one missing coupling variable."""

    missing = [name for name, value in {"mu_norm": mu_norm, "B": B, "kappa": kappa}.items() if value is None]
    if len(missing) != 1:
        raise ValueError("Exactly one of mu_norm, B, kappa must be None.")
    J = M*R**2 + I
    if missing[0] == "kappa":
        kappa = mu_norm*B/J
    elif missing[0] == "B":
        B = kappa*J/mu_norm
    else:
        mu_norm = kappa*J/B
    return {"J": float(J), "mu_norm": float(mu_norm), "B": float(B), "kappa": float(kappa)}


def _quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quaternion_from_xi(xi, body_phase=0.0):
    """Return Eigen-order [w,x,y,z] quaternion mapping body z onto xi."""

    xi = np.asarray(xi, dtype=float)
    xi /= np.linalg.norm(xi)
    if xi[2] > -1.0 + 1e-12:
        w = np.sqrt(0.5*(1.0 + xi[2]))
        q_align = np.array([w, -xi[1]/(2.0*w), xi[0]/(2.0*w), 0.0])
    else:
        q_align = np.array([0.0, 1.0, 0.0, 0.0])
    q_phase = np.array([np.cos(body_phase/2.0), 0.0, 0.0, np.sin(body_phase/2.0)])
    q = _quat_multiply(q_align, q_phase)
    return q/np.linalg.norm(q)


def initial_state_from_reduced(M, R, I, mu_norm, B, spin, C, E, L, theta0=0.0, turning_point="upper", body_phase=0.0):
    """Construct a C++ initial state at a radial turning point from reduced constants."""

    J = M*R**2 + I
    kappa = mu_norm*B/J
    a = spin**2/4.0 - C/2.0
    residual = 2.0*E + spin*L + C**2/4.0 - kappa**2
    if abs(residual) > 1e-8*max(1.0, kappa**2):
        raise ValueError(f"Physical constraint violated: residual={residual:.3e}")
    roots = np.roots([-1.0, -4.0*a, 8.0*E, -4.0*L**2])
    if np.max(np.abs(np.imag(roots))) > 1e-9:
        raise ValueError("Radial cubic does not have three real roots.")
    roots = np.sort(np.real(roots))
    q1, q2, q3 = roots
    if turning_point == "upper":
        q0 = q3
    elif turning_point == "lower":
        q0 = q2
    else:
        raise ValueError("turning_point must be 'upper' or 'lower'.")
    if q0 <= 0.0:
        raise ValueError("Chosen turning point must be positive.")

    rho = np.sqrt(q0)
    y = rho*np.exp(1j*theta0)
    y_dot = 1j*(L/rho)*np.exp(1j*theta0)
    w = y
    w_dot = y_dot + 0.5j*spin*y
    xi_complex = 1j*w_dot/kappa
    xi3 = (q0-C)/(2.0*kappa)
    xi = np.array([xi_complex.real, xi_complex.imag, xi3])
    if abs(np.linalg.norm(xi)-1.0) > 1e-7:
        raise RuntimeError("Initial xi is not unit length.")

    omega_world = np.array([w.real, w.imag, spin])
    velocity = np.array([R*w.imag, -R*w.real, 0.0])
    quaternion = quaternion_from_xi(xi, body_phase)
    rotation = Rotation.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
    Omega_body = rotation.inv().apply(omega_world)

    return {
        "position": np.array([0.0, 0.0, R]),
        "velocity": velocity,
        "omega_world": omega_world,
        "omega_body": Omega_body,
        "quaternion": quaternion,
        "xi": xi,
        "roots": roots,
        "a": a,
        "kappa": kappa,
    }


def initial_state_homoclinic(M, R, I, mu_norm, B, spin, tau, phase0=0.0, body_phase=0.0):
    """Construct a C++ initial state at homoclinic time tau relative to the pulse centre."""

    J = M*R**2 + I
    kappa = mu_norm*B/J
    lambda2 = kappa - spin**2/4.0
    if lambda2 <= 0.0:
        raise ValueError("Homoclinic orbit requires kappa > spin**2/4.")
    lam = np.sqrt(lambda2)
    sech = 1.0/np.cosh(lam*tau)
    rho = 2.0*lam*sech
    rho_dot = -2.0*lam**2*sech*np.tanh(lam*tau)
    phase = phase0 + 0.5*spin*tau
    w = rho*np.exp(1j*phase)
    w_dot = (rho_dot + 0.5j*spin*rho)*np.exp(1j*phase)
    xi_complex = 1j*w_dot/kappa
    xi3 = -1.0 + 2.0*lam**2*sech**2/kappa
    xi = np.array([xi_complex.real, xi_complex.imag, xi3])
    omega_world = np.array([w.real, w.imag, spin])
    velocity = np.array([R*w.imag, -R*w.real, 0.0])
    quaternion = quaternion_from_xi(xi, body_phase)
    rotation = Rotation.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
    Omega_body = rotation.inv().apply(omega_world)
    return {
        "position": np.array([0.0, 0.0, R]),
        "velocity": velocity,
        "omega_world": omega_world,
        "omega_body": Omega_body,
        "quaternion": quaternion,
        "xi": xi,
        "lambda": lam,
        "kappa": kappa,
        "C": 2.0*kappa,
        "E": 0.0,
        "L": 0.0,
        "a": -lam**2,
    }


def write_cpp_input(path, body, state, dt, t_end, field, gravity=9.80665):
    """Write an input compatible with the attached main(8).cpp plus modulatedUniform field."""

    path = Path(path)
    I = body["I"]
    q = state["quaternion"]
    lines = [
        f"mass {body['M']:.17g}",
        f"radius {body['R']:.17g}",
        "charge 0.0",
        f"magneticMoment 0.0 0.0 {body['mu_norm']:.17g}",
        f"inertia {I:.17g} 0 0  0 {I:.17g} 0  0 0 {I:.17g}",
        "magneticPolarizability 0 0 0  0 0 0  0 0 0",
        "",
        f"position {state['position'][0]:.17g} {state['position'][1]:.17g} {state['position'][2]:.17g}",
        f"velocity {state['velocity'][0]:.17g} {state['velocity'][1]:.17g} {state['velocity'][2]:.17g}",
        f"omega {state['omega_body'][0]:.17g} {state['omega_body'][1]:.17g} {state['omega_body'][2]:.17g}",
        f"quaternion {q[0]:.17g} {q[1]:.17g} {q[2]:.17g} {q[3]:.17g}",
        "",
        "solverMode dAlembert",
        "constraint rolling",
        "normal 0 0 1",
        "contactFrictionType none",
        "",
        "gravityType uniform",
        f"gravity 0 0 {-gravity:.17g}",
        "",
    ]

    if field["type"] == "constant":
        lines.extend([
            "emType uniformMagnetic",
            f"magneticField 0 0 {field['B']:.17g}",
        ])
    elif field["type"] == "modulated":
        lines.extend([
            "emType modulatedUniform",
            f"magneticField 0 0 {field['B']:.17g}",
            f"fieldModulationAmplitude {field['amplitude']:.17g}",
            f"fieldAngularFrequency {field['frequency']:.17g}",
            f"fieldPhase {field.get('phase', 0.0):.17g}",
            "electricField 0 0 0",
        ])
    else:
        raise ValueError("Unknown field type.")

    lines.extend([
        "",
        "airType none",
        "rollingResistanceType none",
        "eddyCurrentType none",
        "",
        f"dt {dt:.17g}",
        f"tEnd {t_end:.17g}",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def default_case_definitions():
    """Return a compact selection of exact cases plus one periodically driven case."""

    M = 0.02
    R = 0.01
    I = (2.0/5.0)*M*R**2
    mu_norm = 1.0
    kappa = 25.0
    B = solve_magnetic_coupling(M, R, I, mu_norm=mu_norm, B=None, kappa=kappa)["B"]
    body = {"M": M, "R": R, "I": I, "mu_norm": mu_norm}

    exact = [
        {"name": "hom_s0", "kind": "homoclinic", "spin": 0.0},
        {"name": "hom_s4", "kind": "homoclinic", "spin": 4.0},
        {"name": "hom_sm4", "kind": "homoclinic", "spin": -4.0},
        {"name": "hom_s8", "kind": "homoclinic", "spin": 8.0},
        {"name": "bounded_L10", "kind": "bounded", "spin": 4.0, "C": 45.0, "L": 10.0},
        {"name": "bounded_L5", "kind": "bounded", "spin": 4.0, "C": 48.0, "L": 5.0},
    ]
    for case in exact:
        case["B"] = B
        if case["kind"] == "bounded":
            solved = solve_physical_constraint(E=None, L=case["L"], C=case["C"], kappa=kappa, spin=case["spin"])
            case.update(solved)

    chaos_spin = 4.0
    lam = np.sqrt(kappa-chaos_spin**2/4.0)
    centre_time = 4.0/lam
    drive_frequency = 1.2191318492664007*lam
    chaos = {
        "name": "melnikov_chaos",
        "kind": "chaos",
        "spin": chaos_spin,
        "B": B,
        "kappa": kappa,
        "lambda": lam,
        "centre_time": centre_time,
        "drive_amplitude": 0.12,
        "drive_frequency": drive_frequency,
        "drive_phase": np.pi/2.0-drive_frequency*centre_time,
    }
    return body, exact, chaos


def generate_default_inputs(directory="quartic_inputs"):
    """Generate the exact-comparison and Melnikov input suite."""

    directory = Path(directory)
    body, exact, chaos = default_case_definitions()
    metadata = {}

    for case in exact:
        if case["kind"] == "homoclinic":
            kappa = body["mu_norm"]*case["B"]/(body["M"]*body["R"]**2+body["I"])
            lam = np.sqrt(kappa-case["spin"]**2/4.0)
            centre_time = 4.0/lam
            state = initial_state_homoclinic(
                body["M"], body["R"], body["I"], body["mu_norm"], case["B"], case["spin"],
                tau=-centre_time,
            )
            t_end = 2.0*centre_time
            case.update({"lambda": lam, "centre_time": centre_time, "C": 2*kappa, "E": 0.0, "L": 0.0})
        else:
            state = initial_state_from_reduced(
                body["M"], body["R"], body["I"], body["mu_norm"], case["B"], case["spin"],
                case["C"], case["E"], case["L"],
            )
            q1, q2, q3 = state["roots"]
            from scipy.special import ellipk
            omega_e = 0.5*np.sqrt(q3-q1)
            m = (q3-q2)/(q3-q1)
            radial_period = 2.0*ellipk(m)/omega_e
            t_end = 2.0*radial_period
            case["radial_period"] = float(radial_period)
        write_cpp_input(
            directory/f"{case['name']}.in", body, state, dt=2.0e-4, t_end=t_end,
            field={"type": "constant", "B": case["B"]},
        )
        metadata[case["name"]] = dict(case)

    cstate = initial_state_homoclinic(
        body["M"], body["R"], body["I"], body["mu_norm"], chaos["B"], chaos["spin"],
        tau=-chaos["centre_time"],
    )
    write_cpp_input(
        directory/f"{chaos['name']}.in", body, cstate, dt=1.0e-3, t_end=80.0,
        field={
            "type": "modulated", "B": chaos["B"],
            "amplitude": chaos["drive_amplitude"], "frequency": chaos["drive_frequency"],
            "phase": chaos["drive_phase"],
        },
    )
    metadata[chaos["name"]] = dict(chaos)
    return body, metadata


if __name__ == "__main__":
    body, cases = generate_default_inputs()
    print("Body:", body)
    for name, case in cases.items():
        print(name, case)
