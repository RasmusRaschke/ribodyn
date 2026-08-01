from __future__ import annotations
from pathlib import Path
from manim import *
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation
import numpy as np
import simutils as su


DATA_DIR = Path(__file__).resolve().parent / "data"

def load_dataset(name: str):
    datasets = su.extract(DATA_DIR / f"{name}.csv")
    return datasets[name]


def interpolator(t, values):
    return PchipInterpolator(np.asarray(t), np.asarray(values))

def quaternion_matrix(qw, qx, qy, qz):
    q = np.array([qw, qx, qy, qz], dtype=float)
    q /= np.linalg.norm(q)
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def safe_direction(vector, fallback=OUT):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm < 1.0e-12:
        return np.asarray(fallback, dtype=float)
    return vector / norm


class KeplerOrbit(ThreeDScene):
    def construct(self):
        data = load_dataset("orbit")
        t = np.asarray(data.t)
        x = interpolator(t, data.x)
        y = interpolator(t, data.y)
        z = interpolator(t, data.z)
        time = ValueTracker(t[0])

        # Simulation distances are mapped directly to Manim units.
        length_scale = 1.65

        def position(sim_time):
            tt = np.clip(sim_time, t[0], t[-1])
            return length_scale * np.array([x(tt), y(tt), z(tt)])

        self.set_camera_orientation(
            phi=65 * DEGREES,
            theta=-50 * DEGREES,
            zoom=0.92,
        )

        axes = ThreeDAxes(
            x_range=[-5.5, 5.5, 1],
            y_range=[-5.5, 5.5, 1],
            z_range=[-1.5, 1.5, 1],
            x_length=11,
            y_length=11,
            z_length=3,
        )
        axes.set_opacity(0.28)

        star = Sphere(
            radius=0.38,
            resolution=(36, 36),
            fill_opacity=1.0,
            stroke_width=0,
            color=YELLOW,
        )

        planet = Sphere(
            radius=0.14,
            resolution=(30, 30),
            fill_opacity=1.0,
            stroke_width=0,
            color=BLUE,
        )
        planet.add_updater(
            lambda mob: mob.move_to(position(time.get_value()))
        )

        trail = TracedPath(
            planet.get_center,
            stroke_color=BLUE,
            stroke_width=3,
            dissipating_time=None,
        )

        gravity_field = VGroup()
        mu = 1.0
        grid = np.linspace(-4.5, 4.5, 7)

        for gx in grid:
            for gy in grid:
                p = np.array([gx, gy, 0.0])
                radius = np.linalg.norm(p)
                if radius < 0.9:
                    continue
                physical_r = p / length_scale
                g = -mu * physical_r / np.linalg.norm(physical_r) ** 3
                direction = safe_direction(g)
                arrow_length = 0.22 + 0.38 * np.tanh(np.linalg.norm(g))

                gravity_field.add(
                    Arrow3D(
                        start=p,
                        end=p + arrow_length * direction,
                        color=GREY_B,
                        thickness=0.012,
                        height=0.12,
                        base_radius=0.045,
                    )
                )

        title = Text(
            "Eccentric Kepler orbit",
            font_size=32,
        ).to_corner(UL)
        subtitle = MathTex(
            r"\ddot{\mathbf r}=-\mu\frac{\mathbf r}{\|\mathbf r\|^3}",
            font_size=30,
        ).next_to(title, DOWN, aligned_edge=LEFT)
        self.add_fixed_in_frame_mobjects(title, subtitle)
        self.add(axes, gravity_field, star, trail, planet)
        self.play(
            time.animate.set_value(t[-1]),
            run_time=15,
            rate_func=linear,
        )
        self.wait(0.5)


class MagneticDipoleOscillation(ThreeDScene):
    def construct(self):
        data = load_dataset("dipole")
        t = np.asarray(data.t)
        qw = interpolator(t, data.qw)
        qx = interpolator(t, data.qx)
        qy = interpolator(t, data.qy)
        qz = interpolator(t, data.qz)
        mux = interpolator(t, data.mu_world_x)
        muy = interpolator(t, data.mu_world_y)
        muz = interpolator(t, data.mu_world_z)
        bx = interpolator(t, data.Bx_world)
        by = interpolator(t, data.By_world)
        bz = interpolator(t, data.Bz_world)
        energy = interpolator(t, data.E_total)
        time = ValueTracker(t[0])

        def rotation_at(sim_time):
            tt = np.clip(sim_time, t[0], t[-1])
            return quaternion_matrix(
                qw(tt), qx(tt), qy(tt), qz(tt)
            )

        def moment_at(sim_time):
            tt = np.clip(sim_time, t[0], t[-1])
            return np.array([mux(tt), muy(tt), muz(tt)])

        def field_at(sim_time):
            tt = np.clip(sim_time, t[0], t[-1])
            return np.array([bx(tt), by(tt), bz(tt)])

        self.set_camera_orientation(
            phi=66 * DEGREES,
            theta=-42 * DEGREES,
            zoom=0.9,
        )

        body = Sphere(
            radius=0.72,
            resolution=(40, 40),
            fill_opacity=0.72,
            stroke_width=1.2,
            color=GREY_BROWN,
        )

        def make_body_triad():
            R = rotation_at(time.get_value())
            colors = (RED, GREEN, BLUE)
            arrows = VGroup()
            for i, color in enumerate(colors):
                arrows.add(
                    Arrow3D(
                        start=ORIGIN,
                        end=0.98 * R[:, i],
                        color=color,
                        thickness=0.018,
                        height=0.15,
                        base_radius=0.055,
                    )
                )
            return arrows

        body_triad = always_redraw(make_body_triad)

        def make_moment_arrow():
            direction = safe_direction(moment_at(time.get_value()), RIGHT)
            return Arrow3D(
                start=ORIGIN,
                end=1.65 * direction,
                color=RED_E,
                thickness=0.025,
                height=0.20,
                base_radius=0.075,
            )

        moment_arrow = always_redraw(make_moment_arrow)

        moment_tip_trail = TracedPath(
            moment_arrow.get_end,
            stroke_color=RED_E,
            stroke_width=2,
            dissipating_time=2.5,
        )

        def make_magnetic_field():
            direction = safe_direction(field_at(time.get_value()), OUT)
            arrows = VGroup()
            for gx in (-2.5, -1.25, 0.0, 1.25, 2.5):
                for gy in (-2.5, -1.25, 0.0, 1.25, 2.5):
                    start = np.array([gx, gy, -1.4])
                    arrows.add(
                        Arrow3D(
                            start=start,
                            end=start + 0.72 * direction,
                            color=BLUE_D,
                            thickness=0.009,
                            height=0.10,
                            base_radius=0.035,
                        )
                    )
            return arrows

        magnetic_field = always_redraw(make_magnetic_field)

        title = Text(
            "Magnetic dipole oscillation",
            font_size=32,
        ).to_corner(UL)
        subtitle = MathTex(
            r"\boldsymbol{\tau}_b=\boldsymbol{\mu}_b\times\mathbf B_b",
            font_size=30,
        ).next_to(title, DOWN, aligned_edge=LEFT)

        energy_label = Text("Total energy:", font_size=24).to_corner(DL)
        energy_value = DecimalNumber(
            energy(t[0]),
            num_decimal_places=5,
            font_size=24,
        ).next_to(energy_label, RIGHT)
        energy_value.add_updater(
            lambda mob: mob.set_value(energy(time.get_value()))
        )

        self.add_fixed_in_frame_mobjects(
            title,
            subtitle,
            energy_label,
            energy_value,
        )
        self.add(
            magnetic_field,
            moment_tip_trail,
            body,
            body_triad,
            moment_arrow,
        )
        self.play(
            time.animate.set_value(t[-1]),
            run_time=15,
            rate_func=linear,
        )
        self.wait(0.5)


class RotatingMagneticField(ThreeDScene):
    def construct(self):
        data = load_dataset("rotating")
        t = np.asarray(data.t)
        def spline(values):
            return PchipInterpolator(t, np.asarray(values))
        qw = spline(data.qw)
        qx = spline(data.qx)
        qy = spline(data.qy)
        qz = spline(data.qz)
        mux = spline(data.mu_world_x)
        muy = spline(data.mu_world_y)
        muz = spline(data.mu_world_z)
        bx = spline(data.Bx_world)
        by = spline(data.By_world)
        bz = spline(data.Bz_world)
        total_energy = spline(data.E_total)
        time = ValueTracker(t[0])

        def body_rotation(sim_time):
            tt = np.clip(sim_time, t[0], t[-1])
            return quaternion_matrix(qw(tt), qx(tt), qy(tt), qz(tt),)

        def magnetic_moment(sim_time):
            tt = np.clip(sim_time, t[0], t[-1])
            return np.array(
                [mux(tt), muy(tt), muz(tt)]
            )

        def magnetic_field(sim_time):
            tt = np.clip(sim_time, t[0], t[-1])
            return np.array(
                [bx(tt), by(tt), bz(tt)]
            )

        self.set_camera_orientation(
            phi=65 * DEGREES,
            theta=-42 * DEGREES,
            zoom=0.9,
        )

        body = Sphere(
            radius=0.72,
            resolution=(40, 40),
            fill_opacity=0.75,
            stroke_width=1,
            color=GREY_BROWN,
        )

        def make_body_axes():
            rotation = body_rotation(
                time.get_value()
            )

            arrows = VGroup()

            for index, color in enumerate(
                (RED, GREEN, BLUE)
            ):
                arrows.add(
                    Arrow3D(
                        start=ORIGIN,
                        end=0.98 * rotation[:, index],
                        color=color,
                        thickness=0.018,
                        height=0.15,
                        base_radius=0.055,
                    )
                )

            return arrows

        body_axes = always_redraw(
            make_body_axes
        )

        def make_moment_arrow():
            direction = safe_direction(magnetic_moment(time.get_value()), RIGHT,)
            return Arrow3D(
                start=ORIGIN,
                end=1.65 * direction,
                color=RED_E,
                thickness=0.026,
                height=0.20,
                base_radius=0.075,
            )

        moment_arrow = always_redraw(make_moment_arrow)
        moment_trail = TracedPath(
            moment_arrow.get_end,
            stroke_color=RED_E,
            stroke_width=2,
            dissipating_time=3.0,
        )

        def make_field_lattice():
            direction = safe_direction(magnetic_field(time.get_value()), RIGHT,)
            arrows = VGroup()
            for gx in (-2.5, -1.25, 0.0, 1.25, 2.5):
                for gy in (-2.5, -1.25, 0.0, 1.25, 2.5):
                    start = np.array(
                        [gx, gy, -1.35]
                    )

                    arrows.add(
                        Arrow3D(
                            start=start,
                            end=start + 0.72 * direction,
                            color=BLUE_D,
                            thickness=0.009,
                            height=0.10,
                            base_radius=0.035,
                        )
                    )

            return arrows

        field_lattice = always_redraw(
            make_field_lattice
        )

        def make_field_reference_arrow():
            direction = safe_direction(magnetic_field(time.get_value()),RIGHT,)
            return Arrow3D(
                start=ORIGIN,
                end=2.15 * direction,
                color=BLUE_D,
                thickness=0.023,
                height=0.18,
                base_radius=0.07,
            )

        field_arrow = always_redraw(make_field_reference_arrow)

        title = Text(
            "Rigid dipole in a rotating magnetic field",
            font_size=30,
        ).to_corner(UL)

        equation = MathTex(
            r"\mathbf B(t)=R_{\hat{\mathbf z}}(\omega_f t)\mathbf B_0,"
            r"\qquad"
            r"\boldsymbol\tau_b=\boldsymbol\mu_b\times\mathbf B_b",
            font_size=27,
        ).next_to(
            title,
            DOWN,
            aligned_edge=LEFT,
        )

        energy_label = Text(
            "Mechanical energy:",
            font_size=22,
        ).to_corner(DL)

        energy_value = DecimalNumber(
            total_energy(t[0]),
            num_decimal_places=5,
            font_size=22,
        ).next_to(
            energy_label,
            RIGHT,
        )

        energy_value.add_updater(
            lambda mob: mob.set_value(
                total_energy(
                    time.get_value()
                )
            )
        )

        self.add_fixed_in_frame_mobjects(
            title,
            equation,
            note,
            energy_label,
            energy_value,
        )

        self.add(
            field_lattice,
            moment_trail,
            body,
            body_axes,
            field_arrow,
            moment_arrow,
        )

        self.play(
            time.animate.set_value(t[-1]),
            run_time=15,
            rate_func=linear,
        )

        self.wait(0.5)
