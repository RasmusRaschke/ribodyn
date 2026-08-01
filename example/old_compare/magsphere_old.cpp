#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cmath>
#include <string>
#include <type_traits>

using namespace std;
using vec3 = Eigen::Vector3d;
using quat = Eigen::Quaterniond;
using mat3 = Eigen::Matrix3d;

// STRUCTS
struct Parameters {
    double M;       // mass 
    double R;       // radius
    double g;       // gravitational Acceleration
    double phi;     // Incline Angle, tan phi = alpha
    double I;       // inertia scalar 
    double spin;    // initial spin
    int roll;       // 0 for no resistance, 1 for coulomb rolling, 2 for viscous rolling
    double nu;      // rolling friction coefficient
    double k;       // sign smoothing steepness 
    bool air;       // true for include air resistance
    double rho;     // mass density of air
    double drag;    // drag coefficient
    double area;    // reference area
    vec3 n;         // normal vector field
    vec3 mu_body;   // magnetic moment in body frame
    vec3 B_inert;   // magnetic field in inertial frame
};

struct State {
    vec3 r;
    vec3 Omega;
    quat q;
};

template<typename>
struct always_false : false_type {};

// INPUT HANDLING

string remove_comment(string line) {
    size_t pos = line.find("#");
    if (pos != string::npos) {
        line = line.substr(0, pos);
    }
    line.erase(0, line.find_first_not_of(" \t"));
    line.erase(line.find_last_not_of(" \t") + 1);
    return line;
}

template<typename T>
T get_input_variable(const string& line) {
    string cleaned = remove_comment(line);
    try {
        if constexpr (is_same_v<T, double>) {
            return stod(cleaned);
        }
        else if constexpr (is_same_v<T, int>) {
            return stoi(cleaned);
        }
        else {
            static_assert(always_false<T>::value, "Unsupported type");
        }
    }
    catch (const exception& e){
        cerr << "Error parsing input variable: " << e.what() << "\n";
        exit(1);
    }
}

template<typename T>
T get_next() {
    string line;
    if (!getline(cin, line)) {
        cerr << "Error reading input variable: End of input reached\n";
        exit(1);
    }
    return get_input_variable<T>(line);
}

/*!
 * Calculates the Lie theoretic exponential map so(3) -> SO(3) in quaternion representation
 * @param rotvec Three-dimensional rotation vector describing axis and angle
 * @return Returns the image of exp as quaternion type (Eigen)
 */
static quat exponential(const vec3 &rotvec){
    double theta = rotvec.norm();
    if (theta < 1e-15) return quat::Identity();
    vec3 axis = rotvec / theta;
    double a = cos(0.5 * theta);
    double b = sin(0.5 * theta);
    return quat(a, b * axis.x(), b * axis.y(), b * axis.z());
}

/*!
 * Rotates a given vector by a quaternion
 * @param q Quaternion descibing the rotation
 * @param v 3-vector to be rotated
 * @return Returns rotated vector as vec3 type
 */
static vec3 rotate(const quat &q, const vec3 &v){
    return q * v;
}

/*!
 * Calculates the magnetic torque for a given rotational configuration
 * @param q Unit quaternion describing the rotational state of mu
 * @param mu Magnetic dipole vector at t=0
 * @param B Magnetic field vector
 * @return torque Magnetic torque as 3-vector
 */
static vec3 torque(const quat &q, const vec3 &mu, const vec3 &B){
    vec3 mu_init = rotate(q, mu);
    return mu_init.cross(B);
}

/*!
 * Calculates angular acceleration with or without friction and updates forces
 * @param Omega Angular velocity
 * @param q Rotational state represented as quaternion
 * @param p Parameter structure characterizing the system
 * @param Fr Dry friction force
 * @param Fd Air drag force
 * @param Fext External forces
 * @return Angular acceleration as 3-vector
 */
static vec3 getOmega(const vec3 &Omega, const quat &q, const Parameters &p, vec3 &Fr, vec3 &Fd, vec3 &Fext){
    // calculate inertial and body normals
    const vec3 e3(0.,0.,1.);
    double N = p.M * p.g * p.n.dot(e3);
    // calculate tangential direction of movement
    const double eps = 1e-12;
    vec3 v = - p.R * Omega.cross(p.n);
    vec3 v_t = v - p.n * (p.n.dot(v));
    double speed = v_t.norm();
    Fr.setZero();
    if (p.roll == 1) {
        double sgn = (speed > eps) ? tanh(p.k * speed) : 0.0;
        if (speed > eps) Fr = - p.nu * N * sgn * (v_t / speed);
        else Fr.setZero(); 
    }
    else if (p.roll == 2) {
        Fr = - p.nu * v_t;
    }
    else if (p.roll == 3) {
        if (speed > eps) Fr = - p.nu * N * (v_t / speed);
        else Fr.setZero(); 
    }
    Fd.setZero();
    if (p.air) {
        double modulus = 0.5 * p.rho * p.drag * p.area * speed * speed;
        if (speed > eps) Fd =  - modulus * (v_t / speed);
        else Fd.setZero();
    }
    Fext = Fr + Fd;
    vec3 tau_ext = - p.R * (p.n.cross(Fext));
    vec3 tau_mag = torque(q, p.mu_body, p.B_inert);
    vec3 tau_grav = - ((5.0 * p.g) / (7.0 * p.R)) * p.n.cross(e3);
    vec3 dotOmega_t = tau_grav + (5.0 / (7.0 * p.R * p.R * p.M)) * ((tau_ext + tau_mag) - (p.n.dot((tau_ext + tau_mag))) * p.n);
    double dotOmega_n = (p.n.dot((tau_mag + tau_ext))) / p.I;
    return (dotOmega_t + dotOmega_n * p.n);
}

/*!
 * Calculate the com velocity vector for given angular velocity
 * @param Omega Angular velocity
 * @param p Parameter structure characterizing the system
 * @return COM-velocity as 3-vector
 */
static vec3 getr(const vec3 &Omega, const Parameters &p){
    return p.R * Omega.cross(p.n);
}

/*!
 * Runge-Kutta-Munthe-Kaas (RKMK) algorithm calculates com trajectory, angular velocity and unit quaternions for given initial values
 * @param s Instance of structure describing the state variables of the system at time t
 * @param dt Timespan between two timesteps
 * @param p Parameter structure characterizing the system
 * @param Fr Dry friction force
 * @param Fd Air drag force
 * @param Fext External forces
 * @return State of system at the next increment
 */
static State integrator(const State &s, double dt, const Parameters &p, vec3 &Fr, vec3 &Fd, vec3 &Fext){
    vec3 Omega1 = s.Omega;
    //RKMK Steps
    quat q1 = s.q;
    vec3 k1_O = getOmega(Omega1, q1, p, Fr, Fd, Fext);
    vec3 k1_r = getr(Omega1, p);
    vec3 Omega2 = s.Omega + 0.5 * dt * k1_O;
    quat q2 = exponential(0.5 * dt * Omega1) * s.q;
    vec3 k2_O = getOmega(Omega2, q2, p, Fr, Fd, Fext);
    vec3 k2_r = getr(Omega2, p);
    vec3 Omega3 = s.Omega + 0.5 * dt * k2_O;
    quat q3 = exponential(0.5 * dt * Omega2) * s.q;
    vec3 k3_O = getOmega(Omega3, q3, p, Fr, Fd, Fext);
    vec3 k3_r = getr(Omega3, p);
    vec3 Omega4 = s.Omega + dt * k3_O;
    quat q4 = exponential(dt * Omega3) * s.q;
    vec3 k4_O = getOmega(Omega4, q4, p, Fr, Fd, Fext);
    vec3 k4_r = getr(Omega4, p);  
    //RKMK Evaluation
    vec3 Omega_update = s.Omega + (dt / 6.0) * (k1_O + 2.0 * k2_O + 2.0 * k3_O + k4_O);
    vec3 r_update = s.r + (dt / 6.0) * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r);
    //Quaternion Update
    quat dq = exponential(dt * 0.5 * (s.Omega + Omega_update));
    quat q_update = dq * s.q;
    return State{r_update, Omega_update, q_update};
}

/*!
 * Calculates equations of motion of sphere for given initial data in a text file and writes it to data.csv
 */
int main(){
    cout << "Starting calculation..." << "\n";
    double t = 0.0;
    double dt = get_next<double>();
    double t_end = get_next<double>();

    Parameters p;
    // General Parameters
    p.M = get_next<double>();
    p.R = get_next<double>();
    p.g = get_next<double>();
    p.phi = get_next<double>() * M_PI / 180.0;
    p.n = vec3(0.0, -sin(p.phi), cos(p.phi));
    double mu1 = get_next<double>(); 
    double mu2 = get_next<double>();
    double mu3 = get_next<double>();
    double B1 = get_next<double>();
    double B2 = get_next<double>();
    double B3 = get_next<double>();
    p.mu_body = vec3(mu1, mu2, mu3);
    p.B_inert = vec3(B1, B2, B3);
    p.I = (2.0 / 5.0) * p.M * p.R * p.R;
    p.spin = get_next<double>() * M_PI / 180.0;

    //Rolling Resistance
    p.roll = get_next<int>();
    p.nu = get_next<double>();
    p.k = get_next<double>();

    //Air Resistance
    p.air = get_next<int>();
    p.area = M_PI * p.R * p.R;
    p.rho = get_next<double>();
    p.drag = get_next<double>(); //fixed value for sphere

    int steps = int(t_end / dt);

    State s;
    double x0 = get_next<double>();
    double y0 = get_next<double>();
    s.r = vec3(x0, y0, p.R*(1.0 - p.n.dot(vec3(0,0,1))));
    s.Omega = p.n * p.spin;
    s.q = quat::Identity();
    
    ofstream out("data.csv");
    out << scientific << setprecision(9);
    out << "t,x,y,z,Ox,Oy,Oz,q0,q1,q2,q3,";
    out << "vx, vy, vz, T_trans, T_rot, U_gr, U_em, E, q_norm,";
    out << "Frx, Fry, Frz, Fdx, Fdy, Fdz, Fextx, Fexty, Fextz, mu_x, mu_y, mu_z\n";
    for (int i=0; i <= steps; ++i){
        vec3 Fr, Fd, Fext;
        s = integrator(s, dt, p, Fr, Fd, Fext);
        if ((i % 10) == 0){
            vec3 v = getr(s.Omega, p);
            double T_trans = 0.5 * p.M * v.squaredNorm();
            double T_rot = 0.5 * p.I * s.Omega.squaredNorm();
            double U_gr = p.M * p.g * s.r.z();
            vec3 mu_rot = rotate(s.q, p.mu_body);
            double U_em = - mu_rot.dot(p.B_inert);
            double E = T_trans + T_rot + U_gr + U_em;
            double q_norm = s.q.norm();
            out << t << "," << s.r.x() << "," << s.r.y() << "," << s.r.z() << ","
            << s.Omega.x() << "," << s.Omega.y() << "," << s.Omega.z() << ","
            << s.q.w() << "," << s.q.x() << "," << s.q.y() << "," << s.q.z() << ","
            << v.x() << "," << v.y() << "," << v.z() << "," << T_trans << "," << T_rot << "," << U_gr << "," << U_em << "," << E << "," << q_norm << ","
            << Fr.x() << "," << Fr.y() << "," << Fr.z() << "," << Fd.x() << "," << Fd.y() << "," << Fd.z() << "," << Fext.x() << "," << Fext.y() << "," << Fext.z()
            << "," << mu_rot.x() << "," << mu_rot.y() << "," << mu_rot.z() << "\n";
        }
        t += dt;
    }
    out.close();
    cout << "Done" << "\n";
    return 0;
}
