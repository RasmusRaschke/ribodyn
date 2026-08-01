#include "structures.hpp"
#include "types.hpp"
#include "utility.hpp"
#include <cmath>

ElectromagneticField::ElectromagneticField(double spatialStep_, double timeStep_) : spatialStep(spatialStep_), timeStep(timeStep_) {}

/*
Calculates the gradient of the scalar potential phi numerically:
grad(phi)=(d_x phi, d_y phi, d_z phi)
*/
vec3 ElectromagneticField::scalarPotentialGradient(const State& state, double t) const {
    vec3 gradient;
    for(int j=0; j<3; ++j){
        const double h = spatialStep * (1.0 + std::abs(state.r(j)));
        State plus = state;
        State minus = state;
        plus.r(j) += h;
        minus.r(j) -= h;
        gradient(j) = (scalarPotential(plus, t) - scalarPotential(minus, t)) / (2.0 * h);
    }
    return gradient;
}

/*
Calculates the full Jacobian of the vector potential A numerically:
J_ij(A) = d_j A_i
*/
mat3 ElectromagneticField::vectorPotentialJacobian(const State& state, double t) const {
    mat3 jacobian;
    for (int j=0; j<3; ++j){
        const double h = spatialStep * (1.0 + std::abs(state.r(j)));
        State plus = state;
        State minus = state;
        plus.r(j) += h;
        minus.r(j) -= h;
        jacobian.col(j) = (vectorPotential(plus, t) - vectorPotential(minus, t)) / (2.0 * h);
    }
    return jacobian;
}

/*
Calculates the partial time derivative of the vector potential if A=A(r,t).
*/
vec3 ElectromagneticField::vectorPotentialTimeDerivative(const State& state, double t) const {
    const double h = timeStep * (1.0 + std::abs(t));
    return (vectorPotential(state, t+h) - vectorPotential(state, t-h)) / (2.0 * h);
}

/*
Maxwell's equations yield the electric field as
E = -grad(phi) - d_t A
*/
vec3 ElectromagneticField::electricField(const State& state, double t) const {
    return -scalarPotentialGradient(state, t) - vectorPotentialTimeDerivative(state, t);
}

/*
Maxwell's equations yield the magnetic field as
B = rot(A),
where rot(A) is extracted from the Jacobian of A.
*/
vec3 ElectromagneticField::magneticField(const State& state, double t) const {
    const mat3 jacobian = vectorPotentialJacobian(state, t);
    return util::vee(jacobian - jacobian.transpose());
}

/*
Dipole forces are calculated from the gradient of the magnetic field, which is calculated as the Jacobian
J_ij(B) = d_j B_i
*/
mat3 ElectromagneticField::magneticFieldJacobian(const State& state, double t) const {
    mat3 jacobian;
    for(int j=0; j<3; ++j) {
        const double h = spatialStep * (1.0 + std::abs(state.r(j)));
        State plus = state;
        State minus = state;
        plus.r(j) += h;
        minus.r(j) -= h;
        jacobian.col(j) = (magneticField(plus, t) - magneticField(minus, t)) / (2.0 * h);
    }
    return jacobian;
}