/*
Implements a uniform electromagnetic field given by a scalar and a vector potential.
The equations above the functions follow from Maxwell's equations in appropriate gauge.
*/
#include "systems.hpp"
#include "utility.hpp"

#include "systems.hpp"
#include "utility.hpp"

UniformEMField::UniformEMField(const vec3& electricField_, const vec3& magneticField_) : E0(electricField_), B0(magneticField_){}

// phi = - <E_0, r>
double UniformEMField::scalarPotential(const State& state, double) const {
    return -E0.dot(state.r);
}

// A = 0.5 * B_0 x r
vec3 UniformEMField::vectorPotential(const State& state, double) const{
    return 0.5 * B0.cross(state.r);
}

// grad(phi) = -E_0
vec3 UniformEMField::scalarPotentialGradient(const State&, double) const{
    return -E0;
}

// J(A) = 0.5 B_0 x (-)
mat3 UniformEMField::vectorPotentialJacobian(const State&, double) const{
    return 0.5 * util::hat(B0);
}

// A = A(r), so dA/dt = 0
vec3 UniformEMField::vectorPotentialTimeDerivative(const State&, double) const{
    return vec3::Zero();
}

// E = E_0
vec3 UniformEMField::electricField(const State&, double) const{
    return E0;
}

// B = B_0
vec3 UniformEMField::magneticField(const State&, double) const{
    return B0;
}

// dB/dt = 0
mat3 UniformEMField::magneticFieldJacobian(const State&, double) const{
    return mat3::Zero();
}