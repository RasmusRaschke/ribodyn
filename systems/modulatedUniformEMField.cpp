/*
Implements a spatially uniform, periodically modulated magnetic field
    B(t) = [1 + epsilon cos(omega t + phase)] B0.
*/

#include "systems.hpp"
#include "utility.hpp"
#include <cmath>

ModulatedUniformEMField::ModulatedUniformEMField(
    const vec3& magneticField_,
    double modulationAmplitude_,
    double angularFrequency_,
    double phase_,
    const vec3& electricField_
)
    : B0(magneticField_),
      epsilon(modulationAmplitude_),
      omega(angularFrequency_),
      phase(phase_),
      E0(electricField_)
{}


double ModulatedUniformEMField::modulation(double t) const {
    return 1.0 + epsilon * std::cos(omega * t + phase);
}


double ModulatedUniformEMField::modulationDerivative(double t) const {
    return -epsilon * omega * std::sin(omega * t + phase);
}


vec3 ModulatedUniformEMField::field(double t) const {
    return modulation(t) * B0;
}


vec3 ModulatedUniformEMField::fieldDerivative(double t) const {
    return modulationDerivative(t) * B0;
}


double ModulatedUniformEMField::scalarPotential(const State& state, double) const {
    return -E0.dot(state.r);
}


vec3 ModulatedUniformEMField::vectorPotential(const State& state,double t) const {
    return 0.5 * field(t).cross(state.r);
}


vec3 ModulatedUniformEMField::scalarPotentialGradient(const State&, double) const {
    return -E0;
}


mat3 ModulatedUniformEMField::vectorPotentialJacobian(const State&, double t) const {
    return 0.5 * util::hat(field(t));
}


vec3 ModulatedUniformEMField::vectorPotentialTimeDerivative(const State& state, double t) const {
    return 0.5 * fieldDerivative(t).cross(state.r);
}


vec3 ModulatedUniformEMField::electricField(const State& state, double t) const {
    return E0 - vectorPotentialTimeDerivative(state, t);
}


vec3 ModulatedUniformEMField::magneticField(const State&, double t) const {
    return field(t);
}


mat3 ModulatedUniformEMField::magneticFieldJacobian(const State&, double) const {
    return mat3::Zero();
}
