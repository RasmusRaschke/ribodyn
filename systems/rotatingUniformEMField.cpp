/*
This implements a magnetic field B(t) rotating around an axis u with frequency omega_u. The relevant potentials in symmetric gauge are:
A(r,t) = 0.5 * B(t) x r
phi(r) = <E_0, r>
Note that Faraday's law implies an induced field
E(r,t) = E_0 - 0.5 * dB/dt x r
 */
#include "systems.hpp"
#include "utility.hpp"
#include <Eigen/Geometry>
#include <stdexcept>

RotatingUniformEMField::RotatingUniformEMField(const vec3& initialMagneticField, const vec3& rotationAxis, double angularFrequency,const vec3& uniformElectricField)
    : B0(initialMagneticField),
      axis(rotationAxis),
      omega(angularFrequency),
      E0(uniformElectricField){
    const double axisNorm = axis.norm();
    if(axisNorm < 1e-14){
        throw std::invalid_argument(
            "EM-field rotation axis must be nonzero."
        );
    }
    axis /= axisNorm;
}

vec3 RotatingUniformEMField::magneticField(const State&, double t) const{
    const Eigen::AngleAxisd rotation(omega * t,axis);
    return rotation * B0;
}

vec3 RotatingUniformEMField::magneticFieldTimeDerivative(double t) const{
    const Eigen::AngleAxisd rotation(omega * t, axis);
    const vec3 B = rotation * B0;
    return omega * axis.cross(B);
}

double RotatingUniformEMField::scalarPotential(const State& state, double) const{
    return -E0.dot(state.r);
}

vec3 RotatingUniformEMField::vectorPotential(const State& state, double t) const{
    return 0.5 * magneticField(state, t).cross(state.r);
}

vec3 RotatingUniformEMField::scalarPotentialGradient(const State&, double) const{
    return -E0;
}

mat3 RotatingUniformEMField::vectorPotentialJacobian(const State& state, double t) const{
    return 0.5 * util::hat(magneticField(state, t));
}

vec3 RotatingUniformEMField::vectorPotentialTimeDerivative(const State& state, double t) const{
    return 0.5 * magneticFieldTimeDerivative(t).cross(state.r);
}

vec3 RotatingUniformEMField::electricField(const State& state, double t) const{
    return E0 - 0.5 * magneticFieldTimeDerivative(t).cross(state.r);
}

mat3 RotatingUniformEMField::magneticFieldJacobian(const State&, double) const{
    return mat3::Zero();
}
