/*
Implements homogeneous constant magnetic fields.
*/

#include "systems.hpp"
#include "utility.hpp"


UniformMagneticField::UniformMagneticField(const vec3& magneticField_) : B0(magneticField_){}

double UniformMagneticField::scalarPotential(const State&, double) const{
    return 0.0;
}


vec3 UniformMagneticField::scalarPotentialGradient(const State&, double) const{
    return vec3::Zero();
}


vec3 UniformMagneticField::vectorPotential(const State& state, double) const{
    return 0.5 * B0.cross(state.r);
}


mat3 UniformMagneticField::vectorPotentialJacobian(const State&, double) const{
    return 0.5 * util::hat(B0);
}


vec3 UniformMagneticField::vectorPotentialTimeDerivative(const State&,double) const{
    return vec3::Zero();
}


vec3 UniformMagneticField::electricField(const State&,double) const{
    return vec3::Zero();
}


vec3 UniformMagneticField::magneticField(const State&, double) const{
    return B0;
}


mat3 UniformMagneticField::magneticFieldJacobian(const State&, double) const{
    return mat3::Zero();
}
