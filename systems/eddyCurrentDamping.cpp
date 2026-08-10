/*
Implements low-frequency Eddy current mediated damping for a conducting sphere in an external field.
*/
#include "systems.hpp"
#include <cmath>
#include <stdexcept>

SphereEddyCurrentDamping::SphereEddyCurrentDamping(
    double conductivity_,
    const std::vector<std::unique_ptr<ElectromagneticField>>& emFields,
    double timeStep_
)
    : conductivity(conductivity_), timeStep(timeStep_){
    if(conductivity < 0.0){
        throw std::invalid_argument(
            "Eddy-current conductivity cannot be negative."
        );
    }
    if(!(timeStep > 0.0)){
        throw std::invalid_argument(
            "Eddy-current time step must be positive."
        );
    }
    fields.reserve(emFields.size());
    for(const auto& field : emFields){
        fields.push_back(field.get());
    }
}

Wrench SphereEddyCurrentDamping::evaluate(const Body& body, const State& state, double t) const{
    Wrench result;
    const double radius = body.radius;
    if(radius <= 0.0 || conductivity == 0.0 || fields.empty()){
        return result;
    }
    vec3 BWorld = vec3::Zero();
    vec3 BTimeDerivativeWorld = vec3::Zero();
    mat3 fieldJacobian = mat3::Zero();
    const double h = timeStep * (1.0 + std::abs(t));
    for(const ElectromagneticField* field : fields){
        BWorld += field->magneticField(state, t);
        fieldJacobian += field->magneticFieldJacobian(state, t);
        BTimeDerivativeWorld += (field->magneticField(state, t + h) - field->magneticField(state, t - h)) / (2.0 * h);
    }
    const mat3 R = state.q.toRotationMatrix();
    const vec3 BBody = R.transpose() * BWorld;
    const vec3 BMaterialDerivativeWorld = BTimeDerivativeWorld + fieldJacobian * state.v;
    const vec3 BTimeDerivativeBody = R.transpose() * BMaterialDerivativeWorld - state.Omega.cross(BBody);
    const double pi = std::acos(-1.0);
    const double dampingCoefficient = 2.0 * pi / 15.0 * conductivity * std::pow(radius, 5);
    const vec3 momentBody = -dampingCoefficient * BTimeDerivativeBody;
    const vec3 momentWorld = R * momentBody;
    result.force = fieldJacobian.transpose() * momentWorld;
    result.torque = momentBody.cross(BBody);
    return result;
}

ViscousEddyCurrentDamping::ViscousEddyCurrentDamping(
    double translationalDamping_,
    double rotationalDamping_
)
    : translationalDamping(translationalDamping_), rotationalDamping(rotationalDamping_){
    if(translationalDamping < 0.0){
        throw std::invalid_argument(
            "Translational eddy-current damping cannot be negative."
        );
    }
    if(rotationalDamping < 0.0){
        throw std::invalid_argument(
            "Rotational eddy-current damping cannot be negative."
        );
    }
}

Wrench ViscousEddyCurrentDamping::evaluate(const Body&, const State& state, double) const{
    Wrench result;
    result.force = -translationalDamping * state.v;
    result.torque = -rotationalDamping * state.Omega;
    return result;
}
