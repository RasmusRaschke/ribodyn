#include "systems.hpp"
#include <cmath>
#include <stdexcept>

namespace{
constexpr double pi =
    3.141592653589793238462643383279502884;
}

SphereAirResistance::SphereAirResistance(
    double airDensity,
    double dynamicViscosity,
    double translationalDragCoefficient,
    double rotationalDragCoefficient,
    const vec3& airVelocity,
    const vec3& airAngularVelocity
)
    : rho(airDensity),
      eta(dynamicViscosity),
      Cd(translationalDragCoefficient),
      Crot(rotationalDragCoefficient),
      flowVelocity(airVelocity),
      flowAngularVelocity(airAngularVelocity){
    if(rho < 0.0){
        throw std::invalid_argument(
            "Air density cannot be negative."
        );
    }
    if(eta < 0.0){
        throw std::invalid_argument(
            "Dynamic viscosity cannot be negative."
        );
    }
    if(Cd < 0.0){
        throw std::invalid_argument(
            "Translational drag coefficient cannot be negative."
        );
    }
    if(Crot < 0.0){
        throw std::invalid_argument(
            "Rotational drag coefficient cannot be negative."
        );
    }
}

Wrench SphereAirResistance::evaluate(const Body& body, const State& state, double) const{
    Wrench result;
    const double radius = body.radius;
    if(radius <= 0.0){
        return result;
    }
    const mat3 R = state.q.toRotationMatrix();
    const vec3 velocityRelative = state.v - flowVelocity;
    const vec3 angularVelocityWorld = R * state.Omega;
    const vec3 angularVelocityRelative = angularVelocityWorld - flowAngularVelocity;
    const double area = pi * radius * radius;
    result.force -= 6.0 * pi * eta * radius * velocityRelative;
    const double relativeSpeed = velocityRelative.norm();
    if(relativeSpeed > 0.0){
        result.force -= 0.5 * rho * Cd * area * relativeSpeed * velocityRelative;
    }

    vec3 torqueWorld = -8.0 * pi * eta * radius * radius * radius * angularVelocityRelative;
    const double relativeAngularSpeed = angularVelocityRelative.norm();
    if(relativeAngularSpeed > 0.0){
        torqueWorld -= 0.5 * rho * Crot * area * radius * radius * radius * relativeAngularSpeed * angularVelocityRelative;
    }
    result.torque = R.transpose() * torqueWorld;
    return result;
}
