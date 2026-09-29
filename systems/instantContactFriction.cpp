/*
Models tangential slip friction at the instantaneous contact point of a sphere and a plane.
*/

#include "structures.hpp"
#include "systems.hpp"
#include <cmath>
#include <stdexcept>


namespace
{
    vec3 spherePlaneTangentialVelocity(
        double radius,
        const vec3& normal,
        const vec3& surfaceVelocity,
        const State& state,
        mat3& rotation,
        vec3& leverBody
    ){
        rotation = state.q.toRotationMatrix();
        const vec3 omegaWorld = rotation * state.Omega;
        const vec3 leverWorld = -radius * normal;
        leverBody = rotation.transpose() * leverWorld;
        const vec3 contactVelocity = state.v + omegaWorld.cross(leverWorld);
        const vec3 relativeVelocity = contactVelocity - surfaceVelocity;
        return relativeVelocity - normal * normal.dot(relativeVelocity);
    }

    Wrench spherePlaneFrictionWrench(const vec3& leverBody, const mat3& rotation, const vec3& forceWorld){
        Wrench wrench;
        const vec3 forceBody = rotation.transpose() * forceWorld;
        wrench.force = forceWorld;
        wrench.torque = leverBody.cross(forceBody);
        return wrench;
    }
}


InstantViscousFriction::InstantViscousFriction(double radius_, const vec3& normal_, const vec3& surfaceVelocity_, double tangentialDamping_)
    : radius(radius_), normal(normal_), surfaceVelocity(surfaceVelocity_), tangentialDamping(tangentialDamping_){
    if(!(radius > 0.0))
        throw std::invalid_argument(
            "Sphere-plane friction radius must be positive."
        );
    if(normal.norm() < 1e-14)
        throw std::invalid_argument(
            "Sphere-plane friction normal cannot be zero."
        );
    if(!(tangentialDamping >= 0.0))
        throw std::invalid_argument(
            "Sphere-plane tangential damping cannot be negative."
        );
    normal.normalize();
}


Wrench InstantViscousFriction::evaluate(const Body&, const State& state, double) const{
    if(tangentialDamping == 0.0)
        return Wrench{};
    mat3 rotation;
    vec3 leverBody;
    const vec3 slipVelocity = spherePlaneTangentialVelocity(
        radius,
        normal,
        surfaceVelocity,
        state,
        rotation,
        leverBody
    );
    const vec3 forceWorld = -tangentialDamping * slipVelocity;
    return spherePlaneFrictionWrench(leverBody, rotation, forceWorld);
}


InstantDryFriction::InstantDryFriction(double radius_, const vec3& normal_, const vec3& surfaceVelocity_, double frictionCoefficient_, double normalLoad_, double smoothingSpeed_)
    : radius(radius_), normal(normal_), surfaceVelocity(surfaceVelocity_), frictionCoefficient(frictionCoefficient_), normalLoad(normalLoad_), smoothingSpeed(smoothingSpeed_){
    if(!(radius > 0.0))
        throw std::invalid_argument(
            "Sphere-plane friction radius must be positive."
        );
    if(normal.norm() < 1e-14)
        throw std::invalid_argument(
            "Sphere-plane friction normal cannot be zero."
        );
    if(!(frictionCoefficient >= 0.0))
        throw std::invalid_argument(
            "Sphere-plane friction coefficient cannot be negative."
        );
    if(!(normalLoad >= 0.0))
        throw std::invalid_argument(
            "Sphere-plane normal load cannot be negative."
        );
    if(!(smoothingSpeed >= 0.0))
        throw std::invalid_argument(
            "Sphere-plane friction smoothing speed cannot be negative."
        );
    normal.normalize();
}


Wrench InstantDryFriction::evaluate(const Body&, const State& state, double) const{
    if(frictionCoefficient == 0.0 || normalLoad == 0.0)
        return Wrench{};
    mat3 rotation;
    vec3 leverBody;
    const vec3 slipVelocity = spherePlaneTangentialVelocity(
        radius,
        normal,
        surfaceVelocity,
        state,
        rotation,
        leverBody
    );
    const double speed = slipVelocity.norm();
    if(speed < 1e-14)
        return Wrench{};
    double magnitude = frictionCoefficient * normalLoad;
    if(smoothingSpeed > 0.0)
        magnitude *= std::tanh(speed / smoothingSpeed);

    const vec3 forceWorld = -magnitude * slipVelocity / speed;
    return spherePlaneFrictionWrench(leverBody, rotation, forceWorld);
}
