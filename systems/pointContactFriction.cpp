/*
Models point-contact friction for a body-fixed tip.
*/

#include "structures.hpp"
#include "systems.hpp"
#include <cmath>
#include <stdexcept>


namespace
{
vec3 pointTangentialVelocity(
    const vec3& contactPointBody,
    const vec3& normal,
    const vec3& surfaceVelocity,
    const State& state,
    mat3& R
)
{
    R = state.q.toRotationMatrix();
    const vec3 omegaWorld = R * state.Omega;
    const vec3 contactPointWorld = R * contactPointBody;
    const vec3 contactVelocity = state.v + omegaWorld.cross(contactPointWorld);
    const vec3 relativeVelocity = contactVelocity - surfaceVelocity;
    return relativeVelocity - normal * normal.dot(relativeVelocity);
}
Wrench pointFrictionWrench(const vec3& contactPointBody, const mat3& R, const vec3& forceWorld){
    Wrench wrench;
    const vec3 forceBody = R.transpose() * forceWorld;
    wrench.force = forceWorld;
    wrench.torque = contactPointBody.cross(forceBody);
    return wrench;
}
}

PointContactViscousFriction::PointContactViscousFriction(
    const vec3& contactPointBody_,
    const vec3& normal_,
    const vec3& surfaceVelocity_,
    double tangentialDamping_
)
    : contactPointBody(contactPointBody_),
      normal(normal_),
      surfaceVelocity(surfaceVelocity_),
      tangentialDamping(tangentialDamping_){
    if(normal.norm() < 1e-14)
        throw std::invalid_argument(
            "Point-contact normal cannot be zero."
        );
    if(contactPointBody.norm() < 1e-14)
        throw std::invalid_argument(
            "Point-contact body offset cannot be zero."
        );
    if(!(tangentialDamping >= 0.0))
        throw std::invalid_argument(
            "Point-contact tangential damping cannot be negative."
        );
    normal.normalize();
}


Wrench PointContactViscousFriction::evaluate(const Body&, const State& state, double) const{
    if(tangentialDamping == 0.0) return Wrench{};
    mat3 R;
    const vec3 tangentialVelocity = pointTangentialVelocity(
        contactPointBody,
        normal,
        surfaceVelocity,
        state,
        R
    );
    const vec3 forceWorld = -tangentialDamping * tangentialVelocity;
    return pointFrictionWrench(contactPointBody, R, forceWorld);
}


PointContactDryFriction::PointContactDryFriction(
    const vec3& contactPointBody_,
    const vec3& normal_,
    const vec3& surfaceVelocity_,
    double frictionCoefficient_,
    double normalLoad_,
    double smoothingSpeed_
)
    : contactPointBody(contactPointBody_),
      normal(normal_),
      surfaceVelocity(surfaceVelocity_),
      frictionCoefficient(frictionCoefficient_),
      normalLoad(normalLoad_),
      smoothingSpeed(smoothingSpeed_){
    if(normal.norm() < 1e-14)
        throw std::invalid_argument(
            "Point-contact normal cannot be zero."
        );
    if(contactPointBody.norm() < 1e-14)
        throw std::invalid_argument(
            "Point-contact body offset cannot be zero."
        );
    if(!(frictionCoefficient >= 0.0))
        throw std::invalid_argument(
            "Point-contact friction coefficient cannot be negative."
        );
    if(!(normalLoad >= 0.0))
        throw std::invalid_argument(
            "Point-contact normal load cannot be negative."
        );
    if(!(smoothingSpeed >= 0.0))
        throw std::invalid_argument(
            "Point-contact friction smoothing speed cannot be negative."
        );
    normal.normalize();
}


Wrench PointContactDryFriction::evaluate(const Body&, const State& state, double) const{
    if(frictionCoefficient == 0.0 || normalLoad == 0.0)
        return Wrench{};
    mat3 R;
    const vec3 tangentialVelocity = pointTangentialVelocity(
        contactPointBody,
        normal,
        surfaceVelocity,
        state,
        R
    );
    const double speed = tangentialVelocity.norm();
    if(speed < 1e-14) 
        return Wrench{};
    double magnitude = frictionCoefficient * normalLoad;
    if(smoothingSpeed > 0.0)
        magnitude *= std::tanh(speed / smoothingSpeed);
    const vec3 forceWorld = -magnitude * tangentialVelocity / speed;
    return pointFrictionWrench(contactPointBody, R, forceWorld);
}
