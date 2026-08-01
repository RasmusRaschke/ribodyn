#include "systems.hpp"
#include <cmath>
#include <stdexcept>

RegularizedCoulombRollingResistance::
RegularizedCoulombRollingResistance(
    double frictionCoefficient,
    double normalLoad,
    const vec3& planeNormal,
    double smoothingSpeed,
    const vec3& surfaceVelocity
)
    : mu(frictionCoefficient),
      load(normalLoad),
      normal(planeNormal),
      velocityScale(smoothingSpeed),
      planeVelocity(surfaceVelocity){
    if(mu < 0.0){
        throw std::invalid_argument(
            "Coulomb coefficient cannot be negative."
        );
    }
    if(load < 0.0){
        throw std::invalid_argument(
            "Normal load cannot be negative."
        );
    }
    const double normalNorm = normal.norm();
    if (normalNorm < 1e-14){
        throw std::invalid_argument(
            "Coulomb plane normal must be nonzero."
        );
    }
    normal /= normalNorm;
    if(!(velocityScale > 0.0)){
        throw std::invalid_argument(
            "Coulomb smoothing speed must be positive."
        );
    }
}

Wrench RegularizedCoulombRollingResistance::evaluate(const Body& body, const State& state, double) const{
    Wrench result;
    const vec3 relativeVelocity = state.v - planeVelocity;
    const vec3 tangentialVelocity = relativeVelocity - normal.dot(relativeVelocity) * normal;
    const double speed = tangentialVelocity.norm();
    if(speed < 1e-14 || load == 0.0 || mu == 0.0){
        return result;
    }
    const double regularizedMagnitude = mu * load * std::tanh(speed / velocityScale);
    result.force = -regularizedMagnitude * tangentialVelocity / speed;
    result.torque.setZero();
    return result;
}
