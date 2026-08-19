/*
Implementation of Kalker's linear contact model of a sphere on an elastic surface. The external force is
    Fx = -G a b C11 xi
    Fy = -G a b C22 eta - G (a b)^(3/2) C23 phi
    Mz = +G (a b)^(3/2) C23 eta - G (a b)^2 C33 phi
with
xi, eta = longitudinal/lateral creepage [-]
phi = spin creepage [1/m]
Note that the theory is undefined in rest, so the spehre should always roll to avoid discontinuities.
*/

#include "structures.hpp"
#include "systems.hpp"
#include <cmath>
#include <stdexcept>


KalkerLinearContact::KalkerLinearContact(
    const vec3& normal_,
    const vec3& surfaceVelocity_,
    double shearModulus_,
    double semiAxisA_,
    double semiAxisB_,
    double c11_,
    double c22_,
    double c23_,
    double c33_
)
    : normal(normal_),
      surfaceVelocity(surfaceVelocity_),
      shearModulus(shearModulus_),
      semiAxisA(semiAxisA_),
      semiAxisB(semiAxisB_),
      c11(c11_),
      c22(c22_),
      c23(c23_),
      c33(c33_){
    if(normal.norm() < 1e-14)
        throw std::invalid_argument(
            "Kalker contact normal cannot be zero."
        );
    normal.normalize();
    if(!(shearModulus > 0.0))
        throw std::invalid_argument(
            "Kalker shear modulus must be positive."
        );
    if(!(semiAxisA > 0.0) || !(semiAxisB > 0.0))
        throw std::invalid_argument(
            "Kalker contact semi-axes must be positive."
        );
    if(!(c11 > 0.0) || !(c22 > 0.0) || !(c23 >= 0.0) || !(c33 > 0.0))
        throw std::invalid_argument(
            "Invalid Kalker coefficients."
        );
}


Wrench KalkerLinearContact::evaluate(const Body& body, const State& state, double) const{
    Wrench wrench;
    const mat3 R = state.q.toRotationMatrix();
    const vec3 omegaWorld = R * state.Omega;
    const mat3 tangentProjector = mat3::Identity() - normal * normal.transpose();
    const vec3 contactArm = -body.radius * normal;
    const vec3 slipVelocity = tangentProjector * (state.v + omegaWorld.cross(contactArm) - surfaceVelocity);
    const vec3 rollingVelocity = tangentProjector * (state.v - surfaceVelocity);
    const double rollingSpeed = rollingVelocity.norm();
    if(rollingSpeed <= 1e-14)
        throw std::runtime_error(
            "Kalker linear theory requires nonzero rolling speed."
        );
    const vec3 e1 = rollingVelocity / rollingSpeed;
    vec3 e2 = normal.cross(e1);
    const double e2Norm = e2.norm();
    if(e2Norm <= 1e-14)
        throw std::runtime_error(
            "Could not construct Kalker contact frame."
        );
    e2 /= e2Norm;
    const double xi = slipVelocity.dot(e1) / rollingSpeed;
    const double eta = slipVelocity.dot(e2) / rollingSpeed;
    const double phi = omegaWorld.dot(normal) / rollingSpeed;
    const double ab = semiAxisA * semiAxisB;
    const double Fx = -shearModulus * ab * c11 * xi;
    const double Fy = -shearModulus * ab * c22 * eta -shearModulus * std::pow(ab, 1.5) * c23 * phi;
    const double Mz = +shearModulus * std::pow(ab, 1.5) * c23 * eta -shearModulus * ab * ab * c33 * phi;
    const vec3 contactForce = Fx * e1 + Fy * e2;
    const vec3 torqueWorld = contactArm.cross(contactForce) + Mz * normal;
    wrench.force = contactForce;
    wrench.torque = R.transpose() * torqueWorld;
    return wrench;
}
