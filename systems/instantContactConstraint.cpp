/*
Implements contact constraint of a sphere with a plane at the instantaneous contact point,
which need not be fixed in the body frame.
*/

#include "structures.hpp"
#include "systems.hpp"
#include <stdexcept>


InstantContactConstraint::InstantContactConstraint(double radius_, const vec3& normal_, const vec3& surfaceVelocity_)
    : radius(radius_),
      normal(normal_),
      surfaceVelocity(surfaceVelocity_){
    if(!(radius > 0.0))
        throw std::invalid_argument(
            "Sphere-plane contact radius must be positive."
        );
    if(normal.norm() < 1e-14)
        throw std::invalid_argument(
            "Sphere-plane contact normal cannot be zero."
        );
    normal.normalize();
}


ConstraintData InstantContactConstraint::evaluate(const State&, double) const{
    ConstraintData data;
    data.A = Eigen::MatrixXd::Zero(1, 6);
    data.A.block<1, 3>(0, 0) = normal.transpose();
    data.b = Eigen::VectorXd::Constant(1, normal.dot(surfaceVelocity));
    data.gamma = Eigen::VectorXd::Zero(1);
    return data;
}
