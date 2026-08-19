/*
Implements pure normal contact, i.e., fixes the body onto a plane defined by a normal.
*/

#include "structures.hpp"
#include "systems.hpp"
#include <stdexcept>


NormalContactConstraint::NormalContactConstraint(const vec3& normal_, const vec3& surfaceVelocity_)
    : normal(normal_),
      surfaceVelocity(surfaceVelocity_){
    if(normal.norm() < 1e-14)
        throw std::invalid_argument(
            "Normal-contact normal cannot be zero."
        );
    normal.normalize();
}


ConstraintData NormalContactConstraint::evaluate(const State&, double) const{
    ConstraintData data;
    data.A = Eigen::MatrixXd::Zero(1, 6);
    data.A.block<1, 3>(0, 0) = normal.transpose();
    data.b = Eigen::VectorXd::Constant(1, normal.dot(surfaceVelocity));
    data.gamma = Eigen::VectorXd::Zero(1);
    return data;
}
