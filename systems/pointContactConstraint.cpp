/*
Implements the constraint of a top with a contact point fixed to a plane.
*/

#include "structures.hpp"
#include "systems.hpp"
#include "utility.hpp"
#include <stdexcept>


PointContactConstraint::PointContactConstraint(
    const vec3& contactPointBody_,
    const vec3& normal_,
    const vec3& surfaceVelocity_
)
    : contactPointBody(contactPointBody_),
      normal(normal_),
      surfaceVelocity(surfaceVelocity_){
    if(normal.norm() < 1e-14)
        throw std::invalid_argument(
            "Point-contact normal cannot be zero."
        );
    if(contactPointBody.norm() < 1e-14)
        throw std::invalid_argument(
            "Point-contact body offset cannot be zero."
        );
    normal.normalize();
}


ConstraintData PointContactConstraint::evaluate(const State& state, double) const{
    ConstraintData data;
    const mat3 R = state.q.toRotationMatrix();
    data.A = Eigen::MatrixXd::Zero(1, 6);
    data.A.block<1, 3>(0, 0) = normal.transpose();
    data.A.block<1, 3>(0, 3) = -normal.transpose() * R * util::hat(contactPointBody);
    data.b = Eigen::VectorXd::Constant(1, normal.dot(surfaceVelocity));
    const vec3 centripetalBody = state.Omega.cross(state.Omega.cross(contactPointBody));
    data.gamma = Eigen::VectorXd::Constant(1, -normal.dot(R * centripetalBody));
    return data;
}
