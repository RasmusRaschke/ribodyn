/*
Implements a rolling-without-slipping constraint for a space and time dependent surface, parametrized by the
surface normal n=n(r,t). We have:
v + rho * n x (R*Omega) = v + rho * R * hat(n)Omega = 0 => A = [id_3 | R*rho*hat(n)], b=0
The tangent space is constraint by: dv/dt + rho*R*[hat(dn/dt) Omega + hat(n)hat(Omega)Omega + hat(n)dOmega/dt = 0,
so gamma = -rho*R*hat(dn/dt)Omega
In practice, we normalize the input normal N(r,t) to n = N/||N|| 
*/

#include "structures.hpp"
#include "systems.hpp"
#include "utility.hpp"
#include <stdexcept>
#include <utility>

RollingConstraint::RollingConstraint(double radius_, const vec3& normal_) : radius(radius_){
    if(!(radius >= 0.0)){
        throw std::invalid_argument(
            "Rolling radius cannot be negative."
        );
    }
    if(normal_.norm() < 1e-14){
        throw std::invalid_argument(
            "Rolling normal cannot be zero."
        );
    }
    const vec3 constantNormal = normal_;
    normalFunction = [constantNormal](const State&, double){
        return constantNormal;
    };
    normalRateFunction = [](const State&, double){
        return vec3::Zero();
    };
}

RollingConstraint::RollingConstraint(double radius_, NormalFunction normalFunction_, NormalRateFunction normalRateFunction_) 
    : radius(radius_),
      normalFunction(std::move(normalFunction_)),
      normalRateFunction(
          std::move(normalRateFunction_)
      )
{
    if(!(radius >= 0.0)){
        throw std::invalid_argument(
            "Rolling radius cannot be negative."
        );
    }
    if(!normalFunction){
        throw std::invalid_argument(
            "Rolling normal function is empty."
        );
    }
    if (!normalRateFunction){
        throw std::invalid_argument(
            "Rolling normal-rate function is empty."
        );
    }
}

ConstraintData RollingConstraint::evaluate(const State& s, double t) const{
    const vec3 rawNormal = normalFunction(s, t);
    const vec3 rawNormalRate = normalRateFunction(s, t);
    const double normalNorm = rawNormal.norm();
    if(normalNorm < 1e-14){
        throw std::runtime_error(
            "Rolling normal function returned zero."
        );
    }
    const vec3 n = rawNormal / normalNorm;
    const vec3 nDot = (mat3::Identity() - n * n.transpose()) * rawNormalRate / normalNorm;
    const mat3 R = s.q.toRotationMatrix();
    ConstraintData c;
    c.A.resize(3, 6);
    c.A.leftCols<3>() = mat3::Identity();
    c.A.rightCols<3>() = radius * util::hat(n) * R;
    c.b = Eigen::VectorXd::Zero(3);
    c.gamma = -radius * util::hat(nDot) * R * s.Omega;
    return c;
}