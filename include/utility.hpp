#pragma once
#include "types.hpp"

namespace util{
    quat exponential(const vec3& rotvec);
    quat normalize(const quat& q);
    vec3 rotate(const quat& q, const vec3& v);
    mat3 hat(const vec3& v);
    vec3 vee(const mat3& M);
    vec3 project(const vec3& v, const vec3& n);
    vec3 jacobianInverseSO3(const vec3& rotationVector, const vec3& tangent);
    double smoothSign(double x, double k);
    bool nearZero(double x, double eps=1e-12);
}