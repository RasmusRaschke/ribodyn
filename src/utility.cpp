#include "utility.hpp"
#include <cmath>

using namespace std;

/*!
Calculates the Lie exponential, i.e. maps a infinitesimal rotation vector Theta=theta*u
to the finite rotation quaternion q=(cos theta/2, u sin theta/2).
 */
quat util::exponential(const vec3 &rotvec){
    double theta = rotvec.norm();
    if (theta < 1e-15) return quat::Identity();
    vec3 axis = rotvec / theta;
    double a = cos(0.5 * theta);
    double b = sin(0.5 * theta);
    return quat(a, b * axis.x(), b * axis.y(), b * axis.z());
}

/*!
Rotates a vector by a rotation quaternion.
 */
vec3 util::rotate(const quat &q, const vec3 &v){
    return q * v;
}

/*
Normalizes a quaternion.
*/
quat util::normalize(const quat& q){
    quat out = q;
    out.normalize();
    return out;
}

/*
Maps the cross product with a given vector v to the corresponding matrix representation M_v(w)=v.cross(w).
*/
mat3 util::hat(const vec3& v){
    mat3 M;
    M << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
    return M; 
}

/*
Inverse of the hat-isomorphism.
*/
vec3 util::vee(const mat3& M){
    return vec3(M(2,1), M(0,2), M(1,0));
}

/*
Smooth sign functions by a tanh with steepness parameter k.
*/
double util::smoothSign(double x, double k){
    return std::tanh(k*x);
}

/*
Check if a double is very close to zero.
*/
bool util::nearZero(double x, double eps){
    return std::abs(x) < eps;
}

/*
Returns surface parallel part of a vector v according to:
v_|| = v - <v,n>v
*/
vec3 util::project(const vec3& v, const vec3& n){
    return v - n.dot(v)*n;
}

/*
The derivative of an element in SO(3) to fourth order used in RKMK4. The inverse Jacobian acts on Omega as
J(u)^{-1}Omega = Omega + 0.5 u x Omega + 1/13 u x (u x Omega) + O(||u||^4).
*/
vec3 util::jacobianInverseSO3(const vec3& rotationVector, const vec3& tangent){
    const double thetaSquared = rotationVector.squaredNorm();
    const vec3 firstCommutator = rotationVector.cross(tangent);
    const vec3 secondCommutator = rotationVector.cross(firstCommutator);
    if(thetaSquared < 1e-12){
        const double coefficient = 1.0 / 12.0 + thetaSquared / 720.0;
        return tangent + 0.5 * firstCommutator + coefficient * secondCommutator;
    }
    const double theta = std::sqrt(thetaSquared);
    const double coefficient = (1.0 - 0.5 * theta / std::tan(0.5 * theta)) / thetaSquared;
    return tangent + 0.5 * firstCommutator + coefficient * secondCommutator;
}