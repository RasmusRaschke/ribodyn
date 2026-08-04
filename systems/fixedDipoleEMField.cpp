/*
Implements the magnetic field of an arbitrary number of dipole sources.
The vector potential for a dipole is given by 
A(d) = scale * (mu x d)/d^3
with magnetic field
B(d) = scale * (3r<mu,d>/d^5 - mu/d^3).
We calculate the contribution of each dipole and sum up the field as superposition.
 */

#include "systems.hpp"
#include "utility.hpp"
#include <cmath>
#include <stdexcept>
#include <utility>


FixedDipoleEMField::FixedDipoleEMField(
    std::vector<FixedMagneticDipoleSource> sources_,
    double fieldScale,
    double minimumDistance
)
    : sources(std::move(sources_)),
      scale(fieldScale),
      minimumRadius(minimumDistance){
    if(sources.empty()){
        throw std::invalid_argument(
            "Fixed dipole array must contain at least one source."
        );
    }
    if(!std::isfinite(scale)){
        throw std::invalid_argument(
            "Dipole field scale must be finite."
        );
    }
    if(!(minimumRadius > 0.0)){
        throw std::invalid_argument(
            "Dipole minimum distance must be positive."
        );
    }

    for(const auto& source : sources){
        if(!source.position.allFinite() || !source.moment.allFinite()){
            throw std::invalid_argument(
                "Dipole source data must be finite."
            );
        }
    }
}


void FixedDipoleEMField::validateDistance(const vec3& displacement) const{
    if(displacement.norm() < minimumRadius){
        throw std::runtime_error(
            "Body entered the excluded neighborhood of a point magnetic dipole."
        );
    }
}


double FixedDipoleEMField::scalarPotential(const State&, double) const{
    return 0.0;
}


vec3 FixedDipoleEMField::scalarPotentialGradient(const State&, double) const{
    return vec3::Zero();
}


vec3 FixedDipoleEMField::vectorPotential(const State& state, double) const{
    vec3 result = vec3::Zero();
    for(const auto& source : sources){
        const vec3 d = state.r - source.position;
        validateDistance(d);
        const double radiusSquared = d.squaredNorm();
        const double inverseRadiusCubed = 1.0 / (radiusSquared * std::sqrt(radiusSquared));
        result += scale * source.moment.cross(d) * inverseRadiusCubed;
    }
    return result;
}


mat3 FixedDipoleEMField::vectorPotentialJacobian(const State& state, double) const{
    mat3 result = mat3::Zero();
    for(const auto& source : sources){
        const vec3 d = state.r - source.position;
        validateDistance(d);
        const double radiusSquared = d.squaredNorm();
        const double radius = std::sqrt(radiusSquared);
        const double inverseRadiusCubed = 1.0 / (radiusSquared * radius);
        const double inverseRadiusFifth = inverseRadiusCubed / radiusSquared;
        const vec3 momentCrossDisplacement = source.moment.cross(d);
        result += scale * (util::hat(source.moment) * inverseRadiusCubed - 3.0
                * momentCrossDisplacement * d.transpose() * inverseRadiusFifth);
    }
    return result;
}


vec3 FixedDipoleEMField::vectorPotentialTimeDerivative(const State&, double) const{
    return vec3::Zero();
}


vec3 FixedDipoleEMField::electricField(const State&, double) const{
    return vec3::Zero();
}


vec3 FixedDipoleEMField::magneticField(const State& state, double) const{
    vec3 result = vec3::Zero();
    for (const auto& source : sources){
        const vec3 d = state.r - source.position;
        validateDistance(d);
        const double radiusSquared = d.squaredNorm();
        const double radius = std::sqrt(radiusSquared);
        const double inverseRadiusCubed = 1.0 / (radiusSquared * radius);
        const double inverseRadiusFifth = inverseRadiusCubed / radiusSquared;
        const double momentProjection = source.moment.dot(d);
        result += scale * (3.0 * momentProjection * d
                * inverseRadiusFifth - source.moment * inverseRadiusCubed);
    }
    return result;
}


mat3 FixedDipoleEMField::magneticFieldJacobian(const State& state, double) const{
    mat3 result = mat3::Zero();
    for(const auto& source : sources){
        const vec3 d = state.r - source.position;
        validateDistance(d);
        const double radius = d.norm();
        const double inverseRadiusFifth = 1.0 / (radius * radius * radius);
        const double inverseRadiusSeventh = inverseRadiusFifth / (radius * radius);
        const double momentProjection = source.moment.dot(d);
        result += scale * (3.0 * momentProjection * mat3::Identity() * inverseRadiusFifth
                + 3.0 * d * source.moment.transpose() * inverseRadiusFifth + 3.0
                * source.moment * d.transpose() * inverseRadiusFifth - 15.0 * momentProjection
                * d * d.transpose() * inverseRadiusSeventh
            );
    }
    return result;
}
