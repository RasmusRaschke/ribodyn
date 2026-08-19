/*
Implements fixed electric monopols.
*/

#include "systems.hpp"
#include <stdexcept>
#include <utility>


FixedMonopoleEMField::FixedMonopoleEMField(
    std::vector<vec3> positions_,
    std::vector<double> charges_,
    double fieldScale_,
    double minimumDistance_
)
    : positions(std::move(positions_)),
      charges(std::move(charges_)),
      fieldScale(fieldScale_),
      minimumDistance(minimumDistance_){
    if(positions.size() != charges.size()){
        throw std::invalid_argument(
            "Monopole positions and charges must have the same size."
        );
    }
    if(!(minimumDistance > 0.0)){
        throw std::invalid_argument(
            "Monopole minimum distance must be positive."
        );
    }
}


double FixedMonopoleEMField::scalarPotential(const State& state, double) const{
    double potential = 0.0;
    for(std::size_t i = 0; i < positions.size(); ++i){
        const vec3 displacement = state.r - positions[i];
        const double distance = displacement.norm();
        if(distance < minimumDistance){
            throw std::runtime_error(
                "Electric monopole evaluated inside minimum distance."
            );
        }
        potential += fieldScale * charges[i] / distance;
    }
    return potential;
}


vec3 FixedMonopoleEMField::scalarPotentialGradient(const State& state, double t) const{
    return -electricField(state, t);
}


vec3 FixedMonopoleEMField::vectorPotential(const State&,double) const{
    return vec3::Zero();
}


mat3 FixedMonopoleEMField::vectorPotentialJacobian(const State&,double) const{
    return mat3::Zero();
}


vec3 FixedMonopoleEMField::vectorPotentialTimeDerivative(const State&, double) const{
    return vec3::Zero();
}


vec3 FixedMonopoleEMField::electricField(const State& state, double) const{
    vec3 field = vec3::Zero();
    for(std::size_t i = 0; i < positions.size(); ++i){
        const vec3 displacement = state.r - positions[i];
        const double distance = displacement.norm();
        if(distance < minimumDistance){
            throw std::runtime_error(
                "Electric monopole evaluated inside minimum distance."
            );
        }
        field += fieldScale * charges[i] * displacement / (distance * distance * distance);
    }
    return field;
}


vec3 FixedMonopoleEMField::magneticField(const State&, double) const{
    return vec3::Zero();
}


mat3 FixedMonopoleEMField::magneticFieldJacobian(const State&, double) const{
    return mat3::Zero();
}
