/*
Implements a radial gravitational field U ~ 1/|r|.
*/
#include "structures.hpp"
#include "systems.hpp"
#include "types.hpp"

CentralGravity::CentralGravity(double mu_) : mu(mu_) {}

/*
Calculates the gravitational acceleration in a central field according to 
a = -mu * r/||r||^3
*/
vec3 CentralGravity::acceleration(const State& s, double) const {
    double r = s.r.norm();
    return -mu*s.r / (r*r*r);
}

/*
Calculates the potential at a point given by
U_grav = mu * ||r||
*/
double CentralGravity::potential(const State& s, double) const {
    return -mu / s.r.norm();
}