/*
Implements a uniform gravitational field with a = mg.
*/
#include "structures.hpp"
#include "systems.hpp"

UniformGravity::UniformGravity(const vec3& gravity) : g(gravity) {}

vec3 UniformGravity::acceleration(const State&, double) const {
    return g;
}

double UniformGravity::potential(const State& state, double) const {
    return -g.dot(state.r);
}