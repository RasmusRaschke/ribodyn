#include "integrator.hpp"
#include "solver.hpp"
#include "utility.hpp"

/*
Runge-Kutta-Munthe-Kaas Algorithm calculating a derivative up to fourth order in both R^3 and so(3).
*/
State RKMK4::step(const State& s, Solver& solver, double t, double dt){
    State stage1 = s;
    solver.projectVelocity(stage1, t);
    const StateDerivative k1 = solver.rhs(stage1, t);
    const vec3 u1 = vec3::Zero();
    const vec3 K1 = util::jacobianInverseSO3(u1, k1.Omega);

    State stage2 = s;
    stage2.r += 0.5 * dt * k1.rDot;
    stage2.v += 0.5 * dt * k1.vDot;
    stage2.Omega += 0.5 * dt * k1.OmegaDot;
    const vec3 u2 = 0.5 * dt * K1;
    stage2.q = s.q * util::exponential(u2);
    stage2.q.normalize();
    solver.projectVelocity(stage2,t + 0.5 * dt);
    const StateDerivative k2 =solver.rhs(stage2, t + 0.5 * dt);
    const vec3 K2 =util::jacobianInverseSO3(u2, k2.Omega);

    State stage3 = s;
    stage3.r += 0.5 * dt * k2.rDot;
    stage3.v += 0.5 * dt * k2.vDot;
    stage3.Omega += 0.5 * dt * k2.OmegaDot;
    const vec3 u3 = 0.5 * dt * K2;
    stage3.q = s.q * util::exponential(u3);
    stage3.q.normalize();
    solver.projectVelocity(stage3, t + 0.5 * dt);
    const StateDerivative k3 = solver.rhs(stage3, t + 0.5 * dt);
    const vec3 K3 = util::jacobianInverseSO3(u3, k3.Omega);

    State stage4 = s;
    stage4.r += dt * k3.rDot;
    stage4.v += dt * k3.vDot;
    stage4.Omega += dt * k3.OmegaDot;
    const vec3 u4 = dt * K3;
    stage4.q = s.q * util::exponential(u4);
    stage4.q.normalize();
    solver.projectVelocity(stage4, t + dt);
    const StateDerivative k4 =solver.rhs(stage4, t + dt);
    const vec3 K4 = util::jacobianInverseSO3(u4,k4.Omega);

    State next = s;
    next.r += dt / 6.0 * (
            k1.rDot
            + 2.0 * k2.rDot
            + 2.0 * k3.rDot
            + k4.rDot
        );
    next.v += dt / 6.0 * (
            k1.vDot
            + 2.0 * k2.vDot
            + 2.0 * k3.vDot
            + k4.vDot
        );
    next.Omega += dt / 6.0 * (
            k1.OmegaDot
            + 2.0 * k2.OmegaDot
            + 2.0 * k3.OmegaDot
            + k4.OmegaDot
        );
    const vec3 finalIncrement = dt / 6.0 * (
            K1
            + 2.0 * K2
            + 2.0 * K3
            + K4
        );
    next.q = s.q * util::exponential(finalIncrement);
    next.q.normalize();
    solver.projectVelocity(next, t + dt);
    return next;
}