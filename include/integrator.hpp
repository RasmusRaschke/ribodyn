#pragma once
#include "solver.hpp"

class Integrator{
    public:
        virtual State step(
            const State&,
            Solver&,
            double t,
            double dt
        ) = 0;

        virtual ~Integrator() = default;
};

class RKMK4 : public Integrator{
    public:
        State step(const State& state, Solver& solver, double t, double dt) override;
};