#pragma once
#include "structures.hpp"

struct StateDerivative{
    vec3 rDot;
    vec3 vDot;
    vec3 Omega;
    vec3 OmegaDot;
};

class Solver{
    public:
        Solver(const MechanicalSystem&, SolverMode);
        StateDerivative rhs(const State&, double t);
        void projectVelocity(State& state, double t) const;
        double constraintResidual(const State& state, double t) const;
        const Eigen::VectorXd& getLambda() const;
        Diagnostics diagnostics(const State& state, double t) const;
    private:
        const MechanicalSystem& system;
        SolverMode mode;
        Eigen::VectorXd lastLambda;
        mat6 massMatrix() const;
        Wrench assembleGeneralizedForce(const State&, double) const;
        Wrench assembleLagrangianWrench(const State&, double) const;
        ConstraintData assembleConstraints(const State& state, double t) const;
};

