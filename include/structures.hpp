#pragma once

#include <memory>
#include <vector>
#include "types.hpp"

struct State{
    vec3 r;
    vec3 v;
    quat q;
    vec3 Omega;
};

struct Body{
    double mass = 0.0;
    double radius = 0.0;
    double charge = 0.0;
    vec3 magneticMoment = vec3::Zero();
    mat3 magneticPolarizability = mat3::Zero();
    mat3 inertia = mat3::Identity();
};

struct ConstraintData{
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    Eigen::VectorXd gamma;
};

class Constraint{
    public:
        virtual ConstraintData evaluate(
            const State& state,
            double time
        ) const = 0;

        virtual ~Constraint() = default;
};

//Force is computed in the inertial frame, torque in the body frame
struct Wrench{
    vec3 force = vec3::Zero();
    vec3 torque = vec3::Zero();
    Wrench& operator+=(const Wrench& other){
        force += other.force;
        torque += other.torque;
        return *this;
    }
};

class Force{
    public:
        virtual Wrench evaluate(const Body& body, const State& state, double t) const = 0;
        virtual ~Force() = default;
};

class Potential{
    public:
        virtual double value(const Body& body, const State& state, double t) const = 0;
        virtual Wrench gradient(const Body& body, const State& state, double t) const = 0;
        virtual ~Potential() = default;
};

class GravitationalField{
    public:
        virtual vec3 acceleration(const State& state, double t) const = 0;
        virtual double potential(const State& state, double t) const = 0;
        virtual ~GravitationalField() = default;
};

class ElectromagneticField{
    public:
        explicit ElectromagneticField(double spatialStep = 1e-6, double timeStep = 1e-6);
        virtual double scalarPotential(const State& state, double t) const = 0;
        virtual vec3 vectorPotential(const State& state, double t) const = 0;
        virtual vec3 scalarPotentialGradient(const State& state, double t) const;
        virtual mat3 vectorPotentialJacobian(const State& state, double t) const;
        virtual vec3 vectorPotentialTimeDerivative(const State& state, double t) const;
        virtual vec3 electricField(const State& state, double t) const;
        virtual vec3 magneticField(const State& state, double t) const;
        virtual mat3 magneticFieldJacobian(const State& state, double t) const;

        virtual ~ElectromagneticField() = default;

    protected:
        double spatialStep;
        double timeStep;
};

class MechanicalSystem{
    public:
        Body body;
        std::vector<std::unique_ptr<Constraint>> constraints;
        std::vector<std::unique_ptr<Force>> forces;
        std::vector<std::unique_ptr<Potential>> potentials;
        std::vector<std::unique_ptr<GravitationalField>> gravityFields;
        std::vector<std::unique_ptr<ElectromagneticField>> emFields;
};

enum class SolverMode{
    Lagrangian,
    dAlembert
};

struct SimulationParameters{
    double dt;
    double tEnd;
    SolverMode mode;
};

struct InitialCondition{
    State state;
};

struct Diagnostics{
    double T_trans = 0.0;
    double T_rot = 0.0;
    double U_generic = 0.0;
    double U_gr = 0.0;
    double U_em = 0.0;
    double E_total = 0.0;
    vec3 E_world = vec3::Zero();
    vec3 E_body = vec3::Zero();
    vec3 B_world = vec3::Zero();
    vec3 B_body = vec3::Zero();
    vec3 mu_world = vec3::Zero();
    vec3 mu_body = vec3::Zero();
    double constraintResidual = 0.0;
    double quaternionNorm = 1.0;
};