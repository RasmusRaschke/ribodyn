#pragma once
#include "types.hpp"
#include "structures.hpp"
#include <functional>

class RollingConstraint final : public Constraint{
    public:
        using NormalFunction = std::function<vec3(const State&, double)>;
        using NormalRateFunction = std::function<vec3(const State&, double)>;
        RollingConstraint(double radius, const vec3& normal);
        RollingConstraint(double radius, NormalFunction normalFunction, NormalRateFunction normalRateFunction);
        ConstraintData evaluate(const State& state, double t) const override;
    private:
        double radius;
        NormalFunction normalFunction;
        NormalRateFunction normalRateFunction;
};

class CentralGravity final : public GravitationalField {
    private:
        double mu;
    public:
        explicit CentralGravity(double mu);
        vec3 acceleration(const State& state, double t) const override;
        double potential(const State& state, double t) const override;
};

class UniformGravity final : public GravitationalField {
    private:
        vec3 g;
    public:
        explicit UniformGravity(const vec3& gravity);
        vec3 acceleration(const State&, double t) const override;
        double potential(const State&, double t) const override;
};

class UniformEMField final : public ElectromagneticField {
    private:
        vec3 E0;
        vec3 B0;
    public:
        UniformEMField(const vec3& electricField, const vec3& magneticField);
    double scalarPotential(const State& state, double t) const override;
    vec3 vectorPotential(const State& state, double t) const override;
    vec3 scalarPotentialGradient(const State& state, double t) const override;
    mat3 vectorPotentialJacobian(const State& state, double t) const override;
    vec3 vectorPotentialTimeDerivative(const State& state, double t) const override;
    vec3 electricField(const State& state, double t) const override;
    vec3 magneticField(const State& state, double t) const override;
    mat3 magneticFieldJacobian(const State& state, double t) const override;
};

class RotatingUniformEMField final : public ElectromagneticField{
    private:
        vec3 B0;
        vec3 axis;
        double omega;
        vec3 E0;
        vec3 magneticFieldTimeDerivative(double t) const;
    public:
        RotatingUniformEMField(const vec3& initialMagneticField, const vec3& rotationAxis, double angularFrequency, const vec3& uniformElectricField = vec3::Zero());
        double scalarPotential(const State& state, double t) const override;
        vec3 vectorPotential(const State& state, double t) const override;
        vec3 scalarPotentialGradient(const State& state, double t) const override;
        mat3 vectorPotentialJacobian(const State& state, double t) const override;
        vec3 vectorPotentialTimeDerivative(const State& state, double t) const override;
        vec3 electricField(const State& state, double t) const override;
        vec3 magneticField(const State& state, double t) const override;
        mat3 magneticFieldJacobian(const State& state, double t) const override;
};

class SphereAirResistance final : public Force{
    private:
        double rho;
        double eta;
        double Cd;
        double Crot;
        vec3 flowVelocity;
        vec3 flowAngularVelocity;
    public:
        SphereAirResistance(
            double airDensity,
            double dynamicViscosity,
            double translationalDragCoefficient,
            double rotationalDragCoefficient,
            const vec3& airVelocity = vec3::Zero(),
            const vec3& airAngularVelocity = vec3::Zero()
        );
        Wrench evaluate(const Body& body, const State& state, double t) const override;
};

class RegularizedCoulombRollingResistance final : public Force{
    private:
        double mu;
        double load;
        vec3 normal;
        double velocityScale;
        vec3 planeVelocity;
    public:
        RegularizedCoulombRollingResistance(
            double frictionCoefficient,
            double normalLoad,
            const vec3& planeNormal,
            double smoothingSpeed,
            const vec3& surfaceVelocity = vec3::Zero()
        );
        Wrench evaluate(const Body& body, const State& state, double t) const override;
};