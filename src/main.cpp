#include "integrator.hpp"
#include "io.hpp"
#include "solver.hpp"
#include "structures.hpp"
#include "systems.hpp"
#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace{
SolverMode parseSolverMode(const std::string& value){
    if (value == "dAlembert")
        return SolverMode::dAlembert;
    if (value == "lagrangian")
        return SolverMode::Lagrangian;
    throw std::runtime_error(
        "Unknown solver mode: " + value
    );
}

void validateBody(const Body& body){
    if(!(body.mass > 0.0)){
        throw std::runtime_error(
            "Body mass must be positive."
        );
    }
    if(!(body.radius >= 0.0)){
        throw std::runtime_error(
            "Body radius cannot be negative."
        );
    }
    if(!body.inertia.isApprox(body.inertia.transpose(), 1e-12)){
        throw std::runtime_error(
            "Inertia tensor must be symmetric."
        );
    }
    Eigen::LLT<mat3> decomposition(body.inertia);
    if(decomposition.info() != Eigen::Success){
        throw std::runtime_error(
            "Inertia tensor must be positive definite."
        );
    }
    if(!body.magneticPolarizability.isApprox(body.magneticPolarizability.transpose(), 1e-12)){
        throw std::runtime_error(
            "Magnetic polarizability tensor must be symmetric."
        );
    }
}

void addConstraint(MechanicalSystem& system, const Input& input){
    const std::string type = input.getString("constraint");
    if(type == "none")
        return;
    if(type == "rolling"){
        vec3 normal = input.getVec3("normal");
        const double norm = normal.norm();
        if(norm < 1e-14){
            throw std::runtime_error(
                "Rolling normal must be nonzero."
            );
        }
        normal /= norm;
        system.constraints.push_back(std::make_unique<RollingConstraint>(system.body.radius, normal));
        return;
    }
    throw std::runtime_error(
        "Unknown constraint type: " + type
    );
}

void addGravity(MechanicalSystem& system, const Input& input){
    const std::string type = input.getString("gravityType");
    if(type == "none")
        return;
    if(type == "uniform"){
        system.gravityFields.push_back(std::make_unique<UniformGravity>(input.getVec3("gravity")));
        return;
    }
    if(type == "central"){
        system.gravityFields.push_back(std::make_unique<CentralGravity>(input.getDouble("mu")));
        return;
    }
    throw std::runtime_error(
        "Unknown gravity type: " + type
    );
}

void addElectromagnetism(MechanicalSystem& system, const Input& input){
    const std::string type = input.getString("emType");
    if(type == "none")
        return;
    if(type == "uniform"){
        system.emFields.push_back(std::make_unique<UniformEMField>(input.getVec3("electricField"),input.getVec3("magneticField")));
        return;
    }
    if(type == "rotatingUniform"){
        system.emFields.push_back(std::make_unique<RotatingUniformEMField>(
            input.getVec3("initialMagneticField"), 
            input.getVec3("fieldRotationAxis"), 
            input.getDouble("fieldAngularFrequency"), 
            input.getVec3("electricField")
        ));
        return;
    }
    throw std::runtime_error(
        "Unknown EM field type: " + type
    );
}
}

int main(int argc, char* argv[]){
    try
    {
        if (argc < 2)
        {
            std::cerr
                << "Usage: Solver <input-file>\n";
            return 1;
        }
        Input input;
        if (!input.read(argv[1]))
        {
            std::cerr
                << "Could not open input file.\n";
            return 1;
        }
        MechanicalSystem system;
        system.body.mass = input.getDouble("mass");
        system.body.radius = input.getDouble("radius");
        system.body.charge = input.getDouble("charge");
        system.body.magneticMoment = input.getVec3("magneticMoment");
        system.body.inertia = input.getMat3("inertia");
        system.body.magneticPolarizability = input.getMat3("magneticPolarizability");
        validateBody(system.body);
        addConstraint(system, input);
        addGravity(system, input);
        addElectromagnetism(system, input);
        State state;
        state.r = input.getVec3("position");
        state.v = input.getVec3("velocity");
        state.Omega = input.getVec3("omega");
        const Eigen::Vector4d quaternion = input.getVec4("quaternion");
        state.q = quat(quaternion(0), quaternion(1), quaternion(2), quaternion(3));
        if (state.q.norm() < 1e-14){
            throw std::runtime_error(
                "Initial quaternion cannot be zero."
            );
        }
        state.q.normalize();
        const SolverMode solverMode = parseSolverMode(input.getString("solverMode"));
        Solver solver(system, solverMode);
        RKMK4 integrator;
        const double dt = input.getDouble("dt");
        const double tEnd = input.getDouble("tEnd");
        if(!(dt > 0.0)){
            throw std::runtime_error(
                "Time step must be positive."
            );
        }
        if(!(tEnd >= 0.0)){
            throw std::runtime_error(
                "End time cannot be negative."
            );
        }
        solver.projectVelocity(state, 0.0);
        OutputWriter output("output.csv");
        output.writeHeader();
        double t = 0.0;
        output.write(t, state, solver.diagnostics(state, t));
        while(t < tEnd){
            const double stepSize = std::min(dt, tEnd - t);
            state = integrator.step(state, solver, t, stepSize);
            t += stepSize;
            if (!state.r.allFinite() || !state.v.allFinite() || !state.Omega.allFinite() || !state.q.coeffs().allFinite()){
                throw std::runtime_error(
                    "Non-finite state encountered."
                );
            }
            output.write(t, state, solver.diagnostics(state, t));
        }
        std::cout
            << "Simulation completed.\n"
            << "Final constraint residual: "
            << solver.constraintResidual(state, t)
            << '\n';
        return 0;
    }
    catch (const std::exception& exception){
        std::cerr
            << "Simulation failed: "
            << exception.what()
            << '\n';
        return 1;
    }
}