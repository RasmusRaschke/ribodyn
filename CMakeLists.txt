cmake_minimum_required(VERSION 3.15)

project(NonholonomicSolver)

set(CMAKE_CXX_STANDARD 20)

find_package(Eigen3 REQUIRED)

include_directories(include)

add_executable(
    solver

    src/main.cpp
    src/io.cpp
    src/utility.cpp
    src/solver.cpp
    src/integrator.cpp
    src/emField.cpp

    systems/rollingConstraint.cpp
    systems/uniformGravity.cpp
    systems/centralGravity.cpp
    systems/uniformEMField.cpp
    systems/rotatingUniformEMField.cpp
    systems/sphereAirResistance.cpp
    systems/regularizedCoulombRollingResistance.cpp
)

target_link_libraries(
    solver
    Eigen3::Eigen
)