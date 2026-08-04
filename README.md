# RIBODYN: RIgid BOdy DYNamics
This repository contains a C++ program to calculate the equations of motion for rigid bodies whose configuration manifold is a submanifold of $\mathbb{R}^3 \times \text{SO}(3)$ given by affine, semi-holonomic constraints. To use the program, you have to clone the repository and build it with ```cmake``` as follows:
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
```
After that, you need to prepare an input file or use one of the examples in [example/input](example/input). The general convention is that vectors are read in from head to tail and matrices are read in from up left to down right. The program is then called as ```./build/solver path/to/input.in```. The output is written as CSV file in the [build](build/) directory. 

## Sources
The program directly implements the theoretical method outlined in 1.3 of [Nonholonomic Mechanics and Control](https://link.springer.com/book/10.1007/978-1-4939-3017-3) by Bloch et al. using a Runge-Kutta-Munthe-Kaas 4 algorithm. The program has two solver modes: Lagrange mode uses conservative potentials to calculate and solve the Euler-Lagrange equations, while d'Alembert mode works with generalized forces directly, hence also allows for easy implementation of non-conservative forces. In general: If you have non-conservative forces, d'Alembert is less of a pain to deal with since not all non-conservative forces can be derived as Rayleigh dissipation. 
If vector forces in your system are tedious to derive, Lagrange mode is better suited.
The program is a major expansion of my previous program magsphere. Generative AI was used to help with generation of header files and error handling.

## Overview
- [src](src/) contains the source files such as the IO handler, the main file, the solver and integrator files, and the mathematical utilities.
- [include](include/) contains the headers of the program, corresponding to the respective cpp files.
- [systems](systems/) contains implementations of physical systems such as forces, potentials, fields, and constraints.
- [example](example/) contains some example applications of the program. The [overview.py](example/overview.py) file can be used to plot pretty much everything quickly, and [maximal_input.in](example/maximal_input.in) shows and explains all possible inputs.

## Implementation of new systems
If you want to implement your own system, you have to create a new file in [systems](systems/) (it is a good idea to use one I created as template). In general, you probably want to implement some field acting on the body, in which case your new field class should probably be a derivative from one of the predefined field classes in [structures.hpp](include/structures.hpp). When you decided on the implementation, add your class to the header [include/systems.hpp](include/systems.hpp) and adapt the input handling in [main.cpp](src/main.cpp), which can be adapted from the handlers in the anonymous namespace at the beginning. The same principle applies for constraints. Do not forget to add the new files to [CMakeLists.txt](./CMakeLists.txt) before rebuilding. It would also be nice if you expanded [maximal_input.in](example/maximal_input.in) to list all your implemented options.

