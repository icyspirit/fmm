# SimpleFMM

A simple Fast Multipole Method (FMM) implementation using MPI shared memory and OpenMP SIMD vectorization.  
This is a header-only library and can be easily included in any C++ project.

## Dependencies and Compilation

- C++17
- MPI
- OpenMP

To compile:
```bash
make CC=$YOUR_OWN_CXX_COMPILER
./a.out               # or: mpirun -n $NP ./a.out
```

## Usage
See `main.cpp` for an example of how to use the library.

## License
This project is licensed under the MIT License.
