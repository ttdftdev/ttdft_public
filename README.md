# TTDFT: Tucker-tensor density functional theory calculation routine

## About
TTDFT is a C++ code with GPU acceleration for real-space density functional calculation developed by the [Computational Materials Physics Group](http://www-personal.umich.edu/~vikramg/) at the University of Michigan - Ann Arbor. The routine is based on tensor-structured basis and L-1 localization techniques. Due to the construction of an efficient tensor-structured basis, TTDFT is capable of providing reduced-order scaling with system size, hence enabling DFT calculations on large-scale systems.

## License
[GNU Lesser General Public License v3.0 or later](https://www.gnu.org/licenses/lgpl-3.0-standalone.html)  
[![LGPL 3 or later](https://www.gnu.org/graphics/lgplv3-88x31.png)](https://www.gnu.org/licenses/lgpl-3.0-standalone.html)
```c++
/******************************************************************************
 * Copyright (c) 2020-2021.                                                   *
 * The Regents of the University of Michigan and TTDFT authors.               *
 *                                                                            *
 * TTDFT is free software: you can redistribute it and/or modify              *
 *  it under the terms of the Lesser GNU General Public License as            *
 *  published by the Free Software Foundation, either version 3 of            *
 *  the License, or (at your option) any later version.                       *
 *                                                                            *
 *  TTDFT is distributed in the hope that it will be useful, but              *
 *  WITHOUT ANY WARRANTY; without even the implied warranty                   *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                      *
 *  See the Lesser GNU General Public License for more details.               *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public           *
 * License at the top level of TTDFT distribution.  If not, see               *
 * <https://www.gnu.org/licenses/>.                                           *
 ******************************************************************************/
 ```

## Dependencies
The libraries listed below are used for this project. We thank the efforts from these developers to make this project possible. The links to the library sites are provided here and a brief installation guide is described in the later part of this README. All packages are listed in the alphabetical order.
1. ALGLIB (http://www.alglib.net/)
2. BLAS (http://www.netlib.org/blas/)
3. Boost (https://www.boost.org/)
4. CUDA (https://docs.nvidia.com/cuda/index.html)
5. LAPACK (http://www.netlib.org/lapack/)
6. PETSc (https://www.mcs.anl.gov/petsc)
7. SLEPc (http://slepc.upv.es)
8. TuckerMPI (https://gitlab.com/tensors/TuckerMPI)

## Directory structure of TTDFT
* **alglib/**  
  The alglib library should be copied to here as a whole for a successful compilation.  
* **atoms/**  
  This folder contains the container of the read-in atomic information (spatial coordinates, atomic numbers, etc.) and non-local PSP information for the atoms.
* **basis/**  
  This folder contains the reader to read-in the given 1-D localized functions for constructing the 3-D localized Tucker tensor basis.
* **blas_lapack/**  
  This folder contains the wrapper for blas/lapack declared in the namespace `clinalg`.
* **dft/**  
  This folder contains files related to the Kohn-Sham DFT calculations. The files include the computation of projected Hamltonian matrix, electron density, and ground state energy.
* **eigensolver/**  
  The eigensolver for diagonalizing the projected Kohn-Sham Hamiltonian. Chebyshev filtering subspace iteration method is used in this project.
* **hartree/**  
  This folder contains files for computing the Hartree potential using kernel expansion based on tensor-structured techniques and a backup option of computing Hartree potential using Poisson solver (not optimized and not recommended). 
* **tensor/**  
  This folder contains a wrapper to the Tucker MPI library used in this project. `Tucker3DMPI` is the class which encapsulates the data in tensor structure, `Tensor3DMPIMap` stores the communication information for local and global data of tensors. `TensorUtils` contains necessary functions in this work for tensor operations. `TuckerTensor` is an adapter class to the TuckerTensor in TuckerMPI.
* **fem/**  
  This folder contains the necessary information for 1-D finite element and the quadrature infomration for Gauss Integration.

## Installation
### Installing Dependencies
Here we list the dependencies in this code. Items 2~5 are usually installed on most supercomputers. If these are not pre-installed on your machine, please refer to the webpages of the packages for installation details.
1. ALGLIB (http://www.alglib.net/)
    * The free version of alglib is used for function interpolation. For compilation, please download and extract the source files on the alglib website. Then please rename the extracted folder to `alglib` and move it to the project folder of TTDFT.
1. BLAS (http://www.netlib.org/blas/)
    * BLAS and LAPACK are numerical linear algebra interfaces. High-performance computing clusters usually have highly optimized precompiled BLAS/LAPACK libraries (e.g. MKL for Intel machine, ESSL for IBM machines). We suggest using the most optimal BLAS/LAPACK library available for better performance.
1. Boost (https://www.boost.org/)
    * Boost is a C++ library which extends the features of C++ language. The library is usually preinstalled on most HPC machines and available with `module load boost`.
1. CUDA (https://docs.nvidia.com/cuda/index.html)
    * CUDA is a parallel computing routine for NVIDIA GPU calculation. Particularly, cuBLAS and cuSparse are used in this work. On HPC machines with GPU support, please `module load cuda` for CMake to find the correct dependency. We note that this project is built hetrogeneously using MPI compiler for `.cc` and `.cpp` files and CUDA compiler for `.cu` files. The underlying compiler should thus be consistent for MPI and CUDA compilation. Inconsistent compilers might still compile but could result in disastrous unexpected runtime error.
1. LAPACK (http://www.netlib.org/lapack/)
    * see BLAS.
1. PETSc (https://www.mcs.anl.gov/petsc)
    * Portable Extensible Toolkit for Scientific computation is a package dedicated to perform large scale MPI linear algebra operations for the CPU part of the code. For compiling TTDFT project, PETSc should be installed with Elemental support. We also recommend users to install versions later than v 3.12 for better memory footprint and performance.
1. SLEPc (http://slepc.upv.es)
    * SLEPc is developed based on PETSc and is used for solving eigenvalue problems for the CPU part of the code.
1. TuckerMPI (https://gitlab.com/tensors/TuckerMPI)
    * TuckerMPI provides distributed memory scheme for Tucker tensor decomposition. To install, please refer to the documentation in the GitLab repository. We note that `TUCKER_BUILD_DIR` and `TUCKER_DIR` should be set as environmental variables storing the full path to the source and build directories. Please be cautioned that TuckerMPI library is known to not work with serveral MPI implementations (see the GitLab repository).

### Configure
With the dependencies properly installed on the machine, please load BLAS/LAPACK, CUDA Toolkit, using `module load` on the high-performance computing environment. The please use `cmake` to configure the compilation. We also note that all libraries should be compiled using the same compilers that will be used to compile the TTDFT code to prevent any unexpected compilation/run-time error. The environmental variables `PETSC_DIR`, `PETSC_ARCH`, `SLEPC_DIR`, `TUCKER_DIR`, `TUCKER_BUILD_DIR` have to be set for the libraries to work correctly. Please refer to the pacakge websites in Installing Dependencies for more details. Please also note that `TUCKER_DIR` is where your TuckerMPI headers are located and `TUCKER_BUILD_DIR` is the path to the build directory of Tucker MPI.

### Compile
1. Create a build folder with `mkdir build` within the project folder and change directory to `build`.
2. Execute `cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=mpicxx ../`. `mpicxx` can be replaced by your choice of MPI compilers. Again, please make it consistent with CUDA compiler and the compilers used to build depenedent libraries. Inconsistent compilers might still compile but will result in disastrous unexpected run-time error. We also recommend users to pass-in proper linkers/flags with `-DCMAKE_CXX_FLAGS` to optimize the dependent libraries and the code.
3. Execute `make` in the same folder. The code should then compile.

## Example
Please see the example folder for a quick-start case for a one-shell aluminum nano-particle calculation. We note that the L-1 localized functions are precomputed and provided as `basis_{x, y, z}.txt` using the code developed by V. Ozolins, R. Lai, R. Caflisch, S. Osher, "Compressed Modes for Variational Problems in Mathematics and Physics", PNAS, 110 (46) [[Code](http://homepages.rpi.edu/~lair/codes/CMs_codes_share.zip)]. The parameters for the expansion terms of the 
Hartree potential calculations using tensor-structured techniques can be found at [[here](http://www.mis.mpg.de/scicomp/EXP_SUM/1_sqrtx/tabelle)].

### Running the example  
The necessary files for a complete run can be found in the `example` folder. Before execution, please copy the properly compiled `main` executable to the example folder.

* The files structure of the exmample:
  * `alphak35_1e8` and `omegak35_1e8` are files adapted from [here](http://www.mis.mpg.de/scicomp/EXP_SUM/1_sqrtx/tabelle) for tensor-structured techniques for computing Hartree potential. The file names are changeable in the `input.inp` file.  
  * `basis_{x, y, z}.txt` are the spatial coordinates of the L-1 localized functions.  
  * shell1 contains the atomic information and the atomic spatial coordinates, the format is  
    Line 1: `number of atom types` `number of atoms` `number of electrons`  
    Line 2: `l_max for the pseudopotential`  
    Line 3~the end of the file (for each atom): `number of valence electrons` `atom type` `x-coordinates` `y-coordinates` `z-coordinates` `atomic number` `book-keeping variable`
  * `Density_AT0`, `nlpV_AT0`, `nlpWaveFun_AT0`, `locPotential_AT0` are the files adapted from `.upf` file for aluminum for pseudopotential calculations.
  * With these files ready and properly setup, inside `example`, run the compiled executable using a proper MPI launcher with `input.inp` as the input. For other options, please refer to the comments in `input.inp` for the details. Please also note that the options after `#DANGER ZONE` in the `input.inp` are experimental features and book-keeping parameters for developers only. We do not recommend modifying any of those parameters.

## Authors
Developer
* Ian C. Lin: iancclin@umich.edu

Mentors
* Vikram Gavini, Professor   
  Department of Mechanical Engineering, University of Michigan, Ann Arbor, USA  
  Department of Materials Science and Engineering, University of Michigan, Ann Arbor, USA
* Phani Motamarri, Assistant Professor  
  Department of Computational and Data Sciences, Indian Institute of Science, Bangalore,  India

## Please cite
The code is developed based on a CPU implementation of the reduced-order scaling tensor-structured algorithm for KS-DFT paper. If you use this code, please kindly cite the following papers.
* Lin, CC., Motamarri, P. & Gavini, V. Tensor-structured algorithm for reduced-order scaling large-scale KohnSham 
  density functional theory calculations. npj Comput Mater 7, 50 (2021). https://doi.org/10.1038/s41524-021-00517-5
