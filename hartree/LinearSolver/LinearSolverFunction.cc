/******************************************************************************
 * Copyright (c) 2020-2021.                                                   *
 * The Regents of the University of Michigan and TTDFT authors.               *
 *                                                                            *
 * This file is part of the TTDFT code.                                       *
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

#include "LinearSolverFunction.h"

void LinearSolverFunction::CopySolutionFromPetsc(Vec &petsc_solution) {
    double *solution_array;
    PetscInt solution_local_size;
    VecGetLocalSize(petsc_solution,
                    &solution_local_size);
    VecGetArray(petsc_solution,
                &solution_array);
    for (PetscInt i = 0; i < solution_local_size; ++i) {
        solution[i] = solution_array[i];
    }
    VecRestoreArray(petsc_solution,
                    &solution_array);
}

void LinearSolverFunction::CopySolutionToPetsc(Vec &petsc_solution) {
    double *solution_array;
    PetscInt solution_local_size;
    VecGetLocalSize(petsc_solution,
                    &solution_local_size);
    VecGetArray(petsc_solution,
                &solution_array);
    for (PetscInt i = 0; i < solution_local_size; ++i) {
        solution_array[i] = solution[i];
    }
    VecRestoreArray(petsc_solution,
                    &solution_array);
    VecAssemblyBegin(petsc_solution);
    VecAssemblyEnd(petsc_solution);
}

LinearSolverFunction::~LinearSolverFunction() {
    MatDestroy(&A);
    VecDestroy(&rhs);
}
