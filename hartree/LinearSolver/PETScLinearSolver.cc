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

#include <petscksp.h>
#include "PETScLinearSolver.h"
#include "LinearSolverFunction.h"

PETScLinearSolver::PETScLinearSolver(const int max_number_iterations,
                                     const double tolerances,
                                     Solver solver,
                                     Preconditioner preconditioner) :
        LinearSolver(max_number_iterations,
                     tolerances),
        solver(solver),
        preconditioner(preconditioner) {
}

LinearSolver::ReturnValueType PETScLinearSolver::solve(LinearSolverFunction *function) {
    KSP
            ksp;
    KSPCreate(PETSC_COMM_WORLD,
              &ksp);
    Mat &A = function->getA();
    KSPSetOperators(ksp,
                    A,
                    A);
    KSPSetTolerances(ksp,
                     tolerances,
                     PETSC_NULL,
                     PETSC_DEFAULT,
                     max_number_iterations);

    if (solver == BCGS) {
        KSPSetType(ksp,
                   KSPBCGS);
    } else if (solver == CG) {
        KSPSetType(ksp,
                   KSPCG);
    } else if (solver == GMRES) {
        KSPSetType(ksp,
                   KSPGMRES);
    }

    PC
            pc;
    KSPGetPC(ksp,
             &pc);
    if (preconditioner == JACOBI) {
        PCSetType(pc,
                  PCJACOBI);
    } else if (preconditioner == BJACOBI) {
        PCSetType(pc,
                  PCBJACOBI);
    }

    KSPSetFromOptions(ksp);

    PCType pctype;
    PCGetType(pc,
              &pctype);
    KSPType kspType;
    KSPGetType(ksp,
               &kspType);
    PetscReal rtol, atol;
    PetscInt
            maxit;
    KSPGetTolerances(ksp,
                     &rtol,
                     &atol,
                     PETSC_NULL,
                     &maxit);
    PetscPrintf(PETSC_COMM_WORLD,
                "ksp rtol: %e\n",
                rtol);
    PetscPrintf(PETSC_COMM_WORLD,
                "ksp atol: %e\n",
                atol);
    PetscPrintf(PETSC_COMM_WORLD,
                "ksp max iters: %d\n",
                maxit);
    PetscPrintf(PETSC_COMM_WORLD,
                "ksptype: ");
    PetscPrintf(PETSC_COMM_WORLD,
                kspType);
    PetscPrintf(PETSC_COMM_WORLD,
                "\n");
    PetscPrintf(PETSC_COMM_WORLD,
                "pytype: ");
    PetscPrintf(PETSC_COMM_WORLD,
                pctype);
    PetscPrintf(PETSC_COMM_WORLD,
                "\n");

    KSPSetInitialGuessNonzero(ksp,
                              PETSC_TRUE);

    Vec
            solution;
    MatCreateVecs(A,
                  &solution,
                  PETSC_NULL);
    function->CopySolutionToPetsc(solution);
    KSPSolve(ksp,
             function->getRhs(),
             solution);

    KSPConvergedReason reason;
    KSPGetConvergedReason(ksp,
                          &reason);
    PetscPrintf(PETSC_COMM_WORLD,
                "reason: %i\n",
                reason);
    PetscInt
            iternum;
    KSPGetIterationNumber(ksp,
                          &iternum);
    PetscPrintf(PETSC_COMM_WORLD,
                "iterations: %i\n",
                iternum);

    function->CopySolutionFromPetsc(solution);

    KSPDestroy(&ksp);

    VecDestroy(&solution);
    return SUCCESS;
}
