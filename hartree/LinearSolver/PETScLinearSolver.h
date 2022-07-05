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

#ifndef TUCKER_TENSOR_KSDFT_PETSCLINEARSOLVER_H
#define TUCKER_TENSOR_KSDFT_PETSCLINEARSOLVER_H

#include <vector>
#include "LinearSolver.h"

class PETScLinearSolver : public LinearSolver {
public:
    typedef enum {
        BCGS = 0, CG, GMRES
    } Solver;
    typedef enum {
        JACOBI = 0, BJACOBI
    } Preconditioner;

    PETScLinearSolver(const int max_number_iterations,
                      const double tolerances,
                      Solver solver,
                      Preconditioner preconditioner);

    ReturnValueType solve(LinearSolverFunction *function) override;

private:
    PETScLinearSolver(const PETScLinearSolver &); // not implemented
    PETScLinearSolver &operator=(const PETScLinearSolver &); // not implemented

// private:
    Solver solver;
    Preconditioner preconditioner;
};

#endif //TUCKER_TENSOR_KSDFT_PETSCLINEARSOLVER_H
