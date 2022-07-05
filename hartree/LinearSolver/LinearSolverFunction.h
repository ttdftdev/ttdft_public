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

#ifndef TUCKER_TENSOR_KSDFT_LINEARSOLVERFUNCTION_H
#define TUCKER_TENSOR_KSDFT_LINEARSOLVERFUNCTION_H

#include <petscmat.h>
#include <vector>

class LinearSolverFunction {
public:
    Mat &getA() { return A; }

    Vec &getRhs() { return rhs; }

    const std::vector<double> &getSolution() { return solution; }

    void CopySolutionFromPetsc(Vec &petsc_solution);

    void CopySolutionToPetsc(Vec &petsc_solution);

    virtual void InitializeSolution(int num_local_entries,
                                    const double *initialize_data) = 0;

    virtual void ComputeA() = 0;

    virtual void ComputeRhs() = 0;

    virtual ~LinearSolverFunction();

protected:
    Mat A;
    Vec rhs;
    std::vector<double> solution;
};

#endif //TUCKER_TENSOR_KSDFT_LINEARSOLVERFUNCTION_H
