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

#ifndef TUCKER_TENSOR_KSDFT_LINEARSOLVER_H
#define TUCKER_TENSOR_KSDFT_LINEARSOLVER_H

class LinearSolverFunction;

class LinearSolver {

public:
    enum ReturnValueType {
        SUCCESS = 0, FAILURE, MAX_ITER_REACHED
    };

public:
    virtual ReturnValueType solve(LinearSolverFunction *function) = 0;

protected:
    LinearSolver(const int max_number_iterations,
                 const double tolerances);

private:
    // copy/assignment opreators are not implemented
    LinearSolver(const LinearSolver &); // not implemented
    LinearSolver &operator=(const LinearSolver &); // not implemented


public:
    const int getMax_number_iterations() const;

    const double getTolerances() const;

protected:
    const int max_number_iterations;
    const double tolerances;
};

#endif //TUCKER_TENSOR_KSDFT_LINEARSOLVER_H
