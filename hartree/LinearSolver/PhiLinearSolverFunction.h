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

#ifndef TUCKER_TENSOR_KSDFT_PHILINEARSOLVERFUNCTION_H
#define TUCKER_TENSOR_KSDFT_PHILINEARSOLVERFUNCTION_H

#include <array>
#include "LinearSolverFunction.h"
#include "../../fem/FEM.h"
#include "../../tensor/Tensor3DMPI.h"
#include "../../tensor/Tensor3DMPIMap.h"
#include "../BoundaryValuesContainer.h"

class PhiLinearSolverFunction : public LinearSolverFunction {
public:
    PhiLinearSolverFunction(const FEM &fem_x,
                            const FEM &fem_y,
                            const FEM &fem_z,
                            Tensor3DMPI &rho,
                            Tensor3DMPIMap &rho_index_map,
                            BoundaryValuesContainer *boundary_values);

    void InitializeSolution(int num_local_entries,
                            const double *initialize_data) override;

    void ComputeA() override;

    void ComputeRhs() override;

private:
    Tensor3DMPI &rho;
    Tensor3DMPIMap &rho_index_map;
    BoundaryValuesContainer *boundary_values;
    const FEM &fem_x, &fem_y, &fem_z;
    const std::vector<std::vector<double>> &nuclei;
    std::vector<PetscInt> diagonal_non_zeros, offdiagonal_non_zeros;
    int number_total_nodes_x, number_total_nodes_y, number_total_nodes_z;

    void ComputeNonZeroPatterns(const FEM &fem_x,
                                const FEM &fem_y,
                                const FEM &fem_z,
                                const std::array<int, 6> &rho_index,
                                std::vector<PetscInt> &diagonal_non_zeros,
                                std::vector<PetscInt> &offdiagonal_non_zeros);

    bool isOnDirichletBoundary(int i,
                               int j,
                               int k);

    void ComputeNINJWithoutBoundaryNodes(Mat &NINJ);

    void ComputedNIdNJForBoundaryNodes(Mat &dNIdNJ);
};

#endif //TUCKER_TENSOR_KSDFT_PHILINEARSOLVERFUNCTION_H
