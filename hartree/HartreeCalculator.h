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

#ifndef TUCKER_TENSOR_KSDFT_HARTREECALCULATOR_H
#define TUCKER_TENSOR_KSDFT_HARTREECALCULATOR_H

#include "../fem/FEM.h"
#include "../tensor/Tensor3DMPI.h"
#include "../tensor/Tensor3DMPIMap.h"
#include "BoundaryValuesContainer.h"
#include "LinearSolver/PETScLinearSolver.h"

namespace HartreeCalculator {
    void CalculateHartreePotentialOnQuadPoints(const FEM &femX,
                                               const FEM &femY,
                                               const FEM &femZ,
                                               const int maxIter,
                                               const double tolerance,
                                               Tensor3DMPI &rho_node,
                                               Tensor3DMPIMap &hartree_index_map,
                                               Tensor3DMPI &hartree_node,
                                               BoundaryValuesContainer *phi_boundary_value,
                                               Tensor3DMPI &hartree_quad);

    void CalculateHartreePotentialOnQuadPointsUsingLargerDomain(const FEM &ks_fem_x,
                                                                const FEM &ks_fem_y,
                                                                const FEM &ks_fem_z,
                                                                const FEM &po_fem_x,
                                                                const FEM &po_fem_y,
                                                                const FEM &po_fem_z,
                                                                const int maxIter,
                                                                const double tolerance,
                                                                PETScLinearSolver::Solver &solver_type,
                                                                PETScLinearSolver::Preconditioner &preconditioner_type,
                                                                Tensor3DMPI &rho_node_ks_domain,
                                                                Tensor3DMPIMap &hartree_index_map_po_domain,
                                                                Tensor3DMPI &hartree_node_po_domain,
                                                                BoundaryValuesContainer *phi_boundary_values,
                                                                Tensor3DMPI &hartreeQuad);
/*void ComputeHartreeQuadFromRhoUsingLargerDomainWithZoverR(const FEM &femX, const FEM &femY, const FEM &femZ,
                                                          int number_elements_outer_domain, double coarsng_factor,
                                                          const std::vector<std::vector<double> > &nuclei,
                                                          Tensor3DMPI &hartreeInitialGuess,
                                                          const int maxIter,
                                                          const double tolerance,
                                                          Tensor3DMPI &rho, Tensor3DMPI &hartreeQuad);*/
}

#endif //TUCKER_TENSOR_KSDFT_HARTREECALCULATOR_H
