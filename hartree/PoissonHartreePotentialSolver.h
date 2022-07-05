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

#ifndef TUCKER_TENSOR_KSDFT_POISSONHARTREEPOTENTIALSOLVER_H
#define TUCKER_TENSOR_KSDFT_POISSONHARTREEPOTENTIALSOLVER_H

#include "../fem/FEM.h"
#include "../tensor/Tensor3DMPIMap.h"
#include "PhiBoundaryValuesContainer.h"
#include "LinearSolver/PETScLinearSolver.h"

class PoissonHartreePotentialSolver {
public:
    PoissonHartreePotentialSolver(const FEM &fem_inner_x,
                                  const FEM &fem_inner_y,
                                  const FEM &fem_inner_z,
                                  const FEM &fem_inner_electro_x,
                                  const FEM &fem_inner_electro_y,
                                  const FEM &fem_inner_electro_z,
                                  PETScLinearSolver::Solver solver_type,
                                  PETScLinearSolver::Preconditioner preconditioner_type,
                                  const std::vector<std::vector<double>> &nuclei,
                                  const int rho_decompose_rank_x,
                                  const int rho_decompose_rank_y,
                                  const int rho_decompose_rank_z,
                                  const std::string &ig_alpha_filename,
                                  const std::string &ig_omega_filename,
                                  const double Asquare,
                                  const double fem_outer_domain_x_start,
                                  const double fem_outer_domain_x_end,
                                  const double fem_outer_domain_y_start,
                                  const double fem_outer_domain_y_end,
                                  const double fem_outer_domain_z_start,
                                  const double fem_outer_domain_z_end,
                                  const double coarsing_ratio,
                                  const int num_additional_elements);

    bool solve(const int maxIter,
               const double tolerance,
               Tensor3DMPI &rho_node,
               Tensor3DMPI &hartree_quad);

    void turn_on_initialize_hartree();

protected:
    std::shared_ptr<FEM> initialize_poisson_fem(const FEM &fem,
                                                const double poisson_outer_domain_start,
                                                const double poisson_outer_domain_end,
                                                int num_additional_elements,
                                                double coarsing_ratio);

    void compute_kernel_expansion_values(const FEM &fem_electro_x,
                                         const FEM &fem_electro_y,
                                         const FEM &fem_electro_z,
                                         const FEM &fem_hartree_x,
                                         const FEM &fem_hartree_y,
                                         const FEM &fem_hartree_z,
                                         const std::vector<double> &alpha,
                                         const std::vector<double> &omega,
                                         const double &Asquare,
                                         const int rho_decompose_rank_x,
                                         const int rho_decompose_rank_y,
                                         const int rho_decompose_rank_z,
                                         Tensor3DMPI &rho,
                                         Tensor3DMPI &hartree);

    int rho_decompose_rank_x, rho_decompose_rank_y, rho_decompose_rank_z;
    const std::vector<std::vector<double>> &nuclei;
    std::vector<double> alpha, omega;
    double Asquare;
    bool is_initialize_hartree;
    PETScLinearSolver::Solver solver_type;
    PETScLinearSolver::Preconditioner preconditioner_type;
    std::shared_ptr<FEM> fem_hartree_x, fem_hartree_y, fem_hartree_z;
    const FEM &fem_inner_x, &fem_inner_y, &fem_inner_z;
    const FEM &fem_inner_electro_x, &fem_inner_electro_y, &fem_inner_electro_z;
    std::shared_ptr<Tensor3DMPI> hartree_nodal;
    std::shared_ptr<PhiBoundaryValuesContainer> phi_boundary_values;
    std::shared_ptr<Tensor3DMPIMap> hartreeNodalMap;

};

#endif //TUCKER_TENSOR_KSDFT_POISSONHARTREEPOTENTIALSOLVER_H
