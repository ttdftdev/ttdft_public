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

#ifndef TUCKERDFTSPARSE_TUCKERBASIS_H
#define TUCKERDFTSPARSE_TUCKERBASIS_H

#include <vector>
#include <memory>
#include <TuckerMPI.hpp>
#include "../fem/FEM.h"
#include "../atoms/NonLocalMapManager.h"
#include "../atoms/NonLocalPSPData.h"
#include "../dft/SeparableHamiltonian.h"

class TuckerBasis {
public:
    TuckerBasis(int rank_x,
                int rank_y,
                int rank_z,
                const FEM &fem_x,
                const FEM &fem_y,
                const FEM &fem_z,
                const FEM &fem_nonloc_x,
                const FEM &fem_nonloc_y,
                const FEM &fem_nonloc_z,
                const TuckerMPI::TuckerTensor *decomposedPotEff,
                std::vector<std::shared_ptr<NonLocalMapManager>> &nonloc_map_manager,
                std::vector<std::shared_ptr<NonLocalPSPData>> &nonloc_psp_data);

    void initialize_ig(Tensor3DMPI &psi);

    std::vector<std::vector<double>> basis_x, basis_y, basis_z;

    virtual void solve_for_basis(double tolerance,
                                 int maxIter,
                                 double alpha,
                                 int number_history,
                                 SeparableSCFType scf_separable_type = SeparableSCFType::ANDERSON);

protected:
    void basis_solver(int rx,
                      int ry,
                      int rz,
                      std::vector<std::vector<double>> &basis_x,
                      std::vector<std::vector<double>> &basis_y,
                      std::vector<std::vector<double>> &basis_z,
                      double tolerance,
                      int maxIter,
                      double alpha,
                      int number_history,
                      SeparableSCFType scf_separable_type = SeparableSCFType::ANDERSON);

    int rank_x, rank_y, rank_z;
    const FEM &fem_x, &fem_y, &fem_z, &fem_nonloc_x, &fem_nonloc_y, &fem_nonloc_z;
    const TuckerMPI::TuckerTensor *decomposedPotEff;
    std::vector<std::shared_ptr<NonLocalMapManager>> &nonloc_map_manager;
    std::vector<std::shared_ptr<NonLocalPSPData>> &nonloc_psp_data;
    std::vector<double> initial_guess_x, initial_guess_y, initial_guess_z;
};

#endif //TUCKERDFTSPARSE_TUCKERBASIS_H
