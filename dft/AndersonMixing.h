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

#ifndef TUCKER_TENSOR_KSDFT_ANDERSONMIXING_H
#define TUCKER_TENSOR_KSDFT_ANDERSONMIXING_H

#include <TuckerMPI_Tensor.hpp>
#include <deque>
#include "../tensor/Tensor3DMPI.h"
#include "../fem/FEM.h"

class AndersonMixing {
public:

    AndersonMixing(const FEM &femX,
                   const FEM &femY,
                   const FEM &femZ,
                   const Tensor3DMPI &jacob3DMat,
                   const Tensor3DMPI &weight3DMat,
                   const std::string &rho_path,
                   const double alpha,
                   const int maxIterHistory = 40,
                   bool restart = false,
                   bool keep_older_history = false);

    void computeMixingConstants(const std::deque<Tensor3DMPI> &vectorRhoIn,
                                const std::deque<Tensor3DMPI> &vectorRhoOut,
                                std::vector<double> &mixingConstants);

    void computeRhoIn(const int scfIter,
                      Tensor3DMPI &rhoNodalIn,
                      Tensor3DMPI &rhoGridIn);

    void updateRho(Tensor3DMPI &rhoNodalIn,
                   Tensor3DMPI &rhoGridIn,
                   Tensor3DMPI &rhoNodalOut,
                   Tensor3DMPI &rhoGridOut,
                   int scfIter);

    void clearHistory(int scf_iter);

private:
    const FEM &femX;
    const FEM &femY;
    const FEM &femZ;
    std::string rho_path, rho_nodal_in_folder, rho_nodal_out_folder, rho_quad_in_folder, rho_quad_out_folder;
    const int maxIterHistory;
    const double alpha;
    bool keep_older_history;
    int history_offset; // used when the history is cleaned up
    std::deque<Tensor3DMPI> vectorGridRhoIn, vectorGridRhoOut, vectorNodalRhoIn, vectorNodalRhoOut;
    std::string rho_nodal_in_prefix = "rho_nodal_in";
    std::string rho_nodal_out_prefix = "rho_nodal_out";
    std::string rho_quad_in_prefix = "rho_quad_in";
    std::string rho_quad_out_prefix = "rho_quad_out";

    const Tensor3DMPI &jacob3DMat;
    const Tensor3DMPI &weight3DMat;

    void computeRhoIn(const std::deque<Tensor3DMPI> &vectorRhoIn,
                      const std::deque<Tensor3DMPI> &vectorRhoOut,
                      const std::vector<double> &mixingConstants,
                      Tensor3DMPI &rhoIn);

    void print_rho(const std::string &path,
                   const std::string &file_name,
                   Tensor3DMPI &tensor);

    void read_rho(const std::string &path,
                  const std::string &file_name,
                  Tensor3DMPI &tensor);
};

#endif //TUCKER_TENSOR_KSDFT_ANDERSONMIXING_H
