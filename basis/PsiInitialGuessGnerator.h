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

#ifndef TUCKER_TENSOR_KSDFT_PSIINITIALGUESSGNERATOR_H
#define TUCKER_TENSOR_KSDFT_PSIINITIALGUESSGNERATOR_H

#include <map>
#include "petscmat.h"
#include "../alglib/src/interpolation.h"
#include "../tensor/Tensor3DMPI.h"
#include "../fem/FEM.h"
#include "../tensor/Tensor3DMPIMap.h"

struct Orbital {
    unsigned int atom_id;
    unsigned int Z, n, l;
    int m;
    alglib::spline1dinterpolant *psi;
};

class PsiInitialGuessGnerator {
public:
    PsiInitialGuessGnerator(const FEM &fem_x,
                            const FEM &fem_y,
                            const FEM &fem_z,
                            const int field_decomposition_rank_x,
                            const int field_decomposition_rank_y,
                            const int field_decomposition_rank_z,
                            const std::vector<std::vector<double>> &atom_information,
                            unsigned int number_eigen_values);

    void ComputeInitialGuessPSI(const std::vector<std::vector<double>> &basis_x,
                                const std::vector<std::vector<double>> &basis_y,
                                const std::vector<std::vector<double>> &basis_z,
                                Mat &eig_vecs);

    virtual ~PsiInitialGuessGnerator();

private:
    void ComputeOwnedEigenvectors();

    void LoadPSIFIles(unsigned int Z,
                      unsigned int n,
                      unsigned int l,
                      unsigned int &file_read_flag);

    void DetermineOrbitalFilling();

    void ComputePSI(Orbital &orbital,
                    Tucker::Tensor *psi);

    void ComputeFieldTuckerDecompositionOnQuad(Tucker::Tensor *quad_field,
                                               Tucker::Tensor *seq_core_tensor,
                                               std::vector<std::vector<double>> &Ux_quad,
                                               std::vector<std::vector<double>> &Uy_quad,
                                               std::vector<std::vector<double>> &Uz_quad);

    void ComputeTuckerBasisOnQuad(const std::vector<std::vector<double>> &basis_x,
                                  const std::vector<std::vector<double>> &basis_y,
                                  const std::vector<std::vector<double>> &basis_z,
                                  std::vector<std::vector<double>> &basis_quad_x,
                                  std::vector<std::vector<double>> &basis_quad_y,
                                  std::vector<std::vector<double>> &basis_quad_z);

    void ProjectFieldOntoTuckerBasis(Tucker::Tensor *seq_core_tensor,
                                     const std::vector<std::vector<double>> &decomposed_field_quad_ux,
                                     const std::vector<std::vector<double>> &decomposed_field_quad_uy,
                                     const std::vector<std::vector<double>> &decomposed_field_quad_uz,
                                     const std::vector<std::vector<double>> &basis_quad_x,
                                     const std::vector<std::vector<double>> &basis_quad_y,
                                     const std::vector<std::vector<double>> &basis_quad_z,
                                     std::vector<double> &projection_coefficients);

    const FEM &fem_x, &fem_y, &fem_z;
    const std::vector<std::vector<double>> &atom_information;
    std::vector<Orbital> wave_functions_vector;
    unsigned owned_eigenvectors_start, owned_eigenvectors_end;
    unsigned numeber_eigen_values;
    Tucker::SizeArray field_decomposition_rank;
    std::map<unsigned int, std::map<unsigned int, std::map<unsigned int, alglib::spline1dinterpolant *> > > radial_values;
    std::map<unsigned int, std::map<unsigned int, std::map<unsigned int, double> > > outerValues;
};

#endif //TUCKER_TENSOR_KSDFT_PSIINITIALGUESSGNERATOR_H
