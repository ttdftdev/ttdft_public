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

#ifndef TUCKER_TENSOR_KSDFT_PHIMULTIPOLEBOUNDARYVALUESCONTAINER_H
#define TUCKER_TENSOR_KSDFT_PHIMULTIPOLEBOUNDARYVALUESCONTAINER_H

#include "../tensor/Tensor3DMPI.h"
#include "PhiBoundaryValuesContainer.h"

class PhiMultiPoleBoundaryValuesContainer : public PhiBoundaryValuesContainer {
public:
    PhiMultiPoleBoundaryValuesContainer(const std::array<int, 6> &owned_index,
                                        const FEM &fem_x,
                                        const FEM &fem_y,
                                        const FEM &fem_z,
                                        const FEM &fem_electro_x,
                                        const FEM &fem_electro_y,
                                        const FEM &fem_electro_z,
                                        std::string omega_file,
                                        std::string alpha_file,
                                        double Asquare,
                                        const unsigned rho_decomposed_rank_x,
                                        const unsigned rho_decomposed_rank_y,
                                        const unsigned rho_decomposed_rank_z);

    void SetRho(Tensor3DMPI *rho);

    void ComputeBoundaryValues() override;

protected:
    Tensor3DMPI *rho_;
    unsigned rho_decomposed_rank_x_, rho_decomposed_rank_y_, rho_decomposed_rank_z_;
    std::vector<double> alpha_;
    std::vector<double> omega_;
    double Asquare_;
    const FEM &fem_electro_x_;
    const FEM &fem_electro_y_;
    const FEM &fem_electro_z_;
};

#endif //TUCKER_TENSOR_KSDFT_PHIMULTIPOLEBOUNDARYVALUESCONTAINER_H
