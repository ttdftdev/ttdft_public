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

#ifndef TUCKER_TENSOR_KSDFT_PHIZOVERRBOUNDARYVALUESCONTAINER_H
#define TUCKER_TENSOR_KSDFT_PHIZOVERRBOUNDARYVALUESCONTAINER_H

#include "PhiBoundaryValuesContainer.h"

class PhiZoverRBoundaryValuesContainer : public PhiBoundaryValuesContainer {
public:
    PhiZoverRBoundaryValuesContainer(const std::array<int, 6> &owned_index,
                                     const std::vector<std::vector<double>> &nuclei,
                                     const FEM &fem_x,
                                     const FEM &fem_y,
                                     const FEM &fem_z);

    void ComputeBoundaryValues() override;

protected:
    const std::vector<std::vector<double>> &nuclei;

};

#endif //TUCKER_TENSOR_KSDFT_PHIZOVERRBOUNDARYVALUESCONTAINER_H
