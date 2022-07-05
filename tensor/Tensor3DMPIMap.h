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

#ifndef TUCKER_TENSOR_KSDFT_TENSOR3DMPIMAP_H
#define TUCKER_TENSOR_KSDFT_TENSOR3DMPIMAP_H

#include <vector>
#include "../utils/Cartesian.h"
#include "Tensor3DMPI.h"
#include "../fem/FEM.h"

class Tensor3DMPIMap {
public:
    Tensor3DMPIMap(const FEM &fem_x,
                   const FEM &fem_y,
                   const FEM &fem_z,
                   Tensor3DMPI &tensor);

    const std::vector<unsigned int> &GetLocalTensorGlobalIndex() const;

    const std::vector<Cartesian<unsigned int>> &GetLocalTensorGlobalCarteisianIndex() const;

protected:
    std::vector<unsigned> local_tensor_global_index_;
    std::vector<Cartesian<unsigned>> local_tensor_global_carteisian_index_;
};

#endif //TUCKER_TENSOR_KSDFT_TENSOR3DMPIMAP_H
