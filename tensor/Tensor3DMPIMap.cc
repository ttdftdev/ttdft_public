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

#include "Tensor3DMPIMap.h"

Tensor3DMPIMap::Tensor3DMPIMap(const FEM &fem_x,
                               const FEM &fem_y,
                               const FEM &fem_z,
                               Tensor3DMPI &tensor) {
    unsigned nubmer_total_nodes_x = fem_x.getTotalNumberNodes();
    unsigned nubmer_total_nodes_y = fem_y.getTotalNumberNodes();
    unsigned nubmer_total_nodes_z = fem_z.getTotalNumberNodes();

    std::array<int, 6> tensor_idx = tensor.getGlobalIndex();
    for (unsigned k = tensor_idx[4]; k < tensor_idx[5]; ++k) {
        for (unsigned j = tensor_idx[2]; j < tensor_idx[3]; ++j) {
            for (unsigned i = tensor_idx[0]; i < tensor_idx[1]; ++i) {
                int global_index = i + j * nubmer_total_nodes_x + k * nubmer_total_nodes_x * nubmer_total_nodes_y;
                local_tensor_global_index_.emplace_back(global_index);
                local_tensor_global_carteisian_index_.emplace_back(Cartesian<unsigned>(i,
                                                                                       j,
                                                                                       k));
            }
        }
    }
}

const std::vector<unsigned int> &Tensor3DMPIMap::GetLocalTensorGlobalIndex() const {
    return local_tensor_global_index_;
}

const std::vector<Cartesian<unsigned int>> &Tensor3DMPIMap::GetLocalTensorGlobalCarteisianIndex() const {
    return local_tensor_global_carteisian_index_;
}
