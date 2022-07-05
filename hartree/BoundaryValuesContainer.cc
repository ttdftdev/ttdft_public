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

#include "BoundaryValuesContainer.h"

BoundaryValuesContainer::BoundaryValuesContainer(const std::array<int, 6> &owned_index,
                                                 const FEM &fem_x,
                                                 const FEM &fem_y,
                                                 const FEM &fem_z)
        : owned_index(owned_index),
          fem_x_(fem_x),
          fem_y_(fem_y),
          fem_z_(fem_z),
          number_total_nodes_x_(fem_x.getTotalNumberNodes()),
          number_total_nodes_y_(fem_y.getTotalNumberNodes()),
          number_total_nodes_z_(fem_z.getTotalNumberNodes()) {
}

const std::vector<double> &BoundaryValuesContainer::GetLocalBoundaryValues() const {
    return local_boundary_values_;
}

const std::vector<unsigned> &BoundaryValuesContainer::GetBoundaryValuesGlobalIndex() const {
    return boundary_values_global_index_;
}

const std::vector<unsigned> &BoundaryValuesContainer::GetBoundaryValuesLocalIndex() const {
    return boundary_values_local_index_;
}

const std::vector<Cartesian<unsigned>> &BoundaryValuesContainer::GetBoundaryValuesCartesianGlobalIndex() const {
    return boundary_values_cartesian_global_index_;
}

void BoundaryValuesContainer::ComputeBoundaryIndices() {
    int local_idx_increment = 0;
    for (int k = owned_index[4]; k != owned_index[5]; ++k) {
        for (int j = owned_index[2]; j != owned_index[3]; ++j) {
            for (int i = owned_index[0]; i != owned_index[1]; ++i) {
                if (CheckIsOnDirichletBoundary(i,
                                               j,
                                               k)) {

                    boundary_values_cartesian_global_index_.emplace_back(Cartesian<unsigned>(i,
                                                                                             j,
                                                                                             k));

                    unsigned global_index =
                            i + j * number_total_nodes_x_ + k * number_total_nodes_x_ * number_total_nodes_y_;
                    boundary_values_global_index_.emplace_back(global_index);

                    boundary_values_local_index_.emplace_back(local_idx_increment);
                } // end of if
                local_idx_increment++;
            }
        }
    }
}

bool BoundaryValuesContainer::CheckIsOnDirichletBoundary(int i,
                                                         int j,
                                                         int k) {
    return ((i == 0) || (i == (number_total_nodes_x_ - 1)
                         || (j == 0) || (j == (number_total_nodes_y_ - 1)) || (k == 0)
                         || (k == (number_total_nodes_z_ - 1))));
}

