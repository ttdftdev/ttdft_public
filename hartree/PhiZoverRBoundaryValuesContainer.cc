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

#include "PhiZoverRBoundaryValuesContainer.h"

PhiZoverRBoundaryValuesContainer::PhiZoverRBoundaryValuesContainer(const std::array<int, 6> &owned_index,
                                                                   const std::vector<std::vector<double>> &nuclei,
                                                                   const FEM &fem_x,
                                                                   const FEM &fem_y,
                                                                   const FEM &fem_z)
        : PhiBoundaryValuesContainer(owned_index,
                                     fem_x,
                                     fem_y,
                                     fem_z),
          nuclei(nuclei) {
    ComputeBoundaryIndices();
}

void PhiZoverRBoundaryValuesContainer::ComputeBoundaryValues() {
    if (local_boundary_values_.size() != 0) {
        // do nothing
    } else {
        local_boundary_values_ = std::vector<double>(boundary_values_cartesian_global_index_.size(),
                                                     0.0);
        const std::vector<double> &nodes_x = fem_x_.getGlobalNodalCoord();
        const std::vector<double> &nodes_y = fem_y_.getGlobalNodalCoord();
        const std::vector<double> &nodes_z = fem_z_.getGlobalNodalCoord();
        for (int i = 0; i < local_boundary_values_.size(); ++i) {
            double temp_boundary_values = 0.0;
            Cartesian<unsigned> &global_index = boundary_values_cartesian_global_index_[i];
            for (int n = 0; n < nuclei.size(); ++n) {
                double rx = nuclei[n][1] - nodes_x[global_index.x];
                double ry = nuclei[n][2] - nodes_y[global_index.y];
                double rz = nuclei[n][3] - nodes_z[global_index.z];
                double r = std::sqrt(rx * rx + ry * ry + rz * rz);
                temp_boundary_values += nuclei[n][0] / r;
            } // end of loop over nuclei
            local_boundary_values_[i] = temp_boundary_values;
        }
    }
}
