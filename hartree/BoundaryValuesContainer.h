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

#ifndef TUCKER_TENSOR_KSDFT_BOUNDARYVALUESCONTAINER_H
#define TUCKER_TENSOR_KSDFT_BOUNDARYVALUESCONTAINER_H

#include <vector>
#include <array>
#include "../utils/Cartesian.h"
#include "../fem/FEM.h"

class BoundaryValuesContainer {
public:
    BoundaryValuesContainer(const std::array<int, 6> &owned_index,
                            const FEM &fem_x,
                            const FEM &fem_y,
                            const FEM &fem_z);

    const std::vector<double> &GetLocalBoundaryValues() const;

    const std::vector<unsigned> &GetBoundaryValuesGlobalIndex() const;

    const std::vector<unsigned> &GetBoundaryValuesLocalIndex() const;

    const std::vector<Cartesian<unsigned>> &GetBoundaryValuesCartesianGlobalIndex() const;

    virtual void ComputeBoundaryIndices();

    virtual void ComputeBoundaryValues() = 0;

protected:
    bool CheckIsOnDirichletBoundary(int i,
                                    int j,
                                    int k);

    std::vector<double> local_boundary_values_;
    std::vector<unsigned> boundary_values_global_index_;
    std::vector<unsigned> boundary_values_local_index_;
    std::vector<Cartesian<unsigned>> boundary_values_cartesian_global_index_;

    const FEM &fem_x_, &fem_y_, &fem_z_;
    const unsigned number_total_nodes_x_, number_total_nodes_y_, number_total_nodes_z_;
    std::array<int, 6> owned_index;
};

#endif //TUCKER_TENSOR_KSDFT_BOUNDARYVALUESCONTAINER_H
