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

#include <Tucker_Matrix.hpp>
#include <Tucker.hpp>

#include "RhoComputer.h"
#include "../blas_lapack/clinalg.h"

RhoComputer::RhoComputer(FEM &fem_x,
                         FEM &fem_y,
                         FEM &fem_z,
                         const std::vector<std::vector<double>> &basis_x,
                         const std::vector<std::vector<double>> &basis_y,
                         const std::vector<std::vector<double>> &basis_z) :
        basis_x(basis_x),
        basis_y(basis_y),
        basis_z(basis_z) {

    rank_x = basis_x.size();
    rank_y = basis_y.size();
    rank_z = basis_z.size();
    basis_quad_x = std::vector<std::vector<double>>(basis_x);
    basis_quad_y = std::vector<std::vector<double>>(basis_y);
    basis_quad_z = std::vector<std::vector<double>>(basis_z);
    for (auto i = 0; i < rank_x; ++i)
        fem_x.computeFieldAtAllQuadPoints(basis_x[i],
                                          basis_quad_x[i]);
    for (auto i = 0; i < rank_y; ++i)
        fem_y.computeFieldAtAllQuadPoints(basis_y[i],
                                          basis_quad_y[i]);
    for (auto i = 0; i < rank_z; ++i)
        fem_z.computeFieldAtAllQuadPoints(basis_z[i],
                                          basis_quad_z[i]);
}