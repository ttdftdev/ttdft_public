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

#ifndef TUCKERDFTSPARSE_RHOCOMPUTER_H
#define TUCKERDFTSPARSE_RHOCOMPUTER_H

#include <vector>
#include <petscmat.h>
#include "../tensor/Tensor3DMPI.h"
#include "../fem/FEM.h"

class RhoComputer {
public:
    RhoComputer(FEM &fem_x,
                FEM &fem_y,
                FEM &fem_z,
                const std::vector<std::vector<double>> &basis_x,
                const std::vector<std::vector<double>> &basis_y,
                const std::vector<std::vector<double>> &basis_z);

private:
    std::vector<std::vector<double>> basis_x, basis_y, basis_z, basis_quad_x, basis_quad_y, basis_quad_z;
    int rank_x, rank_y, rank_z;
};

#endif //TUCKERDFTSPARSE_RHOCOMPUTER_H
