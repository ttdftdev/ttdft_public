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

#ifndef TUCKERDFTSPARSE_ATOMS_NONLOCALMAP1D_H_
#define TUCKERDFTSPARSE_ATOMS_NONLOCALMAP1D_H_

#include <vector>
#include "../fem/FEM.h"

namespace NonLocalMap {
    struct NonLocalMap1D {
        std::vector<int> elemNonLocGridToFullGrid;
        std::vector<double> refPointInFullGrid;
        std::vector<std::vector<double> > shapeFunctionMatrixFullGrid;

        std::vector<int> elemFullGridToNonLocGrid;
        std::vector<double> refPointInNonLocGrid;
        std::vector<std::vector<double> > shapeFunctionMatrixNonLocGrid;
    };

    class NonLocalMap1DFactory {
    public:
        NonLocalMap1DFactory(const FEM &fem,
                             const FEM &fem_linear,
                             const FEM &fem_non_loc,
                             const FEM &fem_non_loc_linear);

        void generateNonLocalMap(const double coord,
                                 NonLocalMap1D &nonLocalMap1D);

    protected:
        const FEM &fem, &femLinear, &femNonLoc, &femNonLocLinear;

        void generateElemNonLocGridRefPointToFullGrid(const double coord,
                                                      std::vector<int> &elemNonLocGridToFullGrid,
                                                      std::vector<double> &refPointInFullGrid);

        void generateShapeFunctionMatrixFullGrid(const std::vector<double> &refPointInFullGrid,
                                                 std::vector<std::vector<double> > &shapeFunctionMatrixFullGrid);

        void generateElemFullGridRefPointToNonLoc(const double coord,
                                                  std::vector<int> &elemFullGridToNonLocGrid,
                                                  std::vector<double> &refPointInNonLocGrid);

        void generateShapeFunctionMatrixNonLocGrid(const std::vector<int> &elemFullGridToNonLocGrid,
                                                   const std::vector<double> &refPointInNonLocGrid,
                                                   std::vector<std::vector<double> > &shapeFunctionMatrixNonLocGrid);
    };

}
#endif //TUCKERDFTSPARSE_ATOMS_NONLOCALMAP1D_H_
