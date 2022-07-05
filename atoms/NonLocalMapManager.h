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

#ifndef TUCKER_TENSOR_KSDFT_NONLOCALMAPMANAGER_H
#define TUCKER_TENSOR_KSDFT_NONLOCALMAPMANAGER_H

#include "../fem/FEM.h"

class NonLocalMapManager {
public:
    NonLocalMapManager(const std::vector<std::vector<double>> &nonLocAtoms,
                       const FEM &femX,
                       const FEM &femY,
                       const FEM &femZ,
                       const FEM &femLinearX,
                       const FEM &femLinearY,
                       const FEM &femLinearZ,
                       const FEM &femNonLocX,
                       const FEM &femNonLocY,
                       const FEM &femNonLocZ,
                       const FEM &femNonLocLinearX,
                       const FEM &femNonLocLinearY,
                       const FEM &femNonLocLinearZ,
                       const double radiusDeltaVlx,
                       const double radiusDeltaVly,
                       const double radiusDeltaVlz);

    int getNumberNonLocAtoms() const;

    int getSizeCompactSupportX() const;

    int getSizeCompactSupportY() const;

    int getSizeCompactSupportZ() const;

    const std::vector<std::vector<int>> &getElemNonLocGridToFullGridX() const;

    const std::vector<std::vector<int>> &getElemNonLocGridToFullGridY() const;

    const std::vector<std::vector<int>> &getElemNonLocGridToFullGridZ() const;

    const std::vector<std::vector<double>> &getRefPointInFullGridX() const;

    const std::vector<std::vector<double>> &getRefPointInFullGridY() const;

    const std::vector<std::vector<double>> &getRefPointInFullGridZ() const;

    const std::vector<std::vector<std::vector<double>>> &getShapeFunctionMatrixFullGridX() const;

    const std::vector<std::vector<std::vector<double>>> &getShapeFunctionMatrixFullGridY() const;

    const std::vector<std::vector<std::vector<double>>> &getShapeFunctionMatrixFullGridZ() const;

    const std::vector<std::vector<int>> &getElemFullGridToNonLocGridX() const;

    const std::vector<std::vector<int>> &getElemFullGridToNonLocGridY() const;

    const std::vector<std::vector<int>> &getElemFullGridToNonLocGridZ() const;

    const std::vector<std::vector<double>> &getRefPointInNonLocGridX() const;

    const std::vector<std::vector<double>> &getRefPointInNonLocGridY() const;

    const std::vector<std::vector<double>> &getRefPointInNonLocGridZ() const;

    const std::vector<std::vector<std::vector<double>>> &getShapeFunctionMatrixNonLocGridX() const;

    const std::vector<std::vector<std::vector<double>>> &getShapeFunctionMatrixNonLocGridY() const;

    const std::vector<std::vector<std::vector<double>>> &getShapeFunctionMatrixNonLocGridZ() const;

    const std::vector<std::vector<int>> &getAtomIdxToElementMap() const;

    const std::vector<std::vector<int>> &getAtomIdyToElementMap() const;

    const std::vector<std::vector<int>> &getAtomIdzToElementMap() const;

    const std::vector<std::vector<int>> &getAtomIdxToGlobalNodeIds() const;

    const std::vector<std::vector<int>> &getAtomIdyToGlobalNodeIds() const;

    const std::vector<std::vector<int>> &getAtomIdzToGlobalNodeIds() const;

    void release() {
        elemNonLocGridToFullGridX.clear();
        elemNonLocGridToFullGridX.shrink_to_fit();
        elemNonLocGridToFullGridY.clear();
        elemNonLocGridToFullGridY.shrink_to_fit();
        elemNonLocGridToFullGridZ.clear();
        elemNonLocGridToFullGridZ.shrink_to_fit();
        refPointInFullGridX.clear();
        refPointInFullGridX.shrink_to_fit();
        refPointInFullGridY.clear();
        refPointInFullGridY.shrink_to_fit();
        refPointInFullGridZ.clear();
        refPointInFullGridZ.shrink_to_fit();
        shapeFunctionMatrixFullGridX.clear();
        shapeFunctionMatrixFullGridX.shrink_to_fit();
        shapeFunctionMatrixFullGridY.clear();
        shapeFunctionMatrixFullGridY.shrink_to_fit();
        shapeFunctionMatrixFullGridZ.clear();
        shapeFunctionMatrixFullGridZ.shrink_to_fit();
        elemFullGridToNonLocGridX.clear();
        elemFullGridToNonLocGridX.shrink_to_fit();
        elemFullGridToNonLocGridY.clear();
        elemFullGridToNonLocGridY.shrink_to_fit();
        elemFullGridToNonLocGridZ.clear();
        elemFullGridToNonLocGridZ.shrink_to_fit();
        refPointInNonLocGridX.clear();
        refPointInNonLocGridX.shrink_to_fit();
        refPointInNonLocGridY.clear();
        refPointInNonLocGridY.shrink_to_fit();
        refPointInNonLocGridZ.clear();
        refPointInNonLocGridZ.shrink_to_fit();
        shapeFunctionMatrixNonLocGridX.clear();
        shapeFunctionMatrixNonLocGridX.shrink_to_fit();
        shapeFunctionMatrixNonLocGridY.clear();
        shapeFunctionMatrixNonLocGridY.shrink_to_fit();
        shapeFunctionMatrixNonLocGridZ.clear();
        shapeFunctionMatrixNonLocGridZ.shrink_to_fit();
        atomIdxToElementMap.clear();
        atomIdxToElementMap.shrink_to_fit();
        atomIdyToElementMap.clear();
        atomIdyToElementMap.shrink_to_fit();
        atomIdzToElementMap.clear();
        atomIdzToElementMap.shrink_to_fit();
        atomIdxToGlobalNodeIds.clear();
        atomIdxToGlobalNodeIds.shrink_to_fit();
        atomIdyToGlobalNodeIds.clear();
        atomIdyToGlobalNodeIds.shrink_to_fit();
        atomIdzToGlobalNodeIds.clear();
        atomIdzToGlobalNodeIds.shrink_to_fit();
    }

private:
    enum Cartesian {
        x = 1, y = 2, z = 3
    };

    int numberNonLocAtoms;
    int sizeCompactSupportX, sizeCompactSupportY, sizeCompactSupportZ;
    double radiusDeltaVlx, radiusDeltaVly, radiusDeltaVlz;

    std::vector<std::vector<int> > elemNonLocGridToFullGridX, elemNonLocGridToFullGridY, elemNonLocGridToFullGridZ;
    std::vector<std::vector<double> > refPointInFullGridX, refPointInFullGridY, refPointInFullGridZ;
    std::vector<std::vector<std::vector<double> > > shapeFunctionMatrixFullGridX, shapeFunctionMatrixFullGridY,
            shapeFunctionMatrixFullGridZ;

    std::vector<std::vector<int> > elemFullGridToNonLocGridX, elemFullGridToNonLocGridY, elemFullGridToNonLocGridZ;
    std::vector<std::vector<double> > refPointInNonLocGridX, refPointInNonLocGridY, refPointInNonLocGridZ;
    std::vector<std::vector<std::vector<double> > > shapeFunctionMatrixNonLocGridX, shapeFunctionMatrixNonLocGridY,
            shapeFunctionMatrixNonLocGridZ;

    std::vector<std::vector<int> > atomIdxToElementMap, atomIdyToElementMap, atomIdzToElementMap;
    std::vector<std::vector<int> > atomIdxToGlobalNodeIds, atomIdyToGlobalNodeIds, atomIdzToGlobalNodeIds;

    void generateElemNonLocGridRefPointToFullGrid(const std::vector<std::vector<double> > &nonLocAtoms,
                                                  const FEM &femNonLoc,
                                                  const FEM &femLinear,
                                                  Cartesian cart,
                                                  std::vector<std::vector<int> > &elemNonLocGridToFullGrid,
                                                  std::vector<std::vector<double> > &refPointInFullGrid);

    void generateShapeFunctionMatrixFullGrid(const std::vector<std::vector<double> > &nonLocAtoms,
                                             const FEM &fem,
                                             const FEM &femNonLoc,
                                             const std::vector<std::vector<double> > &refPointInFullGrid,
                                             std::vector<std::vector<std::vector<double> > > &shapeFunctionMatrixFullGrid);

    void generateElemFullGridRefPointToNonLoc(const std::vector<std::vector<double> > &nonLocAtoms,
                                              const FEM &fem,
                                              const FEM &femNonLoc,
                                              const FEM &femNonLocLinear,
                                              Cartesian cart,
                                              std::vector<std::vector<int> > &elemFullGridToNonLocGrid,
                                              std::vector<std::vector<double> > &refPointInNonLocGrid);

    void generateShapeFunctionMatrixNonLocGrid(const std::vector<std::vector<double> > &nonLocAtoms,
                                               const FEM &fem,
                                               const FEM &femNonLoc,
                                               const std::vector<std::vector<int> > &elemFullGridToNonLocGrid,
                                               const std::vector<std::vector<double> > &refPointInNonLocGrid,
                                               std::vector<std::vector<std::vector<double> > > &shapeFunctionMatrixNonLocGrid);

    void generateAtomMap(const std::vector<std::vector<double> > &nonLocAtoms,
                         const FEM &fem,
                         const FEM &femLinear,
                         const double radiusDeltaVl,
                         Cartesian cart,
                         std::vector<std::vector<int> > &atomIdToElementMap,
                         std::vector<std::vector<int> > &atomIdToGlobalNodeIds);
};

#endif //TUCKER_TENSOR_KSDFT_NONLOCALMAPMANAGER_H
