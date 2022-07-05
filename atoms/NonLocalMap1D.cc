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

#include <algorithm>
#include "NonLocalMap1D.h"

namespace {
    double poly_eval(const std::vector<double> &plist,
                     const double &x);

    std::vector<double> add(const std::vector<double> &p1,
                            const std::vector<double> &p2);

    std::vector<double> poly_multiply(const std::vector<double> &p1,
                                      const std::vector<double> &p2);
}

NonLocalMap::NonLocalMap1DFactory::NonLocalMap1DFactory(const FEM &fem,
                                                        const FEM &fem_linear,
                                                        const FEM &fem_non_loc,
                                                        const FEM &fem_non_loc_linear)
        : fem(fem),
          femLinear(fem_linear),
          femNonLoc(fem_non_loc),
          femNonLocLinear(fem_non_loc_linear) {}

void NonLocalMap::NonLocalMap1DFactory::generateNonLocalMap(const double coord,
                                                            NonLocalMap::NonLocalMap1D &nonLocalMap1D) {
    generateElemNonLocGridRefPointToFullGrid(coord,
                                             nonLocalMap1D.elemNonLocGridToFullGrid,
                                             nonLocalMap1D.refPointInFullGrid);
    generateShapeFunctionMatrixFullGrid(nonLocalMap1D.refPointInFullGrid,
                                        nonLocalMap1D.shapeFunctionMatrixFullGrid);
    generateElemFullGridRefPointToNonLoc(coord,
                                         nonLocalMap1D.elemFullGridToNonLocGrid,
                                         nonLocalMap1D.refPointInNonLocGrid);
    generateShapeFunctionMatrixNonLocGrid(nonLocalMap1D.elemFullGridToNonLocGrid,
                                          nonLocalMap1D.refPointInNonLocGrid,
                                          nonLocalMap1D.shapeFunctionMatrixNonLocGrid);

}

void NonLocalMap::NonLocalMap1DFactory::generateElemNonLocGridRefPointToFullGrid(const double coord,
                                                                                 std::vector<int> &elemNonLocGridToFullGrid,
                                                                                 std::vector<double> &refPointInFullGrid) {
    const std::vector<double> &positionQuadValuesNonLoc = femNonLoc.getPositionQuadPointValues();
    int sizeCompactSupport = femNonLoc.getTotalNumberQuadPoints();

    elemNonLocGridToFullGrid = std::vector<int>(sizeCompactSupport,
                                                0);
    refPointInFullGrid = std::vector<double>(sizeCompactSupport,
                                             0);

    std::vector<double> positionQuadValuesNonLocShift = positionQuadValuesNonLoc;
    std::for_each(positionQuadValuesNonLocShift.begin(),
                  positionQuadValuesNonLocShift.end(),
                  [&coord](double &d) { d += coord; });
    for (int iPoint = 0; iPoint < sizeCompactSupport; ++iPoint) {
        for (int iElem = 0; iElem < femLinear.getNumberElements(); ++iElem) {
            const std::vector<int> &localNodeIds = femLinear.getElementConnectivity()[iElem];
            double localNodeCoordinates0 = femLinear.getGlobalNodalCoord()[localNodeIds[0]];
            double localNodeCoordinates1 = femLinear.getGlobalNodalCoord()[localNodeIds[1]];
            double xi = (2 * positionQuadValuesNonLocShift[iPoint] - (localNodeCoordinates1 + localNodeCoordinates0))
                        / (localNodeCoordinates1 - localNodeCoordinates0);
            if (xi >= -1.0 && xi <= 1.0) {
                elemNonLocGridToFullGrid[iPoint] = iElem;
                refPointInFullGrid[iPoint] = xi;
                break;
            }
        }
    }
}

void
NonLocalMap::NonLocalMap1DFactory::generateShapeFunctionMatrixFullGrid(const std::vector<double> &refPointInFullGrid,
                                                                       std::vector<std::vector<double> > &shapeFunctionMatrixFullGrid) {

    // generate unit bi-nodal data
    int numberNodesPerElement = fem.getNumberNodesPerElement();
    std::vector<double> refNodalCoordinates(numberNodesPerElement,
                                            0.0);
    for (int i = 0; i < int(numberNodesPerElement / 2); ++i) {
        refNodalCoordinates[i] = -1.0 + i * (2.0 / (numberNodesPerElement - 1));
        refNodalCoordinates[numberNodesPerElement - i - 1] = -refNodalCoordinates[i];
    }

    int sizeCompactSupport = femNonLoc.getTotalNumberQuadPoints();

    shapeFunctionMatrixFullGrid =
            std::vector<std::vector<double> >(sizeCompactSupport,
                                              std::vector<double>(numberNodesPerElement,
                                                                  0));

    for (int iPoint = 0; iPoint < sizeCompactSupport; ++iPoint) {
        for (int i = 0; i < numberNodesPerElement; ++i) {
            double xi = refNodalCoordinates[i];
            std::vector<double> shapeFunctionX(1,
                                               1.0);
            for (int j = 0; j != numberNodesPerElement; ++j) {
                if (i != j) {
                    double xj = refNodalCoordinates[j];
                    std::vector<double> temp;
                    temp.push_back(-xj / (xi - xj));
                    temp.push_back(1.0 / (xi - xj));
                    shapeFunctionX = poly_multiply(shapeFunctionX,
                                                   temp);
                }
            }
            shapeFunctionMatrixFullGrid[iPoint][i] = poly_eval(shapeFunctionX,
                                                               refPointInFullGrid[iPoint]);
        }
    }
}

void NonLocalMap::NonLocalMap1DFactory::generateElemFullGridRefPointToNonLoc(const double coord,
                                                                             std::vector<int> &elemFullGridToNonLocGrid,
                                                                             std::vector<double> &refPointInNonLocGrid) {
    elemFullGridToNonLocGrid = std::vector<int>(fem.getTotalNumberQuadPoints(),
                                                -1);
    refPointInNonLocGrid = std::vector<double>(fem.getTotalNumberQuadPoints(),
                                               2.0);

    std::vector<double> positionQuadValuesFullGridShift = fem.getPositionQuadPointValues();
    std::for_each(positionQuadValuesFullGridShift.begin(),
                  positionQuadValuesFullGridShift.end(),
                  [&coord](double &d) { d -= coord; });
    for (int iPoint = 0; iPoint < fem.getTotalNumberQuadPoints(); ++iPoint) {
        for (int iElem = 0; iElem < femNonLoc.getNumberElements(); ++iElem) {
            const std::vector<int> &localNodesIds = femNonLocLinear.getElementConnectivity()[iElem];
            double localNodesCoordinates0 = femNonLocLinear.getGlobalNodalCoord()[localNodesIds[0]];
            double localNodesCoordinates1 = femNonLocLinear.getGlobalNodalCoord()[localNodesIds[1]];
            double xi =
                    (2 * positionQuadValuesFullGridShift[iPoint] - (localNodesCoordinates1 + localNodesCoordinates0))
                    / (localNodesCoordinates1 - localNodesCoordinates0);
            if (xi >= -1.0 && xi <= 1.0) {
                elemFullGridToNonLocGrid[iPoint] = iElem;
                refPointInNonLocGrid[iPoint] = xi;
                break;
            }
        }
    }
}

void
NonLocalMap::NonLocalMap1DFactory::generateShapeFunctionMatrixNonLocGrid(const std::vector<int> &elemFullGridToNonLocGrid,
                                                                         const std::vector<double> &refPointInNonLocGrid,
                                                                         std::vector<std::vector<double> > &shapeFunctionMatrixNonLocGrid) {

    shapeFunctionMatrixNonLocGrid = std::vector<std::vector<double> >(fem.getTotalNumberQuadPoints(),
                                                                      std::vector<double>(femNonLoc.getNumberNodesPerElement(),
                                                                                          0.0));

    // generate unit bi-nodal data
    int numberNodesPerElementNonLoc = femNonLoc.getNumberNodesPerElement();
    std::vector<double> refNodalCoordinatesNonLoc(numberNodesPerElementNonLoc,
                                                  0.0);
    for (int i = 0; i < int(numberNodesPerElementNonLoc / 2); ++i) {
        refNodalCoordinatesNonLoc[i] = -1.0 + i * (2.0 / (numberNodesPerElementNonLoc - 1));
        refNodalCoordinatesNonLoc[numberNodesPerElementNonLoc - i - 1] = -refNodalCoordinatesNonLoc[i];
    }

    for (int iPoint = 0; iPoint < fem.getTotalNumberQuadPoints(); ++iPoint) {
        if (elemFullGridToNonLocGrid[iPoint] > -1) {
            for (int i = 0; i < femNonLoc.getNumberNodesPerElement(); ++i) {
                double xi = refNodalCoordinatesNonLoc[i];
                std::vector<double> shapeFunctionX(1,
                                                   1.0);
                for (int j = 0; j != femNonLoc.getNumberNodesPerElement(); ++j) {
                    if (i != j) {
                        double xj = refNodalCoordinatesNonLoc[j];
                        std::vector<double> temp;
                        temp.push_back(-xj / (xi - xj));
                        temp.push_back(1.0 / (xi - xj));
                        shapeFunctionX = poly_multiply(shapeFunctionX,
                                                       temp);
                    }
                }
                shapeFunctionMatrixNonLocGrid[iPoint][i] =
                        poly_eval(shapeFunctionX,
                                  refPointInNonLocGrid[iPoint]);
            }
        }
    }
}

namespace {
    double poly_eval(const std::vector<double> &plist,
                     const double &x) {
        double value = 0.0;
        for (int i = 0; i != plist.size(); ++i) {
            value += plist[i] * std::pow(x,
                                         double(i));
        }
        return value;
    }

    std::vector<double> add(const std::vector<double> &p1,
                            const std::vector<double> &p2) {
        std::vector<double> result;
        if (p1.size() > p2.size()) {
            result = p1;
            for (int i = 0; i != p2.size(); ++i)
                result[i] += p2[i];
        } else {
            result = p2;
            for (int i = 0; i != p1.size(); ++i)
                result[i] += p1[i];
        }
        return result;
    }

//  calculate the coefficients for polynomial interpolating function
    std::vector<double> poly_multiply(const std::vector<double> &p1,
                                      const std::vector<double> &p2) {
        std::vector<double> result;
        if (p1.size() > p2.size()) {
            for (int i = 0; i != p2.size(); ++i) {
                std::vector<double> temp(i,
                                         0.0);
                for (int j = 0; j != p1.size(); ++j)
                    temp.push_back(p1[j] * p2[i]);
                result = add(result,
                             temp);
            }
        } else {
            for (int i = 0; i != p1.size(); ++i) {
                std::vector<double> temp(i,
                                         0.0);
                for (int j = 0; j != p2.size(); ++j)
                    temp.push_back(p2[j] * p1[i]);
                result = add(result,
                             temp);
            }
        }
        return result;
    }
}