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
#include <set>
#include "NonLocalMapManager.h"

namespace {
    double poly_eval(const std::vector<double> &plist,
                     const double &x);

    std::vector<double> add(const std::vector<double> &p1,
                            const std::vector<double> &p2);

    std::vector<double> poly_multiply(const std::vector<double> &p1,
                                      const std::vector<double> &p2);
}

NonLocalMapManager::NonLocalMapManager(const std::vector<std::vector<double>> &nonLocAtoms,
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
                                       const double radiusDeltaVlz) :
        radiusDeltaVlx(radiusDeltaVlx),
        radiusDeltaVly(radiusDeltaVly),
        radiusDeltaVlz(radiusDeltaVlz) {

    numberNonLocAtoms = nonLocAtoms.size();
    sizeCompactSupportX = femNonLocX.getTotalNumberQuadPoints();
    sizeCompactSupportY = femNonLocY.getTotalNumberQuadPoints();
    sizeCompactSupportZ = femNonLocZ.getTotalNumberQuadPoints();

    generateElemNonLocGridRefPointToFullGrid(nonLocAtoms,
                                             femNonLocX,
                                             femLinearX,
                                             x,
                                             elemNonLocGridToFullGridX,
                                             refPointInFullGridX);
    generateElemNonLocGridRefPointToFullGrid(nonLocAtoms,
                                             femNonLocY,
                                             femLinearY,
                                             y,
                                             elemNonLocGridToFullGridY,
                                             refPointInFullGridY);
    generateElemNonLocGridRefPointToFullGrid(nonLocAtoms,
                                             femNonLocZ,
                                             femLinearZ,
                                             z,
                                             elemNonLocGridToFullGridZ,
                                             refPointInFullGridZ);

    generateShapeFunctionMatrixFullGrid(nonLocAtoms,
                                        femX,
                                        femNonLocX,
                                        refPointInFullGridX,
                                        shapeFunctionMatrixFullGridX);
    generateShapeFunctionMatrixFullGrid(nonLocAtoms,
                                        femY,
                                        femNonLocY,
                                        refPointInFullGridY,
                                        shapeFunctionMatrixFullGridY);
    generateShapeFunctionMatrixFullGrid(nonLocAtoms,
                                        femZ,
                                        femNonLocZ,
                                        refPointInFullGridZ,
                                        shapeFunctionMatrixFullGridZ);

    generateElemFullGridRefPointToNonLoc(nonLocAtoms,
                                         femX,
                                         femNonLocX,
                                         femNonLocLinearX,
                                         x,
                                         elemFullGridToNonLocGridX,
                                         refPointInNonLocGridX);
    generateElemFullGridRefPointToNonLoc(nonLocAtoms,
                                         femY,
                                         femNonLocY,
                                         femNonLocLinearY,
                                         y,
                                         elemFullGridToNonLocGridY,
                                         refPointInNonLocGridY);
    generateElemFullGridRefPointToNonLoc(nonLocAtoms,
                                         femZ,
                                         femNonLocZ,
                                         femNonLocLinearZ,
                                         z,
                                         elemFullGridToNonLocGridZ,
                                         refPointInNonLocGridZ);

    generateShapeFunctionMatrixNonLocGrid(nonLocAtoms,
                                          femX,
                                          femNonLocX,
                                          elemFullGridToNonLocGridX,
                                          refPointInNonLocGridX,
                                          shapeFunctionMatrixNonLocGridX);
    generateShapeFunctionMatrixNonLocGrid(nonLocAtoms,
                                          femY,
                                          femNonLocY,
                                          elemFullGridToNonLocGridY,
                                          refPointInNonLocGridY,
                                          shapeFunctionMatrixNonLocGridY);
    generateShapeFunctionMatrixNonLocGrid(nonLocAtoms,
                                          femZ,
                                          femNonLocZ,
                                          elemFullGridToNonLocGridZ,
                                          refPointInNonLocGridZ,
                                          shapeFunctionMatrixNonLocGridZ);

    generateAtomMap(nonLocAtoms,
                    femX,
                    femLinearX,
                    radiusDeltaVlx,
                    x,
                    atomIdxToElementMap,
                    atomIdxToGlobalNodeIds);
    generateAtomMap(nonLocAtoms,
                    femY,
                    femLinearY,
                    radiusDeltaVly,
                    y,
                    atomIdyToElementMap,
                    atomIdyToGlobalNodeIds);
    generateAtomMap(nonLocAtoms,
                    femZ,
                    femLinearZ,
                    radiusDeltaVlz,
                    z,
                    atomIdzToElementMap,
                    atomIdzToGlobalNodeIds);
}

void NonLocalMapManager::generateElemNonLocGridRefPointToFullGrid(const std::vector<std::vector<double> > &nonLocAtoms,
                                                                  const FEM &femNonLoc,
                                                                  const FEM &femLinear,
                                                                  NonLocalMapManager::Cartesian cart,
                                                                  std::vector<std::vector<int> > &elemNonLocGridToFullGrid,
                                                                  std::vector<std::vector<double> > &refPointInFullGrid) {
    const std::vector<double> &positionQuadValuesNonLoc = femNonLoc.getPositionQuadPointValues();
    int numNonLocalAtoms = nonLocAtoms.size();
    int sizeCompactSupport = femNonLoc.getTotalNumberQuadPoints();

    elemNonLocGridToFullGrid = std::vector<std::vector<int> >(numNonLocalAtoms,
                                                              std::vector<int>(sizeCompactSupport,
                                                                               0));
    refPointInFullGrid = std::vector<std::vector<double> >(numNonLocalAtoms,
                                                           std::vector<double>(sizeCompactSupport,
                                                                               0));

    for (int iAtom = 0; iAtom < numNonLocalAtoms; ++iAtom) {
        double coord = nonLocAtoms[iAtom][cart];
        std::vector<double> positionQuadValuesNonLocShift = positionQuadValuesNonLoc;
        std::for_each(positionQuadValuesNonLocShift.begin(),
                      positionQuadValuesNonLocShift.end(),
                      [&coord](double &d) { d += coord; });
        for (int iPoint = 0; iPoint < sizeCompactSupport; ++iPoint) {
            for (int iElem = 0; iElem < femLinear.getNumberElements(); ++iElem) {
                const std::vector<int> &localNodeIds = femLinear.getElementConnectivity()[iElem];
                double localNodeCoordinates0 = femLinear.getGlobalNodalCoord()[localNodeIds[0]];
                double localNodeCoordinates1 = femLinear.getGlobalNodalCoord()[localNodeIds[1]];
                double xi =
                        (2 * positionQuadValuesNonLocShift[iPoint] - (localNodeCoordinates1 + localNodeCoordinates0))
                        / (localNodeCoordinates1 - localNodeCoordinates0);
                if (xi >= -1.0 && xi <= 1.0) {
                    elemNonLocGridToFullGrid[iAtom][iPoint] = iElem;
                    refPointInFullGrid[iAtom][iPoint] = xi;
                    break;
                }
            }
        }
    }
}

void NonLocalMapManager::generateShapeFunctionMatrixFullGrid(const std::vector<std::vector<double> > &nonLocAtoms,
                                                             const FEM &fem,
                                                             const FEM &femNonLoc,
                                                             const std::vector<std::vector<double> > &refPointInFullGrid,
                                                             std::vector<std::vector<std::vector<double> > > &shapeFunctionMatrixFullGrid) {

    // generate unit bi-nodal data
    int numberNodesPerElement = fem.getNumberNodesPerElement();
    std::vector<double> refNodalCoordinates(numberNodesPerElement,
                                            0.0);
    for (int i = 0; i < int(numberNodesPerElement / 2); ++i) {
        refNodalCoordinates[i] = -1.0 + i * (2.0 / (numberNodesPerElement - 1));
        refNodalCoordinates[numberNodesPerElement - i - 1] = -refNodalCoordinates[i];
    }

    int numNonLocalAtoms = nonLocAtoms.size();
    int sizeCompactSupport = femNonLoc.getTotalNumberQuadPoints();

    shapeFunctionMatrixFullGrid = std::vector<std::vector<std::vector<double> > >(numNonLocalAtoms,
                                                                                  std::vector<std::vector<double> >(
                                                                                          sizeCompactSupport,
                                                                                          std::vector<double>(
                                                                                                  numberNodesPerElement,
                                                                                                  0)));

    for (int iAtom = 0; iAtom < numNonLocalAtoms; ++iAtom) {
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
                shapeFunctionMatrixFullGrid[iAtom][iPoint][i] = poly_eval(shapeFunctionX,
                                                                          refPointInFullGrid[iAtom][iPoint]);
            }
        }
    }
}

void NonLocalMapManager::generateElemFullGridRefPointToNonLoc(const std::vector<std::vector<double> > &nonLocAtoms,
                                                              const FEM &fem,
                                                              const FEM &femNonLoc,
                                                              const FEM &femNonLocLinear,
                                                              NonLocalMapManager::Cartesian cart,
                                                              std::vector<std::vector<int> > &elemFullGridToNonLocGrid,
                                                              std::vector<std::vector<double> > &refPointInNonLocGrid) {
    int numNonLocalAtoms = nonLocAtoms.size();

    elemFullGridToNonLocGrid =
            std::vector<std::vector<int> >(numNonLocalAtoms,
                                           std::vector<int>(fem.getTotalNumberQuadPoints(),
                                                            -1));
    refPointInNonLocGrid =
            std::vector<std::vector<double> >(numNonLocalAtoms,
                                              std::vector<double>(fem.getTotalNumberQuadPoints(),
                                                                  2.0));

    for (int iAtom = 0; iAtom < numNonLocalAtoms; ++iAtom) {
        double coord = nonLocAtoms[iAtom][cart];
        std::vector<double> positionQuadValuesFullGridShift = fem.getPositionQuadPointValues();
        std::for_each(positionQuadValuesFullGridShift.begin(),
                      positionQuadValuesFullGridShift.end(),
                      [&coord](double &d) { d -= coord; });
        for (int iPoint = 0; iPoint < fem.getTotalNumberQuadPoints(); ++iPoint) {
            for (int iElem = 0; iElem < femNonLoc.getNumberElements(); ++iElem) {
                const std::vector<int> &localNodesIds = femNonLocLinear.getElementConnectivity()[iElem];
                double localNodesCoordinates0 = femNonLocLinear.getGlobalNodalCoord()[localNodesIds[0]];
                double localNodesCoordinates1 = femNonLocLinear.getGlobalNodalCoord()[localNodesIds[1]];
                double xi = (2 * positionQuadValuesFullGridShift[iPoint] -
                             (localNodesCoordinates1 + localNodesCoordinates0))
                            / (localNodesCoordinates1 - localNodesCoordinates0);
                if (xi >= -1.0 && xi <= 1.0) {
                    elemFullGridToNonLocGrid[iAtom][iPoint] = iElem;
                    refPointInNonLocGrid[iAtom][iPoint] = xi;
                    break;
                }
            }
        }
    }
}

void NonLocalMapManager::generateShapeFunctionMatrixNonLocGrid(const std::vector<std::vector<double> > &nonLocAtoms,
                                                               const FEM &fem,
                                                               const FEM &femNonLoc,
                                                               const std::vector<std::vector<int> > &elemFullGridToNonLocGrid,
                                                               const std::vector<std::vector<double> > &refPointInNonLocGrid,
                                                               std::vector<std::vector<std::vector<double> > > &shapeFunctionMatrixNonLocGrid) {
    int numNonLocalAtoms = nonLocAtoms.size();

    shapeFunctionMatrixNonLocGrid = std::vector<std::vector<std::vector<double> > >(numNonLocalAtoms,
                                                                                    std::vector<std::vector<double> >(fem.getTotalNumberQuadPoints(),
                                                                                                                      std::vector<
                                                                                                                              double>(
                                                                                                                              femNonLoc.getNumberNodesPerElement(),
                                                                                                                              0.0)));

    // generate unit bi-nodal data
    int numberNodesPerElementNonLoc = femNonLoc.getNumberNodesPerElement();
    std::vector<double> refNodalCoordinatesNonLoc(numberNodesPerElementNonLoc,
                                                  0.0);
    for (int i = 0; i < int(numberNodesPerElementNonLoc / 2); ++i) {
        refNodalCoordinatesNonLoc[i] = -1.0 + i * (2.0 / (numberNodesPerElementNonLoc - 1));
        refNodalCoordinatesNonLoc[numberNodesPerElementNonLoc - i - 1] = -refNodalCoordinatesNonLoc[i];
    }

    for (int iAtom = 0; iAtom < numNonLocalAtoms; ++iAtom) {
        for (int iPoint = 0; iPoint < fem.getTotalNumberQuadPoints(); ++iPoint) {
            if (elemFullGridToNonLocGrid[iAtom][iPoint] > -1) {
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
                    shapeFunctionMatrixNonLocGrid[iAtom][iPoint][i] =
                            poly_eval(shapeFunctionX,
                                      refPointInNonLocGrid[iAtom][iPoint]);
                }
            }
        }
    }
}

void NonLocalMapManager::generateAtomMap(const std::vector<std::vector<double> > &nonLocAtoms,
                                         const FEM &fem,
                                         const FEM &femLinear,
                                         const double radiusDeltaVl,
                                         Cartesian cart,
                                         std::vector<std::vector<int> > &atomIdToElementMap,
                                         std::vector<std::vector<int> > &atomIdToGlobalNodeIds) {
    int numNonLocalAtoms = nonLocAtoms.size();

    std::vector<std::set<int> > tempElementMap(numNonLocalAtoms);
    std::vector<std::set<int> > tempGlobalIdMap(numNonLocalAtoms);
    atomIdToElementMap = std::vector<std::vector<int> >(numNonLocalAtoms);
    atomIdToGlobalNodeIds = std::vector<std::vector<int> >(numNonLocalAtoms);

    for (int i = 0; i < numNonLocalAtoms; ++i) {
        double coord = nonLocAtoms[i][cart];
        std::vector<double> distance1DNodalGrid(femLinear.getGlobalNodalCoord());
        std::for_each(distance1DNodalGrid.begin(),
                      distance1DNodalGrid.end(),
                      [&coord](double &d) { d = std::abs(d - coord); });
        for (int p = 0; p < distance1DNodalGrid.size(); ++p) {
            if (distance1DNodalGrid[p] <= radiusDeltaVl) {
                tempElementMap[i].insert(p);
            }
        }
        if (!tempElementMap[i].empty()) {
            auto iter = (--tempElementMap[i].end());
            tempElementMap[i].erase(iter);
        }
        for (auto iter = tempElementMap[i].begin(); iter != tempElementMap[i].end(); ++iter) {
            const std::vector<int> &eleConn = fem.getElementConnectivity()[*iter];
            for (auto conn: eleConn) {
                tempGlobalIdMap[i].insert(conn);
            }
        }
    }

    for (int i = 0; i < numNonLocalAtoms; ++i) {
        for (auto iter = tempElementMap[i].begin(); iter != tempElementMap[i].end(); ++iter) {
            atomIdToElementMap[i].push_back(*iter);
        }
        for (auto iter = tempGlobalIdMap[i].begin(); iter != tempGlobalIdMap[i].end(); ++iter) {
            atomIdToGlobalNodeIds[i].push_back(*iter);
        }
    }
}

int NonLocalMapManager::getNumberNonLocAtoms() const {
    return numberNonLocAtoms;
}

int NonLocalMapManager::getSizeCompactSupportX() const {
    return sizeCompactSupportX;
}

int NonLocalMapManager::getSizeCompactSupportY() const {
    return sizeCompactSupportY;
}

int NonLocalMapManager::getSizeCompactSupportZ() const {
    return sizeCompactSupportZ;
}

const std::vector<std::vector<int>> &NonLocalMapManager::getElemNonLocGridToFullGridX() const {
    return elemNonLocGridToFullGridX;
}

const std::vector<std::vector<int>> &NonLocalMapManager::getElemNonLocGridToFullGridY() const {
    return elemNonLocGridToFullGridY;
}

const std::vector<std::vector<int>> &NonLocalMapManager::getElemNonLocGridToFullGridZ() const {
    return elemNonLocGridToFullGridZ;
}

const std::vector<std::vector<double>> &NonLocalMapManager::getRefPointInFullGridX() const {
    return refPointInFullGridX;
}

const std::vector<std::vector<double>> &NonLocalMapManager::getRefPointInFullGridY() const {
    return refPointInFullGridY;
}

const std::vector<std::vector<double>> &NonLocalMapManager::getRefPointInFullGridZ() const {
    return refPointInFullGridZ;
}

const std::vector<std::vector<std::vector<double>>> &NonLocalMapManager::getShapeFunctionMatrixFullGridX() const {
    return shapeFunctionMatrixFullGridX;
}

const std::vector<std::vector<std::vector<double>>> &NonLocalMapManager::getShapeFunctionMatrixFullGridY() const {
    return shapeFunctionMatrixFullGridY;
}

const std::vector<std::vector<std::vector<double>>> &NonLocalMapManager::getShapeFunctionMatrixFullGridZ() const {
    return shapeFunctionMatrixFullGridZ;
}

const std::vector<std::vector<int>> &NonLocalMapManager::getElemFullGridToNonLocGridX() const {
    return elemFullGridToNonLocGridX;
}

const std::vector<std::vector<int>> &NonLocalMapManager::getElemFullGridToNonLocGridY() const {
    return elemFullGridToNonLocGridY;
}

const std::vector<std::vector<int>> &NonLocalMapManager::getElemFullGridToNonLocGridZ() const {
    return elemFullGridToNonLocGridZ;
}

const std::vector<std::vector<double>> &NonLocalMapManager::getRefPointInNonLocGridX() const {
    return refPointInNonLocGridX;
}

const std::vector<std::vector<double>> &NonLocalMapManager::getRefPointInNonLocGridY() const {
    return refPointInNonLocGridY;
}

const std::vector<std::vector<double>> &NonLocalMapManager::getRefPointInNonLocGridZ() const {
    return refPointInNonLocGridZ;
}

const std::vector<std::vector<std::vector<double>>> &NonLocalMapManager::getShapeFunctionMatrixNonLocGridX() const {
    return shapeFunctionMatrixNonLocGridX;
}

const std::vector<std::vector<std::vector<double>>> &NonLocalMapManager::getShapeFunctionMatrixNonLocGridY() const {
    return shapeFunctionMatrixNonLocGridY;
}

const std::vector<std::vector<std::vector<double>>> &NonLocalMapManager::getShapeFunctionMatrixNonLocGridZ() const {
    return shapeFunctionMatrixNonLocGridZ;
}

const std::vector<std::vector<int>> &NonLocalMapManager::getAtomIdxToElementMap() const {
    return atomIdxToElementMap;
}

const std::vector<std::vector<int>> &NonLocalMapManager::getAtomIdyToElementMap() const {
    return atomIdyToElementMap;
}

const std::vector<std::vector<int>> &NonLocalMapManager::getAtomIdzToElementMap() const {
    return atomIdzToElementMap;
}

const std::vector<std::vector<int>> &NonLocalMapManager::getAtomIdxToGlobalNodeIds() const {
    return atomIdxToGlobalNodeIds;
}

const std::vector<std::vector<int>> &NonLocalMapManager::getAtomIdyToGlobalNodeIds() const {
    return atomIdyToGlobalNodeIds;
}

const std::vector<std::vector<int>> &NonLocalMapManager::getAtomIdzToGlobalNodeIds() const {
    return atomIdzToGlobalNodeIds;
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