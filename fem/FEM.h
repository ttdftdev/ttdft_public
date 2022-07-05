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

#ifndef FEM_H
#define FEM_H

#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <map>

#include "QuadratureRule.h"

class FEM {
public:
    FEM() {}

    FEM(int numberElements,
        std::string quadRule,
        int numNodesPerElement,
        double domainStart,
        double domainEnd,
        double innerDomainSize,
        double innerMeshSize,
        std::string meshType,
        bool electroFlag,
        double coarsingFactor);

    FEM(const FEM &fem);

    FEM(const FEM &fem,
        const int num_outer_mesh,
        double coarsing_factor,
        const double outer_boundary_start,
        const double outer_boundary_end);

    const QuadratureRule &get_quadRule() const;

    const std::string &get_elementType() const;

    double get_domainStart() const;

    double get_domainEnd() const;

    double get_innerDomainSize() const;

    double get_innerMeshSize() const;

    const std::string &get_meshType() const;

    bool is_electroFlag() const;

    int get_numberNodesPerElement() const;

    double getCoarsingFactor() const;

    FEM &operator=(const FEM &fem);

    void computeFieldAndDiffFieldAtAllQuadPoints(const std::vector<double> &NodalVal,
                                                 std::vector<double> &quadVal,
                                                 std::vector<double> &diffQuadVal) const;

    void computeFieldAtAllQuadPoints(const std::vector<double> &NodalVal,
                                     std::vector<double> &quadVal) const;

    void computeDiffFieldAtAllQuadPoints(const std::vector<double> &NodalVal,
                                         std::vector<double> &diffQuadVal) const;

    double integrate_by_nodal_values(const std::vector<double> &nodalFuncVal) const;

    double integrate_by_quad_values(const std::vector<double> &quadFuncVal) const;

    double integrate_inv_by_quad_values(const std::vector<double> &quadFuncVal) const;

    const std::string &get_quadRuleName() const;

    int getNumberElements() const;

    int getNumberNodesPerElement() const;

    int getTotalNumberNodes() const;

    int getNumberQuadPointsPerElement() const;

    int getTotalNumberQuadPoints() const;

    void getLocalNodalCoordinates(const int &elementId,
                                  std::vector<double> &localNodalCoordinate);

    const std::vector<double> &getGlobalNodalCoord() const;

    const std::vector<std::vector<double> > &getShapeFunctionAtQuadPoints() const;

    const std::vector<std::vector<double> > &getShapeFunctionDerivativeAtQuadPoints() const;

    const std::vector<std::vector<int> > &getElementConnectivity() const;

    const std::multimap<int, int> &getNodeToElementMap() const;     //This function is never used, not tested


    const std::vector<double> &getJacobQuadPointValues() const;

    const std::vector<double> &getInvJacobQuadPointValues() const;

    const std::vector<double> &getPositionQuadPointValues() const;

    const std::vector<double> &getWeightQuadPointValues() const;

    const std::vector<std::vector<std::vector<double> > > &getShapeFunctionOverlapIntegral() const;

    const std::vector<std::vector<std::vector<double> > > &getShapeFunctionGradientIntegral() const;

    void computeNodalExternalFunctionShapeFunctionOverlapIntegral(const std::vector<double> &nodalValues,
                                                                  std::vector<std::vector<std::vector<double> > > &funcShpfuncOverlapIntegral);

    void computeQuadExternalFunctionShapeFunctionOverlapIntegral(const std::vector<double> &quadValues,
                                                                 std::vector<std::vector<std::vector<double> > > &funcShpfuncOverlapIntegral) const;

    void computeQuadValuesFromNodalValues(const std::vector<double> &nodalValues,
                                          std::vector<double> &quadValues) const;

private:
    /// number of total elements
    int _numberElements;
    /// number of total nodes
    int _numberNodes;
    /**
     * @brief define quadrature rule
     *
     * Options: 2pt, 3pt, 4pt, 5pt, 6pt, 7pt, 8pt
     */
    QuadratureRule _quadRule;
    std::string _quadRuleName;
    /**
     * @brief define the order of the elment
     *
     * Options: linear, quadratic, cubic, quartic, quintic
     */
    double _domainStart;
    double _domainEnd;
    double _innerDomainSize;
    double _innerMeshSize;
    std::string _meshType;
    bool _electroFlag;
    int _numberNodesPerElement;
    double coarsingFactor;

    std::vector<double> _globalNodalCoord;
    std::vector<std::vector<double> > _shapeFunctionAtQuadPoints;
    std::vector<double> flattened_shape_function_at_quadpoints;
    std::vector<std::vector<double> > _shapeFunctionAtQuadPointsForProjection;
    std::vector<std::vector<double> > _shapeFunctionDerivativeAtQuadPoints;
    std::vector<double> flattened_shape_function_derivative_at_quadpoints;
    std::vector<std::vector<int> > _elementConnectivity;
    std::multimap<int, int> _nodeToElementMap;

    int _numberQuadPointsPerElement;
    std::vector<double> _jacobQuadPointValues;
    std::vector<double> _invJacobQuadPointValues;
    std::vector<double> _positionQuadPointValues;
    std::vector<double> _weightQuadPointValues;
    std::vector<double> _jacob_times_wieght_quad_values;
    std::vector<double> _invjacob_times_wieght_quad_values;

    std::vector<std::vector<std::vector<double> > > _shapeFunctionOverlapIntegral;
    std::vector<std::vector<std::vector<double> > > _shapeFunctionGradientIntegral;
    std::vector<std::vector<std::vector<double> > > shape_function_overlap_quad_values;
    std::vector<std::vector<std::vector<double> > > shape_function_gradient_overlap_quad_values;

    void generateNumberNodes();

    void generateNodes();

    void generateConnectivity();

    void generateShapeFunction();

    void generateShapeFunctionGradient();

    void generateQuadPointData();

    void computeJacobianAtQuadPoints();

    void interpolateNodalToQuadWithinElement(const std::vector<double> &NodalVal,
                                             std::vector<double> &QuadVal) const;

    void interpolateNodalDerivativeToQuadWithinElement(const std::vector<double> &NodalVal,
                                                       std::vector<double> &QuadVal) const;

    void computeShapeFunctionIntegralData(
            std::vector<std::vector<std::vector<double> > > &shapeFunctionOverlapIntegral);

    void computeShapeFunctionGradientIntegralData(
            std::vector<std::vector<std::vector<double> > > &shapeFunctionGradientIntegral);

};

#endif /* FEM_hpp */
