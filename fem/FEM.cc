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

#include <numeric>
#include <functional>
#include <algorithm>
#include <cassert>
#include <iomanip>
#include "FEM.h"

extern "C" {
double ddot_(const int &n,
             const double *dx,
             const int &incx,
             const double *dy,
             const int &incy);
int dgemv_(const char &trans,
           const int &m,
           const int &n,
           const double &alpha,
           const double *a,
           const int &lda,
           const double *x,
           const int &incx,
           const double &beta,
           double *y,
           const int &incy);
}

//  evaluating the polynomial interpolating function by computing c[i]*x^i
double poly_eval(const std::vector<double> &plist,
                 const double &x) {
    double value = 0.0;
    for (int i = 0; i != plist.size(); ++i) {
        value += plist[i] * std::pow(x,
                                     double(i));
    }
    return value;
}

std::vector<double> poly_derivative(const std::vector<double> &plist) {
    std::vector<double> derivative;
    if (plist.empty())
        return plist;
    else {
        for (int i = 1; i != plist.size(); ++i) {
            derivative.emplace_back(double(i) * plist[i]);
        }
        return derivative;
    }
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

void linspace(const double &vStart,
              const double &vEnd,
              const int &numNodes,
              std::vector<double> &vec) {
    vec = std::vector<double>(numNodes,
                              vStart);
    double interval = (vEnd - vStart) / (numNodes - 1.0);
    for (int i = 1; i != vec.size(); ++i) {
        vec[i] = vec[i - 1] + interval;
    }
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
                temp.emplace_back(p1[j] * p2[i]);
            result = add(result,
                         temp);
        }
    } else {
        for (int i = 0; i != p1.size(); ++i) {
            std::vector<double> temp(i,
                                     0.0);
            for (int j = 0; j != p2.size(); ++j)
                temp.emplace_back(p2[j] * p1[i]);
            result = add(result,
                         temp);
        }
    }
    return result;
}


/**
 * @brief Matrices inner dotting vectors
 *
 * \f$\left[\begin{array}{cc}
 *                a_{11} & a_{12}\\
 *                a_{21} & a_{22}\\
 *                a_{31} & a_{32}
 *          \end{array}\right]
 *    MatDotVec
 *    \left[\begin{array}{c}
 *                b_{1}\\
 *                b_{2}\\
 *                b_{3}
 *          \end{array}\right]
 *    =
 *    \left[\begin{array}{c}
 *                a_{11}b_{1}+a_{21}b_2+a_{31}b_3\\
 *                a_{12}b_{1}+a_{22}b_2+a_{32}b_3
 *          \end{array}\right]\f$
 */
// TODO modifiy this part to blas
void MatDotVec(const std::vector<std::vector<double> > &mat,
               const std::vector<double> &vec,
               std::vector<double> &result) {
    for (int i = 0; i != mat[0].size(); ++i) {
        double temp = 0.0;
        for (int j = 0; j != mat.size(); ++j) {
            temp += vec[j] * mat[j][i];
        }
        result.emplace_back(temp);
    }
}

//  Calculate total number of nodes in 1-D element
void FEM::generateNumberNodes() {
    _numberNodes = _numberElements * (_numberNodesPerElement - 1) + 1;
}

//  Generate uniform nodes for 1-D element
//  Need to be extended to adaptive/manual mesh
//  FIXIT
void FEM::generateNodes() {
    try {
        if (_meshType.compare("uniform") == 0) {
            //interval defines the spacing of uniform mesh
            double interval = std::abs((_domainEnd - _domainStart)) / (_numberNodes - 1);
            double node = _domainStart;
            for (int i = 0; i != _numberNodes; ++i) {
                node = _domainStart + interval * i;
                _globalNodalCoord.emplace_back(node);
            }
        } else if (_meshType.compare("adaptive") == 0) {
            double enlargeCoefficient = coarsingFactor;

            int numberElementsInnerDomain = std::floor(_innerDomainSize / _innerMeshSize);
            int numberElementsOuterDomainOneSide = (_numberElements - numberElementsInnerDomain) / 2;
            double domainCenter = (_domainStart + _domainEnd) / 2;
            double innerDomainStart = domainCenter - (_innerDomainSize / 2.0);
            double innerDomainEnd = domainCenter + (_innerDomainSize / 2.0);
            std::vector<double> linearNodes(_numberElements + 1,
                                            0.0);

            int innerDomainStartIdx = numberElementsOuterDomainOneSide;
            int innerDomainEndIdx = numberElementsOuterDomainOneSide + numberElementsInnerDomain;

            int cnt = 0;
            for (int i = innerDomainStartIdx; i <= innerDomainEndIdx; ++i) {
                linearNodes[i] = innerDomainStart + _innerMeshSize * cnt;
                cnt++;
            }
            cnt = 0;
            double outerMeshSize = _innerMeshSize;
            for (int i = 1; i <= numberElementsOuterDomainOneSide; ++i) {
                outerMeshSize = outerMeshSize * enlargeCoefficient;
                linearNodes[innerDomainStartIdx - i] = linearNodes[innerDomainStartIdx - i + 1] - outerMeshSize;
                linearNodes[innerDomainEndIdx + i] = linearNodes[innerDomainEndIdx + i - 1] + outerMeshSize;
            }

            _numberNodes = _numberElements * (_numberNodesPerElement - 1) + 1;
            _globalNodalCoord = std::vector<double>(_numberNodes,
                                                    0.0);

            for (int i = 0; i < linearNodes.size() - 1; ++i) {
                std::vector<double> newNodes;
                linspace(linearNodes[i],
                         linearNodes[i + 1],
                         _numberNodesPerElement,
                         newNodes);
                newNodes.pop_back();
                for (int j = 0; j != newNodes.size(); ++j) {
                    _globalNodalCoord[i * (_numberNodesPerElement - 1) + j] = newNodes[j];
                }
            }
            _globalNodalCoord[_numberNodes - 1] = _domainEnd;
        } else if (_meshType.compare("adaptive2") == 0) {
            _globalNodalCoord = std::vector<double>(_numberNodes,
                                                    0.0);
            int numberElementsInnerDomain = std::floor(_innerDomainSize / _innerMeshSize);
            double domainCenter = (_domainStart + _domainEnd) / 2;
            double numberPointsInner = std::ceil(numberElementsInnerDomain / 2.0) + 1;

            std::vector<double> nodes0;
            linspace(domainCenter,
                     _innerDomainSize / 2.0,
                     numberPointsInner,
                     nodes0);
            nodes0.pop_back();

            int numberOuterIntervalsFromDomainCenter = (_numberElements - numberElementsInnerDomain) / 2.0;
            double constVal = std::log2(_innerDomainSize / 2.0) / 0.5;
            std::vector<double> nodes1;
            linspace(constVal,
                     std::log2(_domainEnd * _domainEnd),
                     numberOuterIntervalsFromDomainCenter + 1,
                     nodes1);
            for (int i = 0; i != nodes1.size(); ++i) {
                nodes1[i] = std::pow(std::sqrt(2),
                                     nodes1[i]);
            }
            _numberElements = (nodes0.size() + nodes1.size() - 1) * 2;
            std::vector<double> linNodalCoordinates(_numberElements,
                                                    0.0);
            int midIdx = nodes0.size() + nodes1.size() - 1;
            linNodalCoordinates[midIdx] = nodes0[0];
            for (int i = 1; i < nodes0.size(); ++i) {
                linNodalCoordinates[midIdx + i] = nodes0[i];
                linNodalCoordinates[midIdx - i] = -nodes0[i];
            }
            for (int i = 0; i < nodes1.size(); ++i) {
                linNodalCoordinates[midIdx + nodes0.size() + i] = nodes1[i];
                linNodalCoordinates[midIdx - nodes0.size() - i] = -nodes1[i];
            }

            _numberNodes = _numberElements * (_numberNodesPerElement - 1) + 1;
            _globalNodalCoord = std::vector<double>(_numberNodes,
                                                    0.0);

            for (int i = 0; i != linNodalCoordinates.size(); ++i) {
                std::vector<double> newNodes;
                linspace(linNodalCoordinates[i],
                         linNodalCoordinates[i + 1],
                         _numberNodesPerElement,
                         newNodes);
                newNodes.pop_back();
                for (int j = 0; j != newNodes.size(); ++j) {
                    _globalNodalCoord[i * (_numberNodesPerElement - 1) + j] = newNodes[j];
                }
            }
            _globalNodalCoord[_numberNodes - 1] = _domainEnd;
        } else if (_meshType.compare("manual1") == 0) {
            double enlargeCoefficient = 1.5;

            int numberElementsInnerDomain = std::floor(_innerDomainSize / _innerMeshSize);
            int numberElementsOuterDomainOneSide = (_numberElements - numberElementsInnerDomain) / 2;
            double domainCenter = (_domainStart + _domainEnd) / 2;
            double innerDomainStart = domainCenter - (_innerDomainSize / 2.0);
            double innerDomainEnd = domainCenter + (_innerDomainSize / 2.0);
            std::vector<double> linearNodes(_numberElements + 1,
                                            0.0);

            int innerDomainStartIdx = numberElementsOuterDomainOneSide;
            int innerDomainEndIdx = numberElementsOuterDomainOneSide + numberElementsInnerDomain;

            int cnt = 0;
            for (int i = innerDomainStartIdx; i <= innerDomainEndIdx; ++i) {
                linearNodes[i] = innerDomainStart + _innerMeshSize * cnt;
                cnt++;
            }
            cnt = 0;
            double outerMeshSize = 1.5;
            for (int i = 1; i <= numberElementsOuterDomainOneSide; ++i) {
                outerMeshSize = outerMeshSize * enlargeCoefficient;
                linearNodes[innerDomainStartIdx - i] = linearNodes[innerDomainStartIdx - i + 1] - outerMeshSize;
                linearNodes[innerDomainEndIdx + i] = linearNodes[innerDomainEndIdx + i - 1] + outerMeshSize;
            }

            _numberNodes = _numberElements * (_numberNodesPerElement - 1) + 1;
            _globalNodalCoord = std::vector<double>(_numberNodes,
                                                    0.0);

            for (int i = 0; i < linearNodes.size() - 1; ++i) {
                std::vector<double> newNodes;
                linspace(linearNodes[i],
                         linearNodes[i + 1],
                         _numberNodesPerElement,
                         newNodes);
                newNodes.pop_back();
                for (int j = 0; j != newNodes.size(); ++j) {
                    _globalNodalCoord[i * (_numberNodesPerElement - 1) + j] = newNodes[j];
                }
            }
            _globalNodalCoord[_numberNodes - 1] = linearNodes.back();
        } else if (_meshType.compare("hartree_outer") == 0) {
            double enlargeCoefficient = coarsingFactor;

            int numberElementsInnerDomain = std::floor(_innerDomainSize / _innerMeshSize);
            int numberElementsOuterDomainOneSide = (_numberElements - numberElementsInnerDomain) / 2;
            double domainCenter = (_domainStart + _domainEnd) / 2;
            double innerDomainStart = domainCenter - (_innerDomainSize / 2.0);
            double innerDomainEnd = domainCenter + (_innerDomainSize / 2.0);
            std::vector<double> linearNodes(_numberElements + 1,
                                            0.0);

            int innerDomainStartIdx = numberElementsOuterDomainOneSide;
            int innerDomainEndIdx = numberElementsOuterDomainOneSide + numberElementsInnerDomain;

            int cnt = 0;
            for (int i = innerDomainStartIdx; i <= innerDomainEndIdx; ++i) {
                linearNodes[i] = innerDomainStart + _innerMeshSize * cnt;
                cnt++;
            }
            double geometric_ratio = 0.0;
            for (int i = 1; i <= numberElementsOuterDomainOneSide; ++i) {
                geometric_ratio += std::pow(enlargeCoefficient,
                                            i);
            }
            cnt = 0;
            double outerMeshSize = ((_domainEnd - _domainStart) - _innerDomainSize) / 2.0 / geometric_ratio;
            for (int i = 1; i <= numberElementsOuterDomainOneSide; ++i) {
                outerMeshSize = outerMeshSize * enlargeCoefficient;
                linearNodes[innerDomainStartIdx - i] = linearNodes[innerDomainStartIdx - i + 1] - outerMeshSize;
                linearNodes[innerDomainEndIdx + i] = linearNodes[innerDomainEndIdx + i - 1] + outerMeshSize;
            }

            _numberNodes = _numberElements * (_numberNodesPerElement - 1) + 1;
            _globalNodalCoord = std::vector<double>(_numberNodes,
                                                    0.0);

            for (int i = 0; i < linearNodes.size() - 1; ++i) {
                std::vector<double> newNodes;
                linspace(linearNodes[i],
                         linearNodes[i + 1],
                         _numberNodesPerElement,
                         newNodes);
                newNodes.pop_back();
                for (int j = 0; j != newNodes.size(); ++j) {
                    _globalNodalCoord[i * (_numberNodesPerElement - 1) + j] = newNodes[j];
                }
            }
            _globalNodalCoord[_numberNodes - 1] = linearNodes.back();
        } else {
            const std::string message("Only support uniform mesh now.");
            throw std::logic_error(message);
        }
    } catch (std::exception &e) {
        std::cerr << e.what();
        std::terminate();
    }

}

//  The elementConnectivity matrix contains:
//      row : element id
//      col : node id in the element
//      element value : global id_t
//  The nodeToElementMap is a multimap with:
//      key : global node id
//      value : corresponding elements
//      eg : Consider a quadratic mesh with 3 elements
//          x-o-x-o-x-o-x
//          0 1 2 3 4 5 6
//          in nodeToElementMap global node 4 corresponds to element 1 and 2
void FEM::generateConnectivity() {
    for (int i = 0; i != _numberElements; ++i) {
        std::vector<int> temp;
        for (int j = 0; j != _numberNodesPerElement; ++j) {
            temp.emplace_back(i * (_numberNodesPerElement - 1) + j);
        }
        _elementConnectivity.emplace_back(temp);
    }
    // USAGE OF MULTIMAP
    //    auto iter = nodeToEleMap.lower_bound(6);
    //    auto endIter = nodeToEleMap.upper_bound(6);
    //    while(iter != endIter) {
    //        cout << iter->second << " ";
    //        ++iter;
    //    }
    for (int i = 0; i != _numberElements; ++i) {
        for (int j = 0; j != _numberNodesPerElement; ++j) {
            int first = i * (_numberNodesPerElement - 1) + j;
            int second = i;
            _nodeToElementMap.insert(std::make_pair(first,
                                                    second));
        }
    }
}

//  To evaluate the shape function at each quad point
//  The result matrix looks like :
//  N1(quad 1) N1(quad 2) ... N1(quad N)
//  N2(quad 1) N2(quad 2) ... N2(quad N)
//     ...        ...     ...    ...
//  Nn(quad 1) Nn(quad 2) ... Nn(quad N)
void FEM::generateShapeFunction() {
    _numberQuadPointsPerElement = _quadRule.getNumberQuadPoints();
    const std::vector<double> &quadPoints = _quadRule.getQuadraturePoints();
    std::vector<double> nodalCoord(_numberNodesPerElement,
                                   0);
    _shapeFunctionAtQuadPoints = std::vector<std::vector<double> >(_numberNodesPerElement,
                                                                   std::vector<double>(_numberQuadPointsPerElement,
                                                                                       1.0));

    for (int i = 0; i != _numberNodesPerElement; ++i) {
        nodalCoord[i] = -1.0 + i * (2.0 / (_numberNodesPerElement - 1));
    }
//    for (int i = 0; i != _numberNodesPerElement; ++i) {
//        double xi = nodalCoord[i];
//        std::vector<double> shapeFunction(1, 1.0);
//        for (int j = 0; j != _numberNodesPerElement; ++j) {
//            if (i != j) {
//                double xj = nodalCoord[j];
//                std::vector<double> temp;
//                temp.emplace_back(-xj / (xi - xj));
//                temp.emplace_back(1.0 / (xi - xj));
//                shapeFunction = poly_multiply(shapeFunction, temp);
//            }
//        }
//        for (int k = 0; k != _numberQuadPointsPerElement; ++k) {
//            _shapeFunctionAtQuadPoints[i][k] = poly_eval(shapeFunction, quadPoints[k]);
//        }
//    }
    for (int i = 0; i < _numberNodesPerElement; ++i) {
        for (int j = 0; j < _numberQuadPointsPerElement; ++j) {
            for (int k = 0; k < _numberNodesPerElement; ++k) {
                if (k != i) {
                    _shapeFunctionAtQuadPoints[i][j] = _shapeFunctionAtQuadPoints[i][j] *
                                                       ((quadPoints[j] - nodalCoord[k]) /
                                                        (nodalCoord[i] - nodalCoord[k]));
                }
            }
        }
    }

    // this variable is a temporary remedy for the function computeQuadValuesFromNodalValues
    _shapeFunctionAtQuadPointsForProjection = std::vector<std::vector<double> >(_numberQuadPointsPerElement,
                                                                                std::vector<double>(_numberNodesPerElement));
    for (int i = 0; i < _shapeFunctionAtQuadPoints.size(); ++i) {
        for (int j = 0; j < _shapeFunctionAtQuadPoints[i].size(); ++j) {
            _shapeFunctionAtQuadPointsForProjection[j][i] = _shapeFunctionAtQuadPoints[i][j];
        }
    }
    flattened_shape_function_at_quadpoints = std::vector<double>(_numberNodesPerElement * _numberQuadPointsPerElement);
    std::vector<double>::iterator flattend_iter = flattened_shape_function_at_quadpoints.begin();
    for (int i = 0; i < _shapeFunctionAtQuadPoints.size(); ++i) {
        flattend_iter =
                std::copy(_shapeFunctionAtQuadPoints[i].begin(),
                          _shapeFunctionAtQuadPoints[i].end(),
                          flattend_iter);
    }
//    std::cout << "shapeFunctionAtQuadPoints: " << std::endl << std::setprecision(17);
//    for(int i = 0; i < _shapeFunctionAtQuadPoints.size(); ++i){
//        for(int j = 0; j < _shapeFunctionAtQuadPoints[i].size(); ++j){
//            std::cout << _shapeFunctionAtQuadPoints[i][j] << " ";
//        }
//        std::cout << std::endl;
//    }

}

void FEM::generateShapeFunctionGradient() {
    const std::vector<double> &quadPoints = _quadRule.getQuadraturePoints();
    std::vector<double> nodalCoord(_numberNodesPerElement,
                                   0);
    _shapeFunctionDerivativeAtQuadPoints = std::vector<std::vector<double> >(_numberNodesPerElement,
                                                                             std::vector<double>(
                                                                                     _numberQuadPointsPerElement,
                                                                                     1.0));
    //C++11 feature
    //_shapeFunctionDerivativeAtQuadPoints.shrink_to_fit();
    for (int i = 0; i != _numberNodesPerElement; ++i) {
        nodalCoord[i] = -1.0 + i * (2.0 / (_numberNodesPerElement - 1));
    }

    for (int i = 0; i != _numberNodesPerElement; ++i) {
        double xi = nodalCoord[i];
        std::vector<double> shapeFunction(1,
                                          1.0);
        std::vector<double> shapeFunctionDerivative;
        for (int j = 0; j != _numberNodesPerElement; ++j) {
            if (i != j) {
                double xj = nodalCoord[j];
                std::vector<double> temp;
                temp.emplace_back(-xj / (xi - xj));
                temp.emplace_back(1.0 / (xi - xj));
                shapeFunction = poly_multiply(shapeFunction,
                                              temp);
            }
            shapeFunctionDerivative = poly_derivative(shapeFunction);
        }
        for (int k = 0; k != _numberQuadPointsPerElement; ++k) {
            _shapeFunctionDerivativeAtQuadPoints[i][k] = poly_eval(shapeFunctionDerivative,
                                                                   quadPoints[k]);
        }
    }

    flattened_shape_function_derivative_at_quadpoints =
            std::vector<double>(_numberNodesPerElement * _numberQuadPointsPerElement);
    std::vector<double>::iterator flattend_iter = flattened_shape_function_derivative_at_quadpoints.begin();
    for (int i = 0; i < _shapeFunctionDerivativeAtQuadPoints.size(); ++i) {
        flattend_iter = std::copy(_shapeFunctionDerivativeAtQuadPoints[i].begin(),
                                  _shapeFunctionDerivativeAtQuadPoints[i].end(),
                                  flattend_iter);
    }
}

/**
 * @brief given nodal values, interpolate them to quad points
 *
 * @param NodalVal nodal value vectors of vector
 *
 * @param QuadVal output quad vectors
 */
void FEM::interpolateNodalToQuadWithinElement(const std::vector<double> &NodalVal,
                                              std::vector<double> &QuadVal) const {
    MatDotVec(_shapeFunctionAtQuadPoints,
              NodalVal,
              QuadVal);
}

void FEM::interpolateNodalDerivativeToQuadWithinElement(const std::vector<double> &NodalVal,
                                                        std::vector<double> &QuadVal) const {
    MatDotVec(_shapeFunctionDerivativeAtQuadPoints,
              NodalVal,
              QuadVal);
}

//UNIT TEST NEEDED
void FEM::generateQuadPointData() {
    std::vector<double> weight = _quadRule.getQuadratureWeights();
    for (int e = 0; e != _numberElements; ++e) {
        //  retrieve the nodal coordination within the element
        std::vector<double> localNodalCoord;
        getLocalNodalCoordinates(e,
                                 localNodalCoord);
        //  compute Jacobian at quad points
        double jacobianValue = (localNodalCoord.back() - localNodalCoord.front()) / 2.0;
        //  compute spatial coordinates of quad points in physical domain
        std::vector<double> elemPositionQuadPointValues;
        interpolateNodalToQuadWithinElement(localNodalCoord,
                                            elemPositionQuadPointValues);
        for (int q = 0; q != _numberQuadPointsPerElement; ++q) {
            _jacobQuadPointValues.emplace_back(jacobianValue);
            _invJacobQuadPointValues.emplace_back(1.0 / jacobianValue);
            _positionQuadPointValues.emplace_back(elemPositionQuadPointValues[q]);
            _weightQuadPointValues.emplace_back(weight[q]);
            _jacob_times_wieght_quad_values.emplace_back(jacobianValue * weight[q]);
            _invjacob_times_wieght_quad_values.emplace_back(weight[q] / jacobianValue);
        }

    }
}

void
FEM::
computeShapeFunctionIntegralData(std::vector<std::vector<std::vector<double> > > &shapeFunctionOverlapIntegral) {

    shapeFunctionOverlapIntegral = std::vector<std::vector<std::vector<double> > >(_numberElements,
                                                                                   std::vector<std::vector<double> >(
                                                                                           _numberNodesPerElement,
                                                                                           std::vector<double>(
                                                                                                   _numberNodesPerElement,
                                                                                                   0.0)));
    shape_function_overlap_quad_values = std::vector<std::vector<std::vector<double>>>(_numberNodesPerElement,
                                                                                       std::vector<std::vector<double>>(
                                                                                               _numberNodesPerElement,
                                                                                               std::vector<double>(
                                                                                                       _numberQuadPointsPerElement)));
    for (int iNode = 0; iNode != _numberNodesPerElement; ++iNode) {
        for (int jNode = 0; jNode != _numberNodesPerElement; ++jNode) {
            std::transform(_shapeFunctionAtQuadPoints[iNode].begin(),
                           _shapeFunctionAtQuadPoints[iNode].end(),
                           _shapeFunctionAtQuadPoints[jNode].begin(),
                           shape_function_overlap_quad_values[iNode][jNode].begin(),
                           std::multiplies<double>());
        }
    }

    for (int ele = 0; ele != _numberElements; ++ele) {
        int offset = ele * _numberQuadPointsPerElement;
        for (int iNode = 0; iNode != _numberNodesPerElement; ++iNode) {
            for (int jNode = 0; jNode != _numberNodesPerElement; ++jNode) {
                shapeFunctionOverlapIntegral[ele][iNode][jNode] = ddot_(_numberQuadPointsPerElement,
                                                                        shape_function_overlap_quad_values[iNode][jNode].data(),
                                                                        1,
                                                                        _jacob_times_wieght_quad_values.data() + offset,
                                                                        1);
            }
        }
    }
}

void
FEM::
computeShapeFunctionGradientIntegralData(
        std::vector<std::vector<std::vector<double> > > &shapeFunctionGradientIntegral) {

    shapeFunctionGradientIntegral = std::vector<std::vector<std::vector<double> > >(_numberElements,
                                                                                    std::vector<std::vector<double> >(
                                                                                            _numberNodesPerElement,
                                                                                            std::vector<double>(
                                                                                                    _numberNodesPerElement)));
    shape_function_gradient_overlap_quad_values = std::vector<std::vector<std::vector<double>>>(_numberNodesPerElement,
                                                                                                std::vector<std::vector<
                                                                                                        double>>(
                                                                                                        _numberNodesPerElement,
                                                                                                        std::vector<double>(
                                                                                                                _numberQuadPointsPerElement)));
    for (int iNode = 0; iNode != _numberNodesPerElement; ++iNode) {
        for (int jNode = 0; jNode != _numberNodesPerElement; ++jNode) {
            std::transform(_shapeFunctionDerivativeAtQuadPoints[iNode].begin(),
                           _shapeFunctionDerivativeAtQuadPoints[iNode].end(),
                           _shapeFunctionDerivativeAtQuadPoints[jNode].begin(),
                           shape_function_gradient_overlap_quad_values[iNode][jNode].begin(),
                           std::multiplies<double>());
        }
    }
    for (int ele = 0; ele != _numberElements; ++ele) {
        int offset = ele * _numberQuadPointsPerElement;
        for (int iNode = 0; iNode != _numberNodesPerElement; ++iNode) {
            for (int jNode = 0; jNode != _numberNodesPerElement; ++jNode) {
                shapeFunctionGradientIntegral[ele][iNode][jNode] = ddot_(_numberQuadPointsPerElement,
                                                                         shape_function_gradient_overlap_quad_values[iNode][jNode].data(),
                                                                         1,
                                                                         _invjacob_times_wieght_quad_values.data() +
                                                                         offset,
                                                                         1);
            }
        }
    }
}

void
FEM::computeNodalExternalFunctionShapeFunctionOverlapIntegral(const std::vector<double> &nodalValues,
                                                              std::vector<std::vector<std::vector<double> > > &funcShpfuncOverlapIntegral) {

    assert(nodalValues.size() == getTotalNumberNodes());

    funcShpfuncOverlapIntegral = std::vector<std::vector<std::vector<double> > >(_numberElements,
                                                                                 std::vector<std::vector<double> >(
                                                                                         _numberNodesPerElement,
                                                                                         std::vector<double>(
                                                                                                 _numberNodesPerElement)));

    std::vector<double> quadVal, diffQuadVal;
    computeFieldAndDiffFieldAtAllQuadPoints(nodalValues,
                                            quadVal,
                                            diffQuadVal);

    std::vector<double> tempShp(_numberQuadPointsPerElement);
    for (int ele = 0; ele != _numberElements; ++ele) {
        int offset = ele * _numberQuadPointsPerElement;
        for (int iNode = 0; iNode != _numberNodesPerElement; ++iNode) {
            for (int jNode = 0; jNode != _numberNodesPerElement; ++jNode) {
                std::transform(_shapeFunctionAtQuadPoints[iNode].begin(),
                               _shapeFunctionAtQuadPoints[iNode].end(),
                               _shapeFunctionAtQuadPoints[jNode].begin(),
                               tempShp.begin(),
                               std::multiplies<double>());
                std::transform(tempShp.begin(),
                               tempShp.end(),
                               quadVal.begin() + offset,
                               tempShp.begin(),
                               std::multiplies<double>());
                std::transform(tempShp.begin(),
                               tempShp.end(),
                               _weightQuadPointValues.begin() + offset,
                               tempShp.begin(),
                               std::multiplies<double>());
                std::transform(tempShp.begin(),
                               tempShp.end(),
                               _jacobQuadPointValues.begin() + offset,
                               tempShp.begin(),
                               std::multiplies<double>());
                funcShpfuncOverlapIntegral[ele][iNode][jNode] = std::accumulate(tempShp.begin(),
                                                                                tempShp.end(),
                                                                                0.0);
            }
        }
    }
}

void FEM::computeQuadExternalFunctionShapeFunctionOverlapIntegral(const std::vector<double> &quadValues,
                                                                  std::vector<std::vector<std::vector<double> > > &funcShpfuncOverlapIntegral) const {
    assert(quadValues.size() == getTotalNumberQuadPoints());

    funcShpfuncOverlapIntegral = std::vector<std::vector<std::vector<double> > >(_numberElements,
                                                                                 std::vector<std::vector<double> >(
                                                                                         _numberNodesPerElement,
                                                                                         std::vector<double>(
                                                                                                 _numberNodesPerElement)));

    std::vector<double> temp(_numberQuadPointsPerElement);
    for (int ele = 0; ele != _numberElements; ++ele) {
        int offset = ele * _numberQuadPointsPerElement;
        for (int iNode = 0; iNode != _numberNodesPerElement; ++iNode) {
            for (int jNode = 0; jNode != _numberNodesPerElement; ++jNode) {
                std::transform(shape_function_overlap_quad_values[iNode][jNode].begin(),
                               shape_function_overlap_quad_values[iNode][jNode].end(),
                               quadValues.begin() + offset,
                               temp.begin(),
                               std::multiplies<double>());
                funcShpfuncOverlapIntegral[ele][iNode][jNode] = ddot_(_numberQuadPointsPerElement,
                                                                      temp.data(),
                                                                      1,
                                                                      _jacob_times_wieght_quad_values.data() + offset,
                                                                      1);
            }
        }
    }
}

FEM::FEM(int numberElements,
         std::string quadRule,
         int numNodesPerElement,
         double domainStart,
         double domainEnd,
         double innerDomainSize,
         double innerMeshSize,
         std::string meshType,
         bool electroFlag,
         double coarsingFactor) :
        _numberElements(numberElements),
        _numberNodesPerElement(numNodesPerElement),
        _domainStart(domainStart),
        _domainEnd(domainEnd),
        _innerDomainSize(innerDomainSize),
        _innerMeshSize(innerMeshSize),
        _meshType(meshType),
        _electroFlag(electroFlag),
        _quadRule(quadRule),
        _quadRuleName(quadRule),
        coarsingFactor(coarsingFactor) {
    //  initialize _numberNodes
    generateNumberNodes();
    //  initialize _nodeCoord, which is the coordinates of nodes in physical domain
    generateNodes();
    //  initialize _elementConnectivity and _nodeToElementMap
    generateConnectivity();
    //  initialize _numberQuadPointsPerElement and  _shapeFunctionAtQuadPoints which contains the shapeFunction at each quad point
    generateShapeFunction();
    //  initialize _shapeFunctionDerivativeAtQuadPoints which containss the derivatives of the shapeFunction at each quad point
    generateShapeFunctionGradient();
    //  initialize _jacobianQuadPointValues, _invJacobianQuadPointValues, _positionQuadPointValues, _weightQuadPointValues
    generateQuadPointData();
    computeShapeFunctionIntegralData(_shapeFunctionOverlapIntegral);
    computeShapeFunctionGradientIntegralData(_shapeFunctionGradientIntegral);
}

FEM::FEM(const FEM &fem)
        : _numberElements(fem._numberElements),
          _numberNodes(fem._numberNodes),
          _quadRule(fem._quadRule),
          _quadRuleName(fem._quadRuleName),
          _domainStart(fem._domainStart),
          _domainEnd(fem._domainEnd),
          _innerDomainSize(fem._innerDomainSize),
          _innerMeshSize(fem._innerMeshSize),
          _meshType(fem._meshType),
          coarsingFactor(fem.coarsingFactor),
          _electroFlag(fem._electroFlag),
          _numberNodesPerElement(fem._numberNodesPerElement),
          _globalNodalCoord(fem._globalNodalCoord),
          _shapeFunctionAtQuadPoints(fem._shapeFunctionAtQuadPoints),
          _shapeFunctionDerivativeAtQuadPoints(fem._shapeFunctionDerivativeAtQuadPoints),
          _elementConnectivity(fem._elementConnectivity),
          _nodeToElementMap(fem._nodeToElementMap),
          _numberQuadPointsPerElement(fem._numberQuadPointsPerElement),
          _jacobQuadPointValues(fem._jacobQuadPointValues),
          _invJacobQuadPointValues(fem._invJacobQuadPointValues),
          _positionQuadPointValues(fem._positionQuadPointValues),
          _weightQuadPointValues(fem._weightQuadPointValues),
          _shapeFunctionOverlapIntegral(fem._shapeFunctionOverlapIntegral),
          _shapeFunctionGradientIntegral(fem._shapeFunctionGradientIntegral) {}

FEM::FEM(const FEM &fem,
         const int num_outer_mesh,
         double coarsing_factor,
         const double outer_boundary_start,
         const double outer_boundary_end) {

    if (num_outer_mesh % 2 != 0) {
        std::cerr << "outer mesh should be even." << std::endl;
        std::terminate();
    }

    _domainStart = outer_boundary_start;
    _domainEnd = outer_boundary_end;
    _numberElements = num_outer_mesh + fem._numberElements;
    _numberNodesPerElement = fem._numberNodesPerElement;
    _innerMeshSize = fem._innerMeshSize;
    _electroFlag = fem._electroFlag;
    _quadRule = fem._quadRule;
    _quadRuleName = fem._quadRuleName;
    coarsingFactor = coarsing_factor;

    _meshType = "hartree_adaptive";

    // generate _numberNodes
    _numberNodes = (_numberNodesPerElement - 1) * _numberElements + 1;

    // generate nodes
    int inner_domain_start_idx = (_numberNodesPerElement - 1) * num_outer_mesh / 2;
    int inner_domain_end_idx = inner_domain_start_idx + fem.getTotalNumberNodes() - 1;
    const std::vector<double> &inner_nodes = fem.getGlobalNodalCoord();
    _globalNodalCoord = std::vector<double>(_numberNodes,
                                            0.0);
    std::copy(inner_nodes.begin(),
              inner_nodes.end(),
              _globalNodalCoord.begin() + inner_domain_start_idx);
    double
            outer_spacing =
            ((outer_boundary_end - outer_boundary_start) - (inner_nodes.back() - inner_nodes.front())) / 2.0;
    int num_outer_mesh_single_side = num_outer_mesh / 2;
    double initial_outer_mesh_size_factor = 1.0;
    for (int i = 1; i < num_outer_mesh_single_side; ++i) {
        initial_outer_mesh_size_factor = initial_outer_mesh_size_factor + std::pow(coarsing_factor,
                                                                                   i);
    }
    double outer_mesh_size = outer_spacing / initial_outer_mesh_size_factor;
    for (int i = 0; i < num_outer_mesh_single_side; ++i) {
        for (int j = 0; j < _numberNodesPerElement - 1; ++j) {
            _globalNodalCoord[inner_domain_end_idx + i * (_numberNodesPerElement - 1) + j + 1] =
                    _globalNodalCoord[inner_domain_end_idx + i * (_numberNodesPerElement - 1) + j]
                    + outer_mesh_size / (_numberNodesPerElement - 1);
            _globalNodalCoord[inner_domain_start_idx - i * (_numberNodesPerElement - 1) - j - 1] =
                    _globalNodalCoord[inner_domain_start_idx - i * (_numberNodesPerElement - 1) - j]
                    - outer_mesh_size / (_numberNodesPerElement - 1);
        }
        outer_mesh_size = outer_mesh_size * coarsing_factor;
    }

    //  initialize _elementConnectivity and _nodeToElementMap
    generateConnectivity();
    //  initialize _numberQuadPointsPerElement and  _shapeFunctionAtQuadPoints which contains the shapeFunction at each quad point
    generateShapeFunction();
    //  initialize _shapeFunctionDerivativeAtQuadPoints which containss the derivatives of the shapeFunction at each quad point
    generateShapeFunctionGradient();
    //  initialize _jacobianQuadPointValues, _invJacobianQuadPointValues, _positionQuadPointValues, _weightQuadPointValues
    generateQuadPointData();
    computeShapeFunctionIntegralData(_shapeFunctionOverlapIntegral);
    computeShapeFunctionGradientIntegralData(_shapeFunctionGradientIntegral);
}

FEM &FEM::operator=(const FEM &fem) {
    _numberElements = fem._numberElements;
    _numberNodes = fem._numberNodes;
    _quadRule = fem._quadRule;
    _quadRuleName = fem._quadRuleName,
            _domainStart = fem._domainStart;
    _domainEnd = fem._domainEnd;
    _innerDomainSize = fem._innerDomainSize;
    _innerMeshSize = fem._innerMeshSize;
    _meshType = fem._meshType;
    coarsingFactor = fem.coarsingFactor;
    _electroFlag = fem._electroFlag;
    _numberNodesPerElement = fem._numberNodesPerElement;
    _globalNodalCoord = fem._globalNodalCoord;
    _shapeFunctionAtQuadPoints = fem._shapeFunctionAtQuadPoints;
    _shapeFunctionDerivativeAtQuadPoints = fem._shapeFunctionDerivativeAtQuadPoints;
    _elementConnectivity = fem._elementConnectivity;
    _nodeToElementMap = fem._nodeToElementMap;
    _numberQuadPointsPerElement = fem._numberQuadPointsPerElement;
    _jacobQuadPointValues = fem._jacobQuadPointValues;
    _invJacobQuadPointValues = fem._invJacobQuadPointValues;
    _positionQuadPointValues = fem._positionQuadPointValues;
    _weightQuadPointValues = fem._weightQuadPointValues;
    _shapeFunctionOverlapIntegral = fem._shapeFunctionOverlapIntegral;
    _shapeFunctionGradientIntegral = fem._shapeFunctionGradientIntegral;
    return *this;
}

//UNIT TEST NEEDED
void FEM::computeFieldAndDiffFieldAtAllQuadPoints(const std::vector<double> &NodalVal,
                                                  std::vector<double> &quadVal,
                                                  std::vector<double> &diffQuadVal) const {
    computeFieldAtAllQuadPoints(NodalVal,
                                quadVal);
    computeDiffFieldAtAllQuadPoints(NodalVal,
                                    diffQuadVal);
}

void FEM::computeFieldAtAllQuadPoints(const std::vector<double> &NodalVal,
                                      std::vector<double> &quadVal) const {
    quadVal = std::vector<double>(_numberElements * _numberQuadPointsPerElement,
                                  0.0);
    const double *nodal_val_data = NodalVal.data();
    double *quad_val_data = quadVal.data();
    for (int i = 0; i < _numberElements; ++i) {
        dgemv_('N',
               _numberQuadPointsPerElement,
               _numberNodesPerElement,
               1.0,
               flattened_shape_function_at_quadpoints.data(),
               _numberQuadPointsPerElement,
               nodal_val_data + i * (_numberNodesPerElement - 1),
               1,
               0.0,
               quad_val_data + i * _numberQuadPointsPerElement,
               1);
    }
}

void FEM::computeDiffFieldAtAllQuadPoints(const std::vector<double> &NodalVal,
                                          std::vector<double> &diffQuadVal) const {
    diffQuadVal = std::vector<double>(_numberElements * _numberQuadPointsPerElement,
                                      0.0);
    const double *nodal_val_data = NodalVal.data();
    double *diffquad_val_data = diffQuadVal.data();
    for (int i = 0; i < _numberElements; ++i) {
        dgemv_('N',
               _numberQuadPointsPerElement,
               _numberNodesPerElement,
               1.0,
               flattened_shape_function_derivative_at_quadpoints.data(),
               _numberQuadPointsPerElement,
               nodal_val_data + i * (_numberNodesPerElement - 1),
               1,
               0.0,
               diffquad_val_data + i * _numberQuadPointsPerElement,
               1);
    }
}

/**
  *@brief given nodal function values at nodal points, integrate over domain
  *
  *@param nodalFuncVal function values at nodal points
  *
  *@return integrated result
  */
double FEM::integrate_by_nodal_values(const std::vector<double> &nodalFuncVal) const {
    double result = 0.0;
    std::vector<double> quadFuncVal;//, quadDiffFuncVal;
    computeFieldAtAllQuadPoints(nodalFuncVal,
                                quadFuncVal);
    //computeFieldAndDiffFieldAtAllQuadPoints(nodalFuncVal, quadFuncVal, quadDiffFuncVal);
    result = ddot_(_numberQuadPointsPerElement * _numberElements,
                   quadFuncVal.data(),
                   1,
                   _jacob_times_wieght_quad_values.data(),
                   1);
    return result;
}

/**
  *@brief given function values at quadrature points, integrate over domain
  *
  * to evaluate the operation like \f$ \int f(x)dx \f$
  *
  *@param quadFuncVal function values at quadrature points
  *
  *@return integrated result
  */
double FEM::integrate_by_quad_values(const std::vector<double> &quadFuncVal) const {
    double result = 0.0;
    result = ddot_(_numberQuadPointsPerElement * _numberElements,
                   quadFuncVal.data(),
                   1,
                   _jacob_times_wieght_quad_values.data(),
                   1);
    return result;
}

/**
  *@brief given function values at quadrature points, integrate some specific functions including the gradient over domain
  *
  * for example, to evaluate \f$ \int (\frac{d}{dx}\Psi_x(x))^2 dx = \int (\frac{d}{d \xi}\Psi_x(\xi))^2 J^{-1} d \xi \f$
  *
  *@param quadFuncVal function values at quadrature points
  *
  *@return integrated result
  */
double FEM::integrate_inv_by_quad_values(const std::vector<double> &quadFuncVal) const {
    double result = 0.0;
    result = ddot_(_numberQuadPointsPerElement * _numberElements,
                   quadFuncVal.data(),
                   1,
                   _invjacob_times_wieght_quad_values.data(),
                   1);
    return result;
}

int FEM::getNumberElements() const {
    return _numberElements;
}

int FEM::getTotalNumberNodes() const {
    return _numberNodes;
}

int FEM::getNumberNodesPerElement() const {
    return _numberNodesPerElement;
}

void FEM::getLocalNodalCoordinates(const int &elementId,
                                   std::vector<double> &localNodalCoordinate) {
    localNodalCoordinate = std::vector<double>(_elementConnectivity[elementId].size(),
                                               0.0);
    for (int i = 0; i != _elementConnectivity[elementId].size(); ++i) {
        int nodeId = _elementConnectivity[elementId][i];
        localNodalCoordinate[i] = _globalNodalCoord[nodeId];
    }
}

const std::vector<double> &FEM::getGlobalNodalCoord() const {
    return _globalNodalCoord;
}

const std::vector<std::vector<double> > &FEM::getShapeFunctionAtQuadPoints() const {
    return _shapeFunctionAtQuadPoints;
}

const std::vector<std::vector<double> > &FEM::getShapeFunctionDerivativeAtQuadPoints() const {
    return _shapeFunctionDerivativeAtQuadPoints;
}

const std::vector<std::vector<int> > &FEM::getElementConnectivity() const {
    return _elementConnectivity;
}

const std::multimap<int, int> &FEM::getNodeToElementMap() const {
    return _nodeToElementMap;
}

int FEM::getNumberQuadPointsPerElement() const {
    return _numberQuadPointsPerElement;
}

const std::vector<double> &FEM::getJacobQuadPointValues() const {
    return _jacobQuadPointValues;
}

const std::vector<double> &FEM::getInvJacobQuadPointValues() const {
    return _invJacobQuadPointValues;
}

const std::vector<double> &FEM::getPositionQuadPointValues() const {
    return _positionQuadPointValues;
}

const std::vector<double> &FEM::getWeightQuadPointValues() const {
    return _weightQuadPointValues;
}

const std::vector<std::vector<std::vector<double> > > &FEM::getShapeFunctionOverlapIntegral() const {
    return _shapeFunctionOverlapIntegral;
}

const std::vector<std::vector<std::vector<double> > > &FEM::getShapeFunctionGradientIntegral() const {
    return _shapeFunctionGradientIntegral;
}

int FEM::getTotalNumberQuadPoints() const {
    return _numberQuadPointsPerElement * _numberElements;
}

const QuadratureRule &FEM::get_quadRule() const {
    return _quadRule;
}

double FEM::get_domainStart() const {
    return _domainStart;
}

double FEM::get_domainEnd() const {
    return _domainEnd;
}

double FEM::get_innerDomainSize() const {
    return _innerDomainSize;
}

double FEM::get_innerMeshSize() const {
    return _innerMeshSize;
}

const std::string &FEM::get_meshType() const {
    return _meshType;
}

bool FEM::is_electroFlag() const {
    return _electroFlag;
}

int FEM::get_numberNodesPerElement() const {
    return _numberNodesPerElement;
}

double FEM::getCoarsingFactor() const {
    return coarsingFactor;
}

const std::string &FEM::get_quadRuleName() const {
    return _quadRuleName;
}

void FEM::computeQuadValuesFromNodalValues(const std::vector<double> &nodalValues,
                                           std::vector<double> &quadValues) const {
    quadValues = std::vector<double>(_numberQuadPointsPerElement * _numberElements,
                                     0.0);
    const double *elementalNodesStart = nodalValues.data();
    int inc = 1;
    for (int ele = 0; ele < _numberElements; ++ele) {
        for (int q = 0; q < _numberQuadPointsPerElement; ++q) {
            quadValues[q + ele * _numberQuadPointsPerElement] = ddot_(_numberNodesPerElement,
                                                                      _shapeFunctionAtQuadPointsForProjection[q].data(),
                                                                      inc,
                                                                      elementalNodesStart,
                                                                      inc);
        }
        elementalNodesStart = elementalNodesStart + (_numberNodesPerElement - 1);
    }
}
