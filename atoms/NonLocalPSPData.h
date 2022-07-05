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

#ifndef TUCKER_TENSOR_KSDFT_NONLOCALPSPDATA_H
#define TUCKER_TENSOR_KSDFT_NONLOCALPSPDATA_H

#include <Tucker_Tensor.hpp>
#include <Tucker_Matrix.hpp>
#include "../fem/FEM.h"
#include "../tensor/Tensor3DMPI.h"
#include "../alglib/src/interpolation.h"

class NonLocalPSPData {
public:
    NonLocalPSPData(const FEM &femNonLocX,
                    const FEM &femNonLocY,
                    const FEM &femNonLocZ,
                    const int numberNonLocalAtoms,
                    const int lMax,
                    const int rankNloc,
                    const std::string &localPSPFilename,
                    const std::string &VlFilename,
                    const std::string &RlFilename);

    static void computeLocalPart(const FEM &femX,
                                 const FEM &femY,
                                 const FEM &femZ,
                                 const std::vector<std::vector<double> > &nuclei,
                                 const std::string &localPSPFilename,
                                 Tensor3DMPI &localPSP);

    int getLMax() const;

    int getRankNloc() const;

    const std::vector<Tucker::Tensor *> &getSigmaQuad() const;

    const std::vector<Tucker::Matrix *> &getUmatQuad() const;

    const std::vector<Tucker::Matrix *> &getVmatQuad() const;

    const std::vector<Tucker::Matrix *> &getWmatQuad() const;

    const std::vector<Tucker::Tensor *> &getSigmaNode() const;

    const std::vector<Tucker::Matrix *> &getUmatNode() const;

    const std::vector<Tucker::Matrix *> &getVmatNode() const;

    const std::vector<Tucker::Matrix *> &getWmatNode() const;

    const std::vector<Tucker::Matrix *> &getUmatNodeInterpolatedQuad() const;

    const std::vector<Tucker::Matrix *> &getVmatNodeInterpolatedQuad() const;

    const std::vector<Tucker::Matrix *> &getWmatNodeInterpolatedQuad() const;

    const std::vector<double> &getC_lm() const;

    int getNumberNonLocalAtoms() const;

    virtual ~NonLocalPSPData();

private:
    int lMax;
    int rankNloc;
    int numberNonLocalAtoms;

    // quad point data
    std::vector<Tucker::Tensor *> sigmaQuad;
    std::vector<Tucker::Matrix *> umatQuad;
    std::vector<Tucker::Matrix *> vmatQuad;
    std::vector<Tucker::Matrix *> wmatQuad;

    // nodal point data and interpolated quad point data
    std::vector<Tucker::Tensor *> sigmaNode;
    std::vector<Tucker::Matrix *> umatNode;
    std::vector<Tucker::Matrix *> vmatNode;
    std::vector<Tucker::Matrix *> wmatNode;
    std::vector<Tucker::Matrix *> umatNodeInterpolatedQuad;
    std::vector<Tucker::Matrix *> vmatNodeInterpolatedQuad;
    std::vector<Tucker::Matrix *> wmatNodeInterpolatedQuad;

    // some other nonlocal PSP data
    std::vector<double> C_lm;

    static void generateNodalDistance(const FEM &femX,
                                      const FEM &femY,
                                      const FEM &femZ,
                                      Tensor3DMPI &distanceNodal);

    static void generateQuadDistance(const FEM &femX,
                                     const FEM &femY,
                                     const FEM &femZ,
                                     Tensor3DMPI &distanceQuad);

    static void generateQuadDistanceFromAtom(const FEM &femX,
                                             const FEM &femY,
                                             const FEM &femZ,
                                             const double x,
                                             const double y,
                                             const double z,
                                             Tensor3DMPI &distanceQuad);

    static void createSplineObject(const std::string &filename,
                                   alglib::spline1dinterpolant &func);

    static void computeFieldFromSpline(const Tensor3DMPI &distance,
                                       alglib::spline1dinterpolant &func,
                                       Tensor3DMPI &field);

    static void projectNodeOntoQuad(const std::vector<Tucker::Matrix *> &matNode,
                                    const FEM &fem,
                                    std::vector<Tucker::Matrix *> &matQuad);

    void createNonlinearSplineObjects(const std::string &localPSPFilename,
                                      const std::string &VlFilename,
                                      const std::string &RlFilename,
                                      alglib::spline1dinterpolant &funcLocalPSP,
                                      std::vector<alglib::spline1dinterpolant> &funcDeltaVl,
                                      std::vector<alglib::spline1dinterpolant> &funcRl);

    void computePhi_lm(const std::vector<double> &x,
                       const std::vector<double> &y,
                       const std::vector<double> &z,
                       const int lMax,
                       const Tensor3DMPI &distance,
                       const std::vector<Tensor3DMPI> &R_l,
                       std::vector<Tensor3DMPI> &phi_lm);

    void computeC_lm(const int lMax,
                     const int rankNloc,
                     const FEM &femNonLocX,
                     const FEM &femNonLocY,
                     const FEM &femNonLocZ,
                     const std::vector<Tensor3DMPI> &deltaVlQuad,
                     const std::vector<Tensor3DMPI> &phi_lmQuad,
                     std::vector<double> &C_lm);

    void computeDecomposedNonlocalData(const std::vector<Tensor3DMPI> &DeltaVl,
                                       const std::vector<Tensor3DMPI> &phi_lm,
                                       const int lMax,
                                       const int rankNloc,
                                       std::vector<Tucker::Tensor *> &sigma,
                                       std::vector<Tucker::Matrix *> &umat,
                                       std::vector<Tucker::Matrix *> &vmat,
                                       std::vector<Tucker::Matrix *> &wmat);

};

#endif //TUCKER_TENSOR_KSDFT_NONLOCALPSPDATA_H
