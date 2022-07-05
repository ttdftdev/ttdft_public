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

#ifndef TUCKER_TENSOR_KSDFT_FUNCTIONALRAYLEIGHQUOTIENTSEPERABLENONLOCAL_H
#define TUCKER_TENSOR_KSDFT_FUNCTIONALRAYLEIGHQUOTIENTSEPERABLENONLOCAL_H

#include "FunctionalRayleighQuotientSeperable.h"
#include "../../atoms/NonLocalMapManager.h"
#include "../../atoms/NonLocalPSPData.h"

class FunctionalRayleighQuotientSeperableNonLocal : public FunctionalRayleighQuotientSeperable {
public:
    FunctionalRayleighQuotientSeperableNonLocal(const FEM &_femX,
                                                const FEM &_femY,
                                                const FEM &_femZ,
                                                const TuckerMPI::TuckerTensor &_tuckerDecomposedVeffMPI,
                                                const FEM &femNonLocX,
                                                const FEM &femNonLocY,
                                                const FEM &femNonLocZ,
                                                const std::vector<std::shared_ptr<NonLocalMapManager>> &nonLocalMapManager,
                                                const std::vector<std::shared_ptr<NonLocalPSPData>> &nonLocalPSPData);

    void computeVectorizedForce(const std::vector<double> &nodalFieldsX,
                                const std::vector<double> &nodalFieldsY,
                                const std::vector<double> &nodalFieldsZ,
                                double lagrangeMultiplier,
                                std::vector<double> &F) override;

    void generateHamiltonianGenericPotential(const std::vector<double> &nodalFieldsX,
                                             const std::vector<double> &nodalFieldsY,
                                             const std::vector<double> &nodalFieldsZ,
                                             Mat &Hx,
                                             Mat &Hy,
                                             Mat &Hz,
                                             Mat &Mx,
                                             Mat &My,
                                             Mat &Mz) override;

protected:
    const FEM &femNonLocX;
    const FEM &femNonLocY;
    const FEM &femNonLocZ;

    const std::vector<std::shared_ptr<NonLocalMapManager>> &nonLocalMapManager;
    const std::vector<std::shared_ptr<NonLocalPSPData>> &nonLocalPSPData;

    enum Cartesian {
        x = 1, y = 2, z = 3
    };

    void
    computeFieldsAtGivenPointsFullGrid(const std::vector<double> &nodalFieldsX,
                                       const std::vector<double> &nodalFieldsY,
                                       const std::vector<double> &nodalFieldsZ,
                                       const FEM &femX,
                                       const FEM &femY,
                                       const FEM &femZ,
                                       const FEM &femNonLocX,
                                       const FEM &femNonLocY,
                                       const FEM &femNonLocZ,
                                       const NonLocalMapManager &nonLocalMapManager,
                                       std::vector<std::vector<double> > &psi_x_nonloc,
                                       std::vector<std::vector<double> > &psi_y_nonloc,
                                       std::vector<std::vector<double> > &psi_z_nonloc);

    void computeSeparableNonLocalPotentialsUsingTucker(const std::vector<std::vector<double> > &psiNonLocQuadValuesX,
                                                       const std::vector<std::vector<double> > &psiNonLocQuadValuesY,
                                                       const std::vector<std::vector<double> > &psiNonLocQuadValuesZ,
                                                       const FEM &femNonLocX,
                                                       const FEM &femNonLocY,
                                                       const FEM &femNonLocZ,
                                                       const NonLocalPSPData &nonLocalPSPData,
                                                       std::vector<std::vector<std::vector<double> > > &nonLocNodalVx,
                                                       std::vector<std::vector<std::vector<double> > > &nonLocNodalVy,
                                                       std::vector<std::vector<std::vector<double> > > &nonLocNodalVz);

    void computeFieldsAtGivenPointsNonLocGrid(const std::vector<std::vector<std::vector<double> > > &nonLocNodalVx,
                                              const std::vector<std::vector<std::vector<double> > > &nonLocNodalVy,
                                              const std::vector<std::vector<std::vector<double> > > &nonLocNodalVz,
                                              const FEM &femX,
                                              const FEM &femY,
                                              const FEM &femZ,
                                              const FEM &femNonLocX,
                                              const FEM &femNonLocY,
                                              const FEM &femNonLocZ,
                                              const NonLocalPSPData &nonLocalPSPData,
                                              const NonLocalMapManager &nonLocalMapManager,
                                              std::vector<std::vector<std::vector<double> > > &nonLocVx,
                                              std::vector<std::vector<std::vector<double> > > &nonLocVy,
                                              std::vector<std::vector<std::vector<double> > > &nonLocVz);

    void computeNonLocalOneDimForce(const FEM &fem,
                                    const NonLocalPSPData &nonLocalPSPData,
                                    const std::vector<std::vector<std::vector<double> > > &vNloc,
                                    std::vector<std::vector<std::vector<std::vector<double> > > > &oneDimForceNloc);

    void computeCmatTranslmTimesPsi(const std::vector<std::vector<std::vector<double> > > &vNloc,
                                    const std::vector<double> &nodalFields,
                                    const FEM &fem,
                                    const NonLocalPSPData &nonLocalPSPData,
                                    std::vector<std::vector<double> > &CmatTranslmTimesPsi);

    void sumOneDimNonLocalForceOverQuadPoints(const FEM &fem,
                                              const std::vector<std::vector<std::vector<std::vector<double> > > > &oneDimForceNloc,
                                              const std::vector<std::vector<double> > &CmatTranslmTimesPsi,
                                              const NonLocalPSPData &nonLocalPSPData,
                                              std::vector<double> &summedNonLocalOneDimForce);

    void computeMatPsiIntegral(const FEM &fem,
                               const int rankNloc,
                               const int numberAtoms,
                               const int lmCount,
                               const std::vector<Tucker::Matrix *> &matNonLoc,
                               const std::vector<std::vector<double>> &psiNonLocQuadValues,
                               std::vector<std::vector<Tucker::Matrix *>> &matPsiIntegral);

//  void computeMatPsiIntegral(const FEM &fem, const int rankNloc, const int numberAtoms, const int lmCount,
//                             const std::vector<Tucker::Matrix*> &matNonLoc,
//                             const std::vector<std::vector<double>> &psiNonLocQuadValues,
//                             std::vector<std::vector<std::vector<double>>> &matPsiIntegral);
    void computeMatShapeIntegral(const FEM &femNonLoc,
                                 const int rankNloc,
                                 const int lMax,
                                 const std::vector<Tucker::Matrix *> &mat,
                                 std::vector<std::vector<std::vector<double>>> &FNLocTimesShp);

    void computeShpTimesShpTimesPsiVNloc(const int numberAtoms,
                                         const int lMax,
                                         const int rankNloc,
                                         const int numberTotalNodesX,
                                         const int numberTotalNodesY,
                                         const int numberTotalNodesZ,
                                         const std::vector<Tucker::Tensor *> &sigma,
                                         const std::vector<std::vector<std::vector<double>>> &Fx,
                                         const std::vector<std::vector<std::vector<double>>> &Fy,
                                         const std::vector<std::vector<std::vector<double>>> &Fz,
                                         const std::vector<std::vector<std::vector<double>>> &intUiPsix,
                                         const std::vector<std::vector<std::vector<double>>> &intViPsiy,
                                         const std::vector<std::vector<std::vector<double>>> &intWiPsiz,
                                         std::vector<std::vector<std::vector<std::vector<double>>>> &shpXi_shpYj_PsiZ_VNloc,
                                         std::vector<std::vector<std::vector<std::vector<double>>>> &shpXi_PsiY_shpZj_VNloc,
                                         std::vector<std::vector<std::vector<std::vector<double>>>> &PsiX_shpYi_shpZj_VNloc);

    void computePsiVNlocProduct(const int numberAtoms,
                                const int lMax,
                                const int rankNloc,
                                const std::vector<Tucker::Tensor *> &sigma,
                                const std::vector<std::vector<std::vector<double>>> &intUiPsix,
                                const std::vector<std::vector<std::vector<double>>> &intViPsiy,
                                const std::vector<std::vector<std::vector<double>>> &intWiPsiz,
                                std::vector<std::vector<double>> &PsiX_PsiY_PsiZ_VNloc);

    void computeShpTimesPsiVNlocTimesPsiVNloc(const int numberAtoms,
                                              const int lMax,
                                              const int rankNloc,
                                              const int numberTotalNodesX,
                                              const int numberTotalNodesY,
                                              const int numberTotalNodesZ,
                                              const std::vector<Tucker::Tensor *> &sigma,
                                              const std::vector<std::vector<std::vector<double>>> &Fx,
                                              const std::vector<std::vector<std::vector<double>>> &Fy,
                                              const std::vector<std::vector<std::vector<double>>> &Fz,
                                              const std::vector<std::vector<std::vector<double>>> &intUiPsix,
                                              const std::vector<std::vector<std::vector<double>>> &intViPsiy,
                                              const std::vector<std::vector<std::vector<double>>> &intWiPsiz,
                                              std::vector<std::vector<std::vector<double>>> &shpXi_PsiY_PsiZ_VNloc,
                                              std::vector<std::vector<std::vector<double>>> &PsiX_shpYi_PsiZ_VNloc,
                                              std::vector<std::vector<std::vector<double>>> &PsiX_PsiY_shpZi_VNloc);

    void computeHNloc(const int numberAtoms,
                      const int lMax,
                      const FEM &fem,
                      const double potConst,
                      const std::vector<double> &C_lm,
                      const std::vector<std::vector<std::vector<std::vector<double> > > > &vnloc,
                      Mat &H);

};

#endif //TUCKER_TENSOR_KSDFT_FUNCTIONALRAYLEIGHQUOTIENTSEPERABLENONLOCAL_H
