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
#include "FunctionalRayleighQuotientSeperableNonLocal.h"
#include "../../blas_lapack/clinalg.h"

FunctionalRayleighQuotientSeperableNonLocal::FunctionalRayleighQuotientSeperableNonLocal(const FEM &_femX,
                                                                                         const FEM &_femY,
                                                                                         const FEM &_femZ,
                                                                                         const TuckerMPI::TuckerTensor &_tuckerDecomposedVeffMPI,
                                                                                         const FEM &femNonLocX,
                                                                                         const FEM &femNonLocY,
                                                                                         const FEM &femNonLocZ,
                                                                                         const std::vector<std::shared_ptr<
                                                                                                 NonLocalMapManager>> &nonLocalMapManager,
                                                                                         const std::vector<std::shared_ptr<
                                                                                                 NonLocalPSPData>> &nonLocalPSPData)
        : FunctionalRayleighQuotientSeperable(_femX,
                                              _femY,
                                              _femZ,
                                              _tuckerDecomposedVeffMPI),
          femNonLocX(femNonLocX),
          femNonLocY(femNonLocY),
          femNonLocZ(femNonLocZ),
          nonLocalMapManager(nonLocalMapManager),
          nonLocalPSPData(nonLocalPSPData) {

}

void FunctionalRayleighQuotientSeperableNonLocal::computeVectorizedForce(const std::vector<double> &nodalFieldsX,
                                                                         const std::vector<double> &nodalFieldsY,
                                                                         const std::vector<double> &nodalFieldsZ,
                                                                         double lagrangeMultiplier,
                                                                         std::vector<double> &F) {
    FunctionalRayleighQuotientSeperable::computeVectorizedForce(nodalFieldsX,
                                                                nodalFieldsY,
                                                                nodalFieldsZ,
                                                                lagrangeMultiplier,
                                                                F);

    int num_atom_type = nonLocalPSPData.size();
    std::vector<std::vector<std::vector<double>>> psiNonLocQuadValuesX(num_atom_type),
            psiNonLocQuadValuesY(num_atom_type), psiNonLocQuadValuesZ(num_atom_type);
    for (int i = 0; i < num_atom_type; ++i) {
        computeFieldsAtGivenPointsFullGrid(nodalFieldsX,
                                           nodalFieldsY,
                                           nodalFieldsZ,
                                           _femX,
                                           _femY,
                                           _femZ,
                                           femNonLocX,
                                           femNonLocY,
                                           femNonLocZ,
                                           *nonLocalMapManager[i],
                                           psiNonLocQuadValuesX[i],
                                           psiNonLocQuadValuesY[i],
                                           psiNonLocQuadValuesZ[i]);
    }

    std::vector<std::vector<std::vector<std::vector<double>>>> nonLocNodalVx(num_atom_type), nonLocNodalVy(num_atom_type),
            nonLocNodalVz(num_atom_type);
    for (int i = 0; i < num_atom_type; ++i) {
        computeSeparableNonLocalPotentialsUsingTucker(psiNonLocQuadValuesX[i],
                                                      psiNonLocQuadValuesY[i],
                                                      psiNonLocQuadValuesZ[i],
                                                      femNonLocX,
                                                      femNonLocY,
                                                      femNonLocZ,
                                                      *nonLocalPSPData[i],
                                                      nonLocNodalVx[i],
                                                      nonLocNodalVy[i],
                                                      nonLocNodalVz[i]);
    }

    std::vector<std::vector<std::vector<std::vector<double>>>> nonLocVx(num_atom_type), nonLocVy(num_atom_type),
            nonLocVz(num_atom_type);
    for (int i = 0; i < num_atom_type; ++i) {
        computeFieldsAtGivenPointsNonLocGrid(nonLocNodalVx[i],
                                             nonLocNodalVy[i],
                                             nonLocNodalVz[i],
                                             _femX,
                                             _femY,
                                             _femZ,
                                             femNonLocX,
                                             femNonLocY,
                                             femNonLocZ,
                                             *nonLocalPSPData[i],
                                             *nonLocalMapManager[i],
                                             nonLocVx[i],
                                             nonLocVy[i],
                                             nonLocVz[i]);
    }

    std::vector<std::vector<std::vector<std::vector<std::vector<double>>>>> X0Nloc(num_atom_type), Y0Nloc(num_atom_type),
            Z0Nloc(num_atom_type);
    for (int i = 0; i < num_atom_type; ++i) {
        computeNonLocalOneDimForce(_femX,
                                   *nonLocalPSPData[i],
                                   nonLocVx[i],
                                   X0Nloc[i]);
        computeNonLocalOneDimForce(_femY,
                                   *nonLocalPSPData[i],
                                   nonLocVy[i],
                                   Y0Nloc[i]);
        computeNonLocalOneDimForce(_femZ,
                                   *nonLocalPSPData[i],
                                   nonLocVz[i],
                                   Z0Nloc[i]);
    }

    std::vector<std::vector<std::vector<double>>> CmatTranslmTimesPsi_x(num_atom_type),
            CmatTranslmTimesPsi_y(num_atom_type), CmatTranslmTimesPsi_z(num_atom_type);
    for (int i = 0; i < num_atom_type; ++i) {
        computeCmatTranslmTimesPsi(nonLocVx[i],
                                   nodalFieldsX,
                                   _femX,
                                   *nonLocalPSPData[i],
                                   CmatTranslmTimesPsi_x[i]);
        computeCmatTranslmTimesPsi(nonLocVy[i],
                                   nodalFieldsY,
                                   _femY,
                                   *nonLocalPSPData[i],
                                   CmatTranslmTimesPsi_y[i]);
        computeCmatTranslmTimesPsi(nonLocVz[i],
                                   nodalFieldsZ,
                                   _femZ,
                                   *nonLocalPSPData[i],
                                   CmatTranslmTimesPsi_z[i]);
    }

    const int numberTotalNodesX = _femX.getTotalNumberNodes();
    const int numberTotalNodesY = _femY.getTotalNumberNodes();
    const int numberTotalNodesZ = _femZ.getTotalNumberNodes();
    std::vector<double> NonLocFx(numberTotalNodesX,
                                 0.0), NonLocFy(numberTotalNodesY,
                                                0.0),
            NonLocFz(numberTotalNodesZ,
                     0.0);
    for (int i = 0; i < num_atom_type; ++i) {
        std::vector<double> temp_nonLocFx, temp_nonLocFy, temp_nonLocFz;
        sumOneDimNonLocalForceOverQuadPoints(_femX,
                                             X0Nloc[i],
                                             CmatTranslmTimesPsi_x[i],
                                             *nonLocalPSPData[i],
                                             temp_nonLocFx);
        sumOneDimNonLocalForceOverQuadPoints(_femY,
                                             Y0Nloc[i],
                                             CmatTranslmTimesPsi_y[i],
                                             *nonLocalPSPData[i],
                                             temp_nonLocFy);
        sumOneDimNonLocalForceOverQuadPoints(_femZ,
                                             Z0Nloc[i],
                                             CmatTranslmTimesPsi_z[i],
                                             *nonLocalPSPData[i],
                                             temp_nonLocFz);

        std::transform(temp_nonLocFx.begin(),
                       temp_nonLocFx.end(),
                       NonLocFx.begin(),
                       NonLocFx.begin(),
                       std::plus<double>());
        std::transform(temp_nonLocFy.begin(),
                       temp_nonLocFy.end(),
                       NonLocFy.begin(),
                       NonLocFy.begin(),
                       std::plus<double>());
        std::transform(temp_nonLocFz.begin(),
                       temp_nonLocFz.end(),
                       NonLocFz.begin(),
                       NonLocFz.begin(),
                       std::plus<double>());
    }

    std::transform(NonLocFx.begin() + 1,
                   NonLocFx.end() - 1,
                   F.begin(),
                   F.begin(),
                   std::plus<double>());
    std::transform(NonLocFy.begin() + 1,
                   NonLocFy.end() - 1,
                   F.begin() + numberTotalNodesX - 2,
                   F.begin() + numberTotalNodesX - 2,
                   std::plus<double>());
    std::transform(NonLocFz.begin() + 1,
                   NonLocFz.end() - 1,
                   F.begin() + numberTotalNodesX + numberTotalNodesY - 4,
                   F.begin() + numberTotalNodesX + numberTotalNodesY - 4,
                   std::plus<double>());
}

void
FunctionalRayleighQuotientSeperableNonLocal::computeFieldsAtGivenPointsFullGrid(const std::vector<double> &nodalFieldsX,
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
                                                                                std::vector<std::vector<double> > &psi_z_nonloc) {
    int numberAtoms = nonLocalMapManager.getNumberNonLocAtoms();

    auto &elemNonLocGridToFullGridX = nonLocalMapManager.getElemNonLocGridToFullGridX();
    auto &elemNonLocGridToFullGridY = nonLocalMapManager.getElemNonLocGridToFullGridY();
    auto &elemNonLocGridToFullGridZ = nonLocalMapManager.getElemNonLocGridToFullGridZ();

    auto &shapeFunctionMatrixFullGridX = nonLocalMapManager.getShapeFunctionMatrixFullGridX();
    auto &shapeFunctionMatrixFullGridY = nonLocalMapManager.getShapeFunctionMatrixFullGridY();
    auto &shapeFunctionMatrixFullGridZ = nonLocalMapManager.getShapeFunctionMatrixFullGridZ();

    int sizeCompactSupportX = femNonLocX.getTotalNumberQuadPoints();
    int sizeCompactSupportY = femNonLocY.getTotalNumberQuadPoints();
    int sizeCompactSupportZ = femNonLocZ.getTotalNumberQuadPoints();
    psi_x_nonloc = std::vector<std::vector<double> >(numberAtoms,
                                                     std::vector<double>(sizeCompactSupportX,
                                                                         0.0));
    psi_y_nonloc = std::vector<std::vector<double> >(numberAtoms,
                                                     std::vector<double>(sizeCompactSupportY,
                                                                         0.0));
    psi_z_nonloc = std::vector<std::vector<double> >(numberAtoms,
                                                     std::vector<double>(sizeCompactSupportZ,
                                                                         0.0));

    for (int iAtom = 0; iAtom < numberAtoms; ++iAtom) {
        for (int iPoint = 0; iPoint < sizeCompactSupportX; ++iPoint) {
            int elemId = elemNonLocGridToFullGridX[iAtom][iPoint];
            const std::vector<int> &localNodesIds = femX.getElementConnectivity()[elemId];
            int n = localNodesIds.size();
            int inc = 1;
            const double *localNodalValueData = &nodalFieldsX[localNodesIds.front()];
            const double *shapeFunctionData = shapeFunctionMatrixFullGridX[iAtom][iPoint].data();
            psi_x_nonloc[iAtom][iPoint] = clinalg::ddot_(n,
                                                         localNodalValueData,
                                                         1,
                                                         shapeFunctionData,
                                                         1);
            //ddot_(&n, localNodalValueData, &inc, shapeFunctionData, &inc);

        }
    }

    for (int iAtom = 0; iAtom < numberAtoms; ++iAtom) {
        for (int iPoint = 0; iPoint < sizeCompactSupportY; ++iPoint) {
            int elemId = elemNonLocGridToFullGridY[iAtom][iPoint];
            const std::vector<int> &localNodesIds = femY.getElementConnectivity()[elemId];
            int n = localNodesIds.size();
            int inc = 1;
            const double *localNodalValueData = &nodalFieldsY[localNodesIds.front()];
            const double *shapeFunctionData = shapeFunctionMatrixFullGridY[iAtom][iPoint].data();
            psi_y_nonloc[iAtom][iPoint] = clinalg::ddot_(n,
                                                         localNodalValueData,
                                                         1,
                                                         shapeFunctionData,
                                                         1);
            //ddot_(&n, localNodalValueData, &inc, shapeFunctionData, &inc);

        }
    }

    for (int iAtom = 0; iAtom < numberAtoms; ++iAtom) {
        for (int iPoint = 0; iPoint < sizeCompactSupportZ; ++iPoint) {
            int elemId = elemNonLocGridToFullGridZ[iAtom][iPoint];
            const std::vector<int> &localNodesIds = femZ.getElementConnectivity()[elemId];
            int n = localNodesIds.size();
            int inc = 1;
            const double *localNodalValueData = &nodalFieldsZ[localNodesIds.front()];
            const double *shapeFunctionData = shapeFunctionMatrixFullGridZ[iAtom][iPoint].data();
            psi_z_nonloc[iAtom][iPoint] = clinalg::ddot_(n,
                                                         localNodalValueData,
                                                         1,
                                                         shapeFunctionData,
                                                         1);
            //ddot_(&n, localNodalValueData, &inc, shapeFunctionData, &inc);
        }
    }
}

void FunctionalRayleighQuotientSeperableNonLocal::computeSeparableNonLocalPotentialsUsingTucker(
        const std::vector<std::vector<double> > &psiNonLocQuadValuesX,
        const std::vector<std::vector<double> > &psiNonLocQuadValuesY,
        const std::vector<std::vector<double> > &psiNonLocQuadValuesZ,
        const FEM &femNonLocX,
        const FEM &femNonLocY,
        const FEM &femNonLocZ,
        const NonLocalPSPData &nonLocalPSPData,
        std::vector<std::vector<std::vector<double> > > &nonLocNodalVx,
        std::vector<std::vector<std::vector<double> > > &nonLocNodalVy,
        std::vector<std::vector<std::vector<double> > > &nonLocNodalVz) {
    // check the lmCount is fixed for now, to be generalized later

    int numberAtoms = nonLocalPSPData.getNumberNonLocalAtoms();
    int lMax = nonLocalPSPData.getLMax();
    int rankNloc = nonLocalPSPData.getRankNloc();
    int lmCount = lMax * lMax;

    nonLocNodalVx = std::vector<std::vector<std::vector<double> > >(numberAtoms,
                                                                    std::vector<std::vector<double> >(lmCount,
                                                                                                      std::vector<double>(
                                                                                                              femNonLocX.getTotalNumberNodes(),
                                                                                                              0.0)));
    nonLocNodalVy = std::vector<std::vector<std::vector<double> > >(numberAtoms,
                                                                    std::vector<std::vector<double> >(lmCount,
                                                                                                      std::vector<double>(
                                                                                                              femNonLocY.getTotalNumberNodes(),
                                                                                                              0.0)));
    nonLocNodalVz = std::vector<std::vector<std::vector<double> > >(numberAtoms,
                                                                    std::vector<std::vector<double> >(lmCount,
                                                                                                      std::vector<double>(
                                                                                                              femNonLocZ.getTotalNumberNodes(),
                                                                                                              0.0)));

    auto &sigma = nonLocalPSPData.getSigmaNode();
    auto &umatNode = nonLocalPSPData.getUmatNode();
    auto &vmatNode = nonLocalPSPData.getVmatNode();
    auto &wmatNode = nonLocalPSPData.getWmatNode();
    auto &umatQuad = nonLocalPSPData.getUmatNodeInterpolatedQuad();
    auto &vmatQuad = nonLocalPSPData.getVmatNodeInterpolatedQuad();
    auto &wmatQuad = nonLocalPSPData.getWmatNodeInterpolatedQuad();

    std::vector<std::vector<Tucker::Matrix *>> intUiPsix(numberAtoms,
                                                         std::vector<Tucker::Matrix *>(lmCount));
    std::vector<std::vector<Tucker::Matrix *>> intViPsiy(numberAtoms,
                                                         std::vector<Tucker::Matrix *>(lmCount));
    std::vector<std::vector<Tucker::Matrix *>> intWiPsiz(numberAtoms,
                                                         std::vector<Tucker::Matrix *>(lmCount));
    for (int i = 0; i < intUiPsix.size(); ++i) {
        for (int j = 0; j < intUiPsix[i].size(); ++j) {
            intUiPsix[i][j] = Tucker::MemoryManager::safe_new<Tucker::Matrix>(1,
                                                                              rankNloc);
            intUiPsix[i][j]->initialize();
        }
    }
    for (int i = 0; i < intViPsiy.size(); ++i) {
        for (int j = 0; j < intViPsiy[i].size(); ++j) {
            intViPsiy[i][j] = Tucker::MemoryManager::safe_new<Tucker::Matrix>(1,
                                                                              rankNloc);
            intViPsiy[i][j]->initialize();
        }
    }
    for (int i = 0; i < intWiPsiz.size(); ++i) {
        for (int j = 0; j < intWiPsiz[i].size(); ++j) {
            intWiPsiz[i][j] = Tucker::MemoryManager::safe_new<Tucker::Matrix>(1,
                                                                              rankNloc);
            intWiPsiz[i][j]->initialize();
        }
    }
    computeMatPsiIntegral(femNonLocX,
                          rankNloc,
                          numberAtoms,
                          lmCount,
                          nonLocalPSPData.getUmatNodeInterpolatedQuad(),
                          psiNonLocQuadValuesX,
                          intUiPsix);
    computeMatPsiIntegral(femNonLocY,
                          rankNloc,
                          numberAtoms,
                          lmCount,
                          nonLocalPSPData.getVmatNodeInterpolatedQuad(),
                          psiNonLocQuadValuesY,
                          intViPsiy);
    computeMatPsiIntegral(femNonLocZ,
                          rankNloc,
                          numberAtoms,
                          lmCount,
                          nonLocalPSPData.getWmatNodeInterpolatedQuad(),
                          psiNonLocQuadValuesZ,
                          intWiPsiz);

    int numberNodesNonLocX = femNonLocX.getTotalNumberNodes();
    int numberNodesNonLocY = femNonLocY.getTotalNumberNodes();
    int numberNodesNonLocZ = femNonLocZ.getTotalNumberNodes();

    for (int iAtom = 0; iAtom < numberAtoms; ++iAtom) {
        for (int lmcomp = 0; lmcomp < lmCount; ++lmcomp) {
            Tucker::Tensor *temp;
            Tucker::Tensor *reconstructedTensor;
            temp = sigma[lmcomp];
            reconstructedTensor = Tucker::ttm(temp,
                                              2,
                                              intWiPsiz[iAtom][lmcomp]);
            temp = reconstructedTensor;
            reconstructedTensor = Tucker::ttm(temp,
                                              1,
                                              intViPsiy[iAtom][lmcomp]);
            Tucker::MemoryManager::safe_delete(temp);
            temp = reconstructedTensor;
            reconstructedTensor = Tucker::ttm(temp,
                                              0,
                                              umatNode[lmcomp]);
            Tucker::MemoryManager::safe_delete(temp);
            std::copy(reconstructedTensor->data(),
                      reconstructedTensor->data() + reconstructedTensor->getNumElements(),
                      nonLocNodalVx[iAtom][lmcomp].begin());
            Tucker::MemoryManager::safe_delete(reconstructedTensor);
        }
    }

    for (int iAtom = 0; iAtom < numberAtoms; ++iAtom) {
        for (int lmcomp = 0; lmcomp < lmCount; ++lmcomp) {
            Tucker::Tensor *temp;
            Tucker::Tensor *reconstructedTensor;
            temp = sigma[lmcomp];
            reconstructedTensor = Tucker::ttm(temp,
                                              0,
                                              intUiPsix[iAtom][lmcomp]);
            temp = reconstructedTensor;
            reconstructedTensor = Tucker::ttm(temp,
                                              2,
                                              intWiPsiz[iAtom][lmcomp]);
            Tucker::MemoryManager::safe_delete(temp);
            temp = reconstructedTensor;
            reconstructedTensor = Tucker::ttm(temp,
                                              1,
                                              vmatNode[lmcomp]);
            Tucker::MemoryManager::safe_delete(temp);
            std::copy(reconstructedTensor->data(),
                      reconstructedTensor->data() + reconstructedTensor->getNumElements(),
                      nonLocNodalVy[iAtom][lmcomp].begin());
            Tucker::MemoryManager::safe_delete(reconstructedTensor);
        }
    }

    for (int iAtom = 0; iAtom < numberAtoms; ++iAtom) {
        for (int lmcomp = 0; lmcomp < lmCount; ++lmcomp) {
            Tucker::Tensor *temp;
            Tucker::Tensor *reconstructedTensor;
            temp = sigma[lmcomp];
            reconstructedTensor = Tucker::ttm(temp,
                                              0,
                                              intUiPsix[iAtom][lmcomp]);
            temp = reconstructedTensor;
            reconstructedTensor = Tucker::ttm(temp,
                                              1,
                                              intViPsiy[iAtom][lmcomp]);
            Tucker::MemoryManager::safe_delete(temp);
            temp = reconstructedTensor;
            reconstructedTensor = Tucker::ttm(temp,
                                              2,
                                              wmatNode[lmcomp]);
            Tucker::MemoryManager::safe_delete(temp);
            std::copy(reconstructedTensor->data(),
                      reconstructedTensor->data() + reconstructedTensor->getNumElements(),
                      nonLocNodalVz[iAtom][lmcomp].begin());
            Tucker::MemoryManager::safe_delete(reconstructedTensor);
        }
    }

    for (int i = 0; i < intUiPsix.size(); ++i) {
        for (int j = 0; j < intUiPsix[i].size(); ++j) {
            Tucker::MemoryManager::safe_delete(intUiPsix[i][j]);
        }
    }
    for (int i = 0; i < intViPsiy.size(); ++i) {
        for (int j = 0; j < intViPsiy[i].size(); ++j) {
            Tucker::MemoryManager::safe_delete(intViPsiy[i][j]);
        }
    }
    for (int i = 0; i < intWiPsiz.size(); ++i) {
        for (int j = 0; j < intWiPsiz[i].size(); ++j) {
            Tucker::MemoryManager::safe_delete(intWiPsiz[i][j]);
        }
    }
}

void FunctionalRayleighQuotientSeperableNonLocal::computeFieldsAtGivenPointsNonLocGrid(
        const std::vector<std::vector<std::vector<double> > > &nonLocNodalVx,
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
        std::vector<std::vector<std::vector<double> > > &nonLocVz) {

    int totalnumberQuadraturePointsX = femX.getTotalNumberQuadPoints();
    int totalnumberQuadraturePointsY = femY.getTotalNumberQuadPoints();
    int totalnumberQuadraturePointsZ = femZ.getTotalNumberQuadPoints();
    int sizeCompactSupportX = femNonLocX.getTotalNumberQuadPoints();
    int sizeCompactSupportY = femNonLocY.getTotalNumberQuadPoints();
    int sizeCompactSupportZ = femNonLocZ.getTotalNumberQuadPoints();

    int numberAtoms = nonLocalPSPData.getNumberNonLocalAtoms();
    int lMax = nonLocalPSPData.getLMax();
    int lmCount = lMax * lMax;
    nonLocVx = std::vector<std::vector<std::vector<double> > >(numberAtoms,
                                                               std::vector<std::vector<double> >(lmCount,
                                                                                                 std::vector<
                                                                                                         double>(
                                                                                                         totalnumberQuadraturePointsX,
                                                                                                         0.0)));
    nonLocVy = std::vector<std::vector<std::vector<double> > >(numberAtoms,
                                                               std::vector<std::vector<double> >(lmCount,
                                                                                                 std::vector<
                                                                                                         double>(
                                                                                                         totalnumberQuadraturePointsY,
                                                                                                         0.0)));
    nonLocVz = std::vector<std::vector<std::vector<double> > >(numberAtoms,
                                                               std::vector<std::vector<double> >(lmCount,
                                                                                                 std::vector<
                                                                                                         double>(
                                                                                                         totalnumberQuadraturePointsZ,
                                                                                                         0.0)));
    auto &elemFullGridToNonLocGridX = nonLocalMapManager.getElemFullGridToNonLocGridX();
    auto &elemFullGridToNonLocGridY = nonLocalMapManager.getElemFullGridToNonLocGridY();
    auto &elemFullGridToNonLocGridZ = nonLocalMapManager.getElemFullGridToNonLocGridZ();
    auto &shapeFunctionMatrixNonLocGridX = nonLocalMapManager.getShapeFunctionMatrixNonLocGridX();
    auto &shapeFunctionMatrixNonLocGridY = nonLocalMapManager.getShapeFunctionMatrixNonLocGridY();
    auto &shapeFunctionMatrixNonLocGridZ = nonLocalMapManager.getShapeFunctionMatrixNonLocGridZ();

    for (int iAtom = 0; iAtom < numberAtoms; ++iAtom) {
        for (int iPoint = 0; iPoint < totalnumberQuadraturePointsX; ++iPoint) {
            int elementId = elemFullGridToNonLocGridX[iAtom][iPoint];
            const std::vector<double> &shapeFunctionX = shapeFunctionMatrixNonLocGridX[iAtom][iPoint];
            if (elementId > -1) {
                const std::vector<int> &localNodeIds = femNonLocX.getElementConnectivity()[elementId];
                for (int lmcomp = 0; lmcomp < lmCount; ++lmcomp) {
                    const double *localNodalValues = nonLocNodalVx[iAtom][lmcomp].data() + localNodeIds.front();
                    int n = shapeFunctionX.size();
                    int inc = 1;
                    nonLocVx[iAtom][lmcomp][iPoint] = clinalg::ddot_(n,
                                                                     localNodalValues,
                                                                     1,
                                                                     shapeFunctionX.data(),
                                                                     1);
                    //ddot_(&n, localNodalValues, &inc, shapeFunctionX.data(), &inc);
                }
            }
        }
    }

    for (int iAtom = 0; iAtom < numberAtoms; ++iAtom) {
        for (int iPoint = 0; iPoint < totalnumberQuadraturePointsY; ++iPoint) {
            int elementId = elemFullGridToNonLocGridY[iAtom][iPoint];
            const std::vector<double> &shapeFunctionY = shapeFunctionMatrixNonLocGridY[iAtom][iPoint];
            if (elementId > -1) {
                const std::vector<int> &localNodeIds = femNonLocY.getElementConnectivity()[elementId];
                for (int lmcomp = 0; lmcomp < lmCount; ++lmcomp) {
                    const double *localNodalValues = nonLocNodalVy[iAtom][lmcomp].data() + localNodeIds.front();
                    int n = shapeFunctionY.size();
                    int inc = 1;
                    nonLocVy[iAtom][lmcomp][iPoint] = clinalg::ddot_(n,
                                                                     localNodalValues,
                                                                     1,
                                                                     shapeFunctionY.data(),
                                                                     1);
                    //ddot_(&n, localNodalValues, &inc, shapeFunctionY.data(), &inc);
                }
            }
        }
    }

    for (int iAtom = 0; iAtom < numberAtoms; ++iAtom) {
        for (int iPoint = 0; iPoint < totalnumberQuadraturePointsZ; ++iPoint) {
            int elementId = elemFullGridToNonLocGridZ[iAtom][iPoint];
            const std::vector<double> &shapeFunctionZ = shapeFunctionMatrixNonLocGridZ[iAtom][iPoint];
            if (elementId > -1) {
                const std::vector<int> &localNodeIds = femNonLocZ.getElementConnectivity()[elementId];
                for (int lmcomp = 0; lmcomp < lmCount; ++lmcomp) {
                    const double *localNodalValues = nonLocNodalVz[iAtom][lmcomp].data() + localNodeIds.front();
                    int n = shapeFunctionZ.size();
                    int inc = 1;
                    nonLocVz[iAtom][lmcomp][iPoint] = clinalg::ddot_(n,
                                                                     localNodalValues,
                                                                     1,
                                                                     shapeFunctionZ.data(),
                                                                     1);
                    //ddot_(&n, localNodalValues, &inc, shapeFunctionZ.data(), &inc);
                }
            }
        }
    }
}

void FunctionalRayleighQuotientSeperableNonLocal::computeNonLocalOneDimForce(const FEM &fem,
                                                                             const NonLocalPSPData &nonLocalPSPData,
                                                                             const std::vector<std::vector<std::vector<
                                                                                     double> > > &vNloc,
                                                                             std::vector<std::vector<std::vector<std::vector<
                                                                                     double> > > > &oneDimForceNloc) {

    const int numberNodePerElement = fem.getNumberNodesPerElement();
    const int numberElements = fem.getNumberElements();
    const int numberQuadPointsPerElement = fem.getNumberQuadPointsPerElement();
    const int numberTotalQuadPoints = fem.getTotalNumberQuadPoints();
    const std::vector<double> &jacobianQuadPointValues = fem.getJacobQuadPointValues();
    const std::vector<double> &weightQuadPointValues = fem.getWeightQuadPointValues();
    const std::vector<std::vector<double> > &shapeFunction = fem.getShapeFunctionAtQuadPoints();

    int numberAtoms = nonLocalPSPData.getNumberNonLocalAtoms();
    int lMax = nonLocalPSPData.getLMax();
    int lmCount = lMax * lMax;

    oneDimForceNloc = std::vector<std::vector<std::vector<std::vector<double> > > >
            (numberNodePerElement,
             std::vector<std::vector<std::vector<double> > >(
                     numberAtoms,
                     std::vector<std::vector<double> >(
                             lmCount,
                             std::vector<double>(numberTotalQuadPoints,
                                                 0.0))));

    for (int iNode = 0; iNode != numberNodePerElement; ++iNode) {
        for (int iAtoms = 0; iAtoms < numberAtoms; ++iAtoms) {
            for (int lmcomp = 0; lmcomp < lmCount; ++lmcomp) {
                for (int ele = 0; ele < numberElements; ++ele) {
                    for (int iquad = 0; iquad < numberQuadPointsPerElement; ++iquad) {
                        oneDimForceNloc[iNode][iAtoms][lmcomp][ele * numberQuadPointsPerElement + iquad] =
                                vNloc[iAtoms][lmcomp][ele * numberQuadPointsPerElement + iquad] *
                                weightQuadPointValues[ele * numberQuadPointsPerElement + iquad] *
                                jacobianQuadPointValues[ele * numberQuadPointsPerElement + iquad] *
                                shapeFunction[iNode][iquad];
                    }
                }
            }
        }
    }
}

void FunctionalRayleighQuotientSeperableNonLocal::computeCmatTranslmTimesPsi(const std::vector<std::vector<std::vector<
        double> > > &vNloc,
                                                                             const std::vector<double> &nodalFields,
                                                                             const FEM &fem,
                                                                             const NonLocalPSPData &nonLocalPSPData,
                                                                             std::vector<std::vector<double> > &CmatTranslmTimesPsi) {
    int numberAtoms = nonLocalPSPData.getNumberNonLocalAtoms();
    int lMax = nonLocalPSPData.getLMax();
    int lmCount = lMax * lMax;
    int numberTotalQuadPoints = fem.getTotalNumberQuadPoints();

    auto &weightQuadPointValues = fem.getWeightQuadPointValues();
    auto &jacobianQuadPointValues = fem.getJacobQuadPointValues();

    std::vector<double> psiQuadValues, DPsiQuadValues;
    fem.computeFieldAndDiffFieldAtAllQuadPoints(nodalFields,
                                                psiQuadValues,
                                                DPsiQuadValues);

    CmatTranslmTimesPsi = std::vector<std::vector<double> >(numberAtoms,
                                                            std::vector<double>(lmCount,
                                                                                0.0));

    for (int iAtom = 0; iAtom < numberAtoms; ++iAtom) {
        for (int lmcomp = 0; lmcomp < lmCount; ++lmcomp) {
            for (int i = 0; i < numberTotalQuadPoints; ++i) {
                CmatTranslmTimesPsi[iAtom][lmcomp] +=
                        vNloc[iAtom][lmcomp][i] * psiQuadValues[i] * weightQuadPointValues[i] *
                        jacobianQuadPointValues[i];
            }
        }
    }
}

void FunctionalRayleighQuotientSeperableNonLocal::sumOneDimNonLocalForceOverQuadPoints(const FEM &fem,
                                                                                       const std::vector<std::vector<std::vector<
                                                                                               std::vector<double> > > > &oneDimForceNloc,
                                                                                       const std::vector<std::vector<
                                                                                               double> > &CmatTranslmTimesPsi,
                                                                                       const NonLocalPSPData &nonLocalPSPData,
                                                                                       std::vector<double> &summedNonLocalOneDimForce) {
    const int numberElements = fem.getNumberElements();
    const int numberNodesPerElement = fem.getNumberNodesPerElement();
    const int numberQuadPointsPerElement = fem.getNumberQuadPointsPerElement();
    const int numberTotalNodes = fem.getTotalNumberNodes();
    const std::vector<std::vector<int> > &elementConnectivity = fem.getElementConnectivity();

    const int numberAToms = CmatTranslmTimesPsi.size();
    auto &C_lm = nonLocalPSPData.getC_lm();

    summedNonLocalOneDimForce = std::vector<double>(numberTotalNodes,
                                                    0.0);

    for (int ele = 0; ele != numberElements; ++ele) {
        int start = ele * numberQuadPointsPerElement;
        int end = (ele + 1) * numberQuadPointsPerElement;
        for (int iNode = 0; iNode != numberNodesPerElement; ++iNode) {
            int iGlobal = elementConnectivity[ele][iNode];
            for (int iAtom = 0; iAtom < numberAToms; ++iAtom) {
                for (int lmcomp = 0; lmcomp < CmatTranslmTimesPsi[iAtom].size(); ++lmcomp) {
                    double sum = 0.0;
                    double invClm = 1.0 / C_lm[lmcomp];
                    for (int iquad = start; iquad < end; ++iquad) {
                        sum += oneDimForceNloc[iNode][iAtom][lmcomp][iquad];
                    }
                    summedNonLocalOneDimForce[iGlobal] += sum * invClm * CmatTranslmTimesPsi[iAtom][lmcomp];
                }
            }
        }
    }
}

/**
 * @brief a lmbada function to compute integral of u*Psix, v*Psiy, w*Psiz
 */
void FunctionalRayleighQuotientSeperableNonLocal::computeMatPsiIntegral(const FEM &fem,
                                                                        const int rankNloc,
                                                                        const int numberAtoms,
                                                                        const int lmCount,
                                                                        const std::vector<Tucker::Matrix *> &matNonLoc,
                                                                        const std::vector<std::vector<double>> &psiNonLocQuadValues,
                                                                        std::vector<std::vector<Tucker::Matrix *>> &matPsiIntegral) {
    int sizeCompactSupport = fem.getTotalNumberQuadPoints();

    auto &weightQuadPointValues = fem.getWeightQuadPointValues();
    auto &jacobianQuadPointValues = fem.getJacobQuadPointValues();

    for (int irank = 0; irank < rankNloc; ++irank) {
        for (int iAtom = 0; iAtom < numberAtoms; ++iAtom) {
            for (int lmcomp = 0; lmcomp < lmCount; ++lmcomp) {
                const double *matNonLocData = matNonLoc[lmcomp]->data() + irank * sizeCompactSupport;
                double *matPsiIntegral_data = matPsiIntegral[iAtom][lmcomp]->data();
                for (int index = 0; index < sizeCompactSupport; ++index) {
                    matPsiIntegral_data[irank] +=
                            weightQuadPointValues[index] * psiNonLocQuadValues[iAtom][index] * matNonLocData[index]
                            * jacobianQuadPointValues[index];
                }
            }
        }
    }
}

void FunctionalRayleighQuotientSeperableNonLocal::computeMatShapeIntegral(const FEM &femNonLoc,
                                                                          const int rankNloc,
                                                                          const int lMax,
                                                                          const std::vector<Tucker::Matrix *> &mat,
                                                                          std::vector<std::vector<std::vector<double>>> &FNLocTimesShp) {
    int lmCount = lMax * lMax;
    int numberElements = femNonLoc.getNumberElements();
    int numberQuadPointsPerElement = femNonLoc.getNumberQuadPointsPerElement();
    int numberNodesPerElement = femNonLoc.getNumberNodesPerElement();
    int numberTotalQuadPoints = femNonLoc.getTotalNumberQuadPoints();
    int numberTotalNodes = femNonLoc.getTotalNumberNodes();

    std::vector<std::vector<std::vector<std::vector<double>>>> matNlocTimesShp(
            numberNodesPerElement,
            std::vector<std::vector<std::vector<double>>>(
                    rankNloc,
                    std::vector<std::vector<double>>(
                            lmCount,
                            std::vector<double>(
                                    numberTotalQuadPoints,
                                    0.0))));

    auto &weightQuadPointValues = femNonLoc.getWeightQuadPointValues();
    auto &jacobianQuadPointValues = femNonLoc.getJacobQuadPointValues();
    auto &shapeFunctionAtQuadPoints = femNonLoc.getShapeFunctionAtQuadPoints();

    for (int iNode = 0; iNode < numberNodesPerElement; ++iNode) {
        for (int irank = 0; irank < rankNloc; ++irank) {
            for (int lmcomp = 0; lmcomp < lmCount; ++lmcomp) {
                int cnt = 0;
                double *matData = mat[lmcomp]->data() + irank * numberTotalQuadPoints;
                for (int ele = 0; ele < numberElements; ++ele) {
                    for (int iquad = 0; iquad < numberQuadPointsPerElement; ++iquad) {
                        matNlocTimesShp[iNode][irank][lmcomp][cnt] = weightQuadPointValues[cnt] * matData[cnt] *
                                                                     shapeFunctionAtQuadPoints[iNode][iquad] *
                                                                     jacobianQuadPointValues[cnt];
                        cnt++;
                    }
                }
            }
        }
    }

    FNLocTimesShp = std::vector<std::vector<std::vector<double>>>(
            rankNloc,
            std::vector<std::vector<double>>(
                    lmCount,
                    std::vector<double>(
                            numberTotalNodes,
                            0.0)));
    auto &elementConnectivity = femNonLoc.getElementConnectivity();
    for (int ele = 0; ele < numberElements; ++ele) {
        for (int iNode = 0; iNode < numberNodesPerElement; ++iNode) {
            int iGlobal = elementConnectivity[ele][iNode];
            for (int irank = 0; irank < rankNloc; ++irank) {
                for (int lmcomp = 0; lmcomp < lmCount; ++lmcomp) {
                    for (int iquad = ele * numberQuadPointsPerElement;
                         iquad < (ele + 1) * numberQuadPointsPerElement; ++iquad) {
                        FNLocTimesShp[irank][lmcomp][iGlobal] += matNlocTimesShp[iNode][irank][lmcomp][iquad];
                    }
                }
            }

        }
    }
}

void FunctionalRayleighQuotientSeperableNonLocal::computeShpTimesShpTimesPsiVNloc(const int numberAtoms,
                                                                                  const int lMax,
                                                                                  const int rankNloc,
                                                                                  const int numberTotalNodesX,
                                                                                  const int numberTotalNodesY,
                                                                                  const int numberTotalNodesZ,
                                                                                  const std::vector<Tucker::Tensor *> &sigma,
                                                                                  const std::vector<std::vector<std::vector<
                                                                                          double>>> &Fx,
                                                                                  const std::vector<std::vector<std::vector<
                                                                                          double>>> &Fy,
                                                                                  const std::vector<std::vector<std::vector<
                                                                                          double>>> &Fz,
                                                                                  const std::vector<std::vector<std::vector<
                                                                                          double>>> &intUiPsix,
                                                                                  const std::vector<std::vector<std::vector<
                                                                                          double>>> &intViPsiy,
                                                                                  const std::vector<std::vector<std::vector<
                                                                                          double>>> &intWiPsiz,
                                                                                  std::vector<std::vector<std::vector<
                                                                                          std::vector<double>>>> &shpXi_shpYj_PsiZ_VNloc,
                                                                                  std::vector<std::vector<std::vector<
                                                                                          std::vector<double>>>> &shpXi_PsiY_shpZj_VNloc,
                                                                                  std::vector<std::vector<std::vector<
                                                                                          std::vector<double>>>> &PsiX_shpYi_shpZj_VNloc) {

    int lmCount = lMax * lMax;

    shpXi_shpYj_PsiZ_VNloc = std::vector<std::vector<std::vector<std::vector<double>>>>(
            numberAtoms,
            std::vector<std::vector<std::vector<double>>>(
                    lmCount,
                    std::vector<std::vector<double>>(
                            numberTotalNodesX,
                            std::vector<double>(
                                    numberTotalNodesY,
                                    0.0))));
    for (int iAtom = 0; iAtom < numberAtoms; ++iAtom) {
        for (int lmcomp = 0; lmcomp < lmCount; ++lmcomp) {
            for (int inodeGlobal = 0; inodeGlobal < numberTotalNodesX; ++inodeGlobal) {
                for (int jnodeGlobal = 0; jnodeGlobal < numberTotalNodesY; ++jnodeGlobal) {
                    double *sigmaData = sigma[lmcomp]->data();
                    for (int r = 0; r < rankNloc; ++r) {
                        for (int q = 0; q < rankNloc; ++q) {
                            for (int p = 0; p < rankNloc; ++p) {
                                shpXi_shpYj_PsiZ_VNloc[iAtom][lmcomp][inodeGlobal][jnodeGlobal] +=
                                        sigmaData[p + q * rankNloc + r * rankNloc * rankNloc] *
                                        Fx[p][lmcomp][inodeGlobal] *
                                        Fy[q][lmcomp][jnodeGlobal] * intWiPsiz[iAtom][lmcomp][r];
                            }
                        }
                    }
                }
            }
        }
    }

    shpXi_PsiY_shpZj_VNloc = std::vector<std::vector<std::vector<std::vector<double>>>>(
            numberAtoms,
            std::vector<std::vector<std::vector<double>>>(
                    lmCount,
                    std::vector<std::vector<double>>(
                            numberTotalNodesX,
                            std::vector<double>(
                                    numberTotalNodesZ,
                                    0.0))));
    for (int iAtom = 0; iAtom < numberAtoms; ++iAtom) {
        for (int lmcomp = 0; lmcomp < lmCount; ++lmcomp) {
            for (int inodeGlobal = 0; inodeGlobal < numberTotalNodesX; ++inodeGlobal) {
                for (int jnodeGlobal = 0; jnodeGlobal < numberTotalNodesZ; ++jnodeGlobal) {
                    double *sigmaData = sigma[lmcomp]->data();
                    for (int r = 0; r < rankNloc; ++r) {
                        for (int q = 0; q < rankNloc; ++q) {
                            for (int p = 0; p < rankNloc; ++p) {
                                shpXi_PsiY_shpZj_VNloc[iAtom][lmcomp][inodeGlobal][jnodeGlobal] +=
                                        sigmaData[p + q * rankNloc + r * rankNloc * rankNloc] *
                                        Fx[p][lmcomp][inodeGlobal] *
                                        Fz[r][lmcomp][jnodeGlobal] * intViPsiy[iAtom][lmcomp][q];
                            }
                        }
                    }
                }
            }
        }
    }

    PsiX_shpYi_shpZj_VNloc = std::vector<std::vector<std::vector<std::vector<double>>>>(
            numberAtoms,
            std::vector<std::vector<std::vector<double>>>(
                    lmCount,
                    std::vector<std::vector<double>>(
                            numberTotalNodesY,
                            std::vector<double>(
                                    numberTotalNodesZ,
                                    0.0))));
    for (int iAtom = 0; iAtom < numberAtoms; ++iAtom) {
        for (int lmcomp = 0; lmcomp < lmCount; ++lmcomp) {
            for (int inodeGlobal = 0; inodeGlobal < numberTotalNodesY; ++inodeGlobal) {
                for (int jnodeGlobal = 0; jnodeGlobal < numberTotalNodesZ; ++jnodeGlobal) {
                    double *sigmaData = sigma[lmcomp]->data();
                    for (int r = 0; r < rankNloc; ++r) {
                        for (int q = 0; q < rankNloc; ++q) {
                            for (int p = 0; p < rankNloc; ++p) {
                                PsiX_shpYi_shpZj_VNloc[iAtom][lmcomp][inodeGlobal][jnodeGlobal] +=
                                        sigmaData[p + q * rankNloc + r * rankNloc * rankNloc] *
                                        Fy[q][lmcomp][inodeGlobal] *
                                        Fz[r][lmcomp][jnodeGlobal] * intUiPsix[iAtom][lmcomp][p];
                            }
                        }
                    }
                }
            }
        }
    }
}

/**
 *
 * @brief compute the product of the integral of u*Psix, v*Psiy, w*Psiz
 */
void FunctionalRayleighQuotientSeperableNonLocal::computePsiVNlocProduct(const int numberAtoms,
                                                                         const int lMax,
                                                                         const int rankNloc,
                                                                         const std::vector<Tucker::Tensor *> &sigma,
                                                                         const std::vector<std::vector<std::vector<
                                                                                 double>>> &intUiPsix,
                                                                         const std::vector<std::vector<std::vector<
                                                                                 double>>> &intViPsiy,
                                                                         const std::vector<std::vector<std::vector<
                                                                                 double>>> &intWiPsiz,
                                                                         std::vector<std::vector<double>> &PsiX_PsiY_PsiZ_VNloc) {
    int lmCount = lMax * lMax;
    PsiX_PsiY_PsiZ_VNloc = std::vector<std::vector<double>>(
            numberAtoms,
            std::vector<double>(
                    lmCount,
                    0.0));
    for (int iAtom = 0; iAtom < numberAtoms; ++iAtom) {
        for (int lmcomp = 0; lmcomp < lmCount; ++lmcomp) {
            double *sigmaData = sigma[lmcomp]->data();
            for (int r = 0; r < rankNloc; ++r) {
                for (int q = 0; q < rankNloc; ++q) {
                    for (int p = 0; p < rankNloc; ++p) {
                        PsiX_PsiY_PsiZ_VNloc[iAtom][lmcomp] +=
                                sigmaData[p + q * rankNloc + r * rankNloc * rankNloc] *
                                intUiPsix[iAtom][lmcomp][p] * intViPsiy[iAtom][lmcomp][q] * intWiPsiz[iAtom][lmcomp][r];
                    }
                }
            }
        }
    }
}

void FunctionalRayleighQuotientSeperableNonLocal::computeShpTimesPsiVNlocTimesPsiVNloc(const int numberAtoms,
                                                                                       const int lMax,
                                                                                       const int rankNloc,
                                                                                       const int numberTotalNodesX,
                                                                                       const int numberTotalNodesY,
                                                                                       const int numberTotalNodesZ,
                                                                                       const std::vector<Tucker::Tensor *> &sigma,
                                                                                       const std::vector<std::vector<std::vector<
                                                                                               double>>> &Fx,
                                                                                       const std::vector<std::vector<std::vector<
                                                                                               double>>> &Fy,
                                                                                       const std::vector<std::vector<std::vector<
                                                                                               double>>> &Fz,
                                                                                       const std::vector<std::vector<std::vector<
                                                                                               double>>> &intUiPsix,
                                                                                       const std::vector<std::vector<std::vector<
                                                                                               double>>> &intViPsiy,
                                                                                       const std::vector<std::vector<std::vector<
                                                                                               double>>> &intWiPsiz,
                                                                                       std::vector<std::vector<std::vector<
                                                                                               double>>> &shpXi_PsiY_PsiZ_VNloc,
                                                                                       std::vector<std::vector<std::vector<
                                                                                               double>>> &PsiX_shpYi_PsiZ_VNloc,
                                                                                       std::vector<std::vector<std::vector<
                                                                                               double>>> &PsiX_PsiY_shpZi_VNloc) {

    int lmCount = lMax * lMax;

    shpXi_PsiY_PsiZ_VNloc = std::vector<std::vector<std::vector<double>>>(
            numberAtoms,
            std::vector<std::vector<double>>(
                    lmCount,
                    std::vector<double>(
                            numberTotalNodesX,
                            0.0)));
    for (int iAtom = 0; iAtom < numberAtoms; ++iAtom) {
        for (int lmcomp = 0; lmcomp < lmCount; ++lmcomp) {
            for (int inodeGlobal = 0; inodeGlobal < numberTotalNodesX; ++inodeGlobal) {
                double *sigmaData = sigma[lmcomp]->data();
                for (int r = 0; r < rankNloc; ++r) {
                    for (int q = 0; q < rankNloc; ++q) {
                        for (int p = 0; p < rankNloc; ++p) {
                            shpXi_PsiY_PsiZ_VNloc[iAtom][lmcomp][inodeGlobal] +=
                                    sigmaData[p + q * rankNloc + r * rankNloc * rankNloc] *
                                    Fx[p][lmcomp][inodeGlobal] * intViPsiy[iAtom][lmcomp][q] *
                                    intWiPsiz[iAtom][lmcomp][r];
                        }
                    }
                }
            }
        }
    }

    PsiX_shpYi_PsiZ_VNloc = std::vector<std::vector<std::vector<double>>>(
            numberAtoms,
            std::vector<std::vector<double>>(
                    lmCount,
                    std::vector<double>(
                            numberTotalNodesY,
                            0.0)));
    for (int iAtom = 0; iAtom < numberAtoms; ++iAtom) {
        for (int lmcomp = 0; lmcomp < lmCount; ++lmcomp) {
            for (int inodeGlobal = 0; inodeGlobal < numberTotalNodesY; ++inodeGlobal) {
                double *sigmaData = sigma[lmcomp]->data();
                for (int r = 0; r < rankNloc; ++r) {
                    for (int q = 0; q < rankNloc; ++q) {
                        for (int p = 0; p < rankNloc; ++p) {
                            PsiX_shpYi_PsiZ_VNloc[iAtom][lmcomp][inodeGlobal] +=
                                    sigmaData[p + q * rankNloc + r * rankNloc * rankNloc] *
                                    Fy[q][lmcomp][inodeGlobal] * intUiPsix[iAtom][lmcomp][p] *
                                    intWiPsiz[iAtom][lmcomp][r];
                        }
                    }
                }
            }
        }
    }

    PsiX_PsiY_shpZi_VNloc = std::vector<std::vector<std::vector<double>>>(
            numberAtoms,
            std::vector<std::vector<double>>(
                    lmCount,
                    std::vector<double>(
                            numberTotalNodesZ,
                            0.0)));
    for (int iAtom = 0; iAtom < numberAtoms; ++iAtom) {
        for (int lmcomp = 0; lmcomp < lmCount; ++lmcomp) {
            for (int inodeGlobal = 0; inodeGlobal < numberTotalNodesZ; ++inodeGlobal) {
                double *sigmaData = sigma[lmcomp]->data();
                for (int r = 0; r < rankNloc; ++r) {
                    for (int q = 0; q < rankNloc; ++q) {
                        for (int p = 0; p < rankNloc; ++p) {
                            PsiX_PsiY_shpZi_VNloc[iAtom][lmcomp][inodeGlobal] +=
                                    sigmaData[p + q * rankNloc + r * rankNloc * rankNloc] *
                                    Fz[r][lmcomp][inodeGlobal] * intUiPsix[iAtom][lmcomp][p] *
                                    intViPsiy[iAtom][lmcomp][q];
                        }
                    }
                }
            }
        }
    }

}

void FunctionalRayleighQuotientSeperableNonLocal::generateHamiltonianGenericPotential(
        const std::vector<double> &nodalFieldsX,
        const std::vector<double> &nodalFieldsY,
        const std::vector<double> &nodalFieldsZ,
        Mat &Hx,
        Mat &Hy,
        Mat &Hz,
        Mat &Mx,
        Mat &My,
        Mat &Mz) {

    FunctionalRayleighQuotientSeperable::generateHamiltonianGenericPotential(nodalFieldsX,
                                                                             nodalFieldsY,
                                                                             nodalFieldsZ,
                                                                             Hx,
                                                                             Hy,
                                                                             Hz,
                                                                             Mx,
                                                                             My,
                                                                             Mz);
    double t0, t1;
    //TODO this part is computed twice, try to optimize later
    std::vector<double> psiQuadValuesX, DPsiQuadValuesX, psiQuadValuesY, DPsiQuadValuesY, psiQuadValuesZ, DPsiQuadValuesZ;
    _femX.computeFieldAndDiffFieldAtAllQuadPoints(nodalFieldsX,
                                                  psiQuadValuesX,
                                                  DPsiQuadValuesX);
    _femY.computeFieldAndDiffFieldAtAllQuadPoints(nodalFieldsY,
                                                  psiQuadValuesY,
                                                  DPsiQuadValuesY);
    _femZ.computeFieldAndDiffFieldAtAllQuadPoints(nodalFieldsZ,
                                                  psiQuadValuesZ,
                                                  DPsiQuadValuesZ);
    double normPsiX, normPsiY, normPsiZ, normDPsiX, normDPsiY, normDPsiZ;
    computeIntegralPsiSquare(psiQuadValuesX,
                             psiQuadValuesY,
                             psiQuadValuesZ,
                             DPsiQuadValuesX,
                             DPsiQuadValuesY,
                             DPsiQuadValuesZ,
                             normPsiX,
                             normPsiY,
                             normPsiZ,
                             normDPsiX,
                             normDPsiY,
                             normDPsiZ);
    double cx = 1.0 / (normPsiY * normPsiZ), cy = 1.0 / (normPsiX * normPsiZ), cz = 1.0 / (normPsiX * normPsiY);

    int num_atom_type = nonLocalPSPData.size();

    std::vector<std::vector<std::vector<double>>> psiNonLocQuadValuesX(num_atom_type),
            psiNonLocQuadValuesY(num_atom_type), psiNonLocQuadValuesZ(num_atom_type);
    for (int i = 0; i < num_atom_type; ++i) {
        computeFieldsAtGivenPointsFullGrid(nodalFieldsX,
                                           nodalFieldsY,
                                           nodalFieldsZ,
                                           _femX,
                                           _femY,
                                           _femZ,
                                           femNonLocX,
                                           femNonLocY,
                                           femNonLocZ,
                                           *nonLocalMapManager[i],
                                           psiNonLocQuadValuesX[i],
                                           psiNonLocQuadValuesY[i],
                                           psiNonLocQuadValuesZ[i]);
    }

    std::vector<std::vector<std::vector<std::vector<double>>>> nonLocNodalVx(num_atom_type), nonLocNodalVy(num_atom_type),
            nonLocNodalVz(num_atom_type);
    for (int i = 0; i < num_atom_type; ++i) {
        computeSeparableNonLocalPotentialsUsingTucker(psiNonLocQuadValuesX[i],
                                                      psiNonLocQuadValuesY[i],
                                                      psiNonLocQuadValuesZ[i],
                                                      femNonLocX,
                                                      femNonLocY,
                                                      femNonLocZ,
                                                      *nonLocalPSPData[i],
                                                      nonLocNodalVx[i],
                                                      nonLocNodalVy[i],
                                                      nonLocNodalVz[i]);
    }
    std::vector<std::vector<std::vector<std::vector<double>>>> nonLocVx(num_atom_type), nonLocVy(num_atom_type),
            nonLocVz(num_atom_type);
    for (int i = 0; i < num_atom_type; ++i) {
        computeFieldsAtGivenPointsNonLocGrid(nonLocNodalVx[i],
                                             nonLocNodalVy[i],
                                             nonLocNodalVz[i],
                                             _femX,
                                             _femY,
                                             _femZ,
                                             femNonLocX,
                                             femNonLocY,
                                             femNonLocZ,
                                             *nonLocalPSPData[i],
                                             *nonLocalMapManager[i],
                                             nonLocVx[i],
                                             nonLocVy[i],
                                             nonLocVz[i]);
    }
    std::vector<std::vector<std::vector<std::vector<std::vector<double>>>>> X0Nloc(num_atom_type), Y0Nloc(num_atom_type),
            Z0Nloc(num_atom_type);

    for (int i = 0; i < num_atom_type; ++i) {
        computeNonLocalOneDimForce(_femX,
                                   *nonLocalPSPData[i],
                                   nonLocVx[i],
                                   X0Nloc[i]);
        computeNonLocalOneDimForce(_femY,
                                   *nonLocalPSPData[i],
                                   nonLocVy[i],
                                   Y0Nloc[i]);
        computeNonLocalOneDimForce(_femZ,
                                   *nonLocalPSPData[i],
                                   nonLocVz[i],
                                   Z0Nloc[i]);
    }
    for (int i = 0; i < num_atom_type; ++i) {
        int numberAtoms = nonLocalPSPData[i]->getNumberNonLocalAtoms();
        int lMax = nonLocalPSPData[i]->getLMax();
        auto &C_lm = nonLocalPSPData[i]->getC_lm();

        computeHNloc(numberAtoms,
                     lMax,
                     _femX,
                     cx,
                     C_lm,
                     X0Nloc[i],
                     Hx);
        computeHNloc(numberAtoms,
                     lMax,
                     _femY,
                     cy,
                     C_lm,
                     Y0Nloc[i],
                     Hy);
        computeHNloc(numberAtoms,
                     lMax,
                     _femZ,
                     cz,
                     C_lm,
                     Z0Nloc[i],
                     Hz);
    }
}

void FunctionalRayleighQuotientSeperableNonLocal::computeHNloc(const int numberAtoms,
                                                               const int lMax,
                                                               const FEM &fem,
                                                               const double potConst,
                                                               const std::vector<double> &C_lm,
                                                               const std::vector<std::vector<std::vector<std::vector<
                                                                       double> > > > &vnloc,
                                                               Mat &H) {
    int lmCount = lMax * lMax;
    int numberNodesPerElement = fem.getNumberNodesPerElement();
    int numberTotalNodes = fem.getTotalNumberNodes();
    int numberQuadPointsPerElement = fem.getNumberQuadPointsPerElement();
    // the boundary nodes need to be taken care of later
    std::vector<std::vector<double> > CNloc(numberAtoms,
                                            std::vector<double>(lmCount * numberTotalNodes,
                                                                0.0));
    for (auto ele = 0; ele < fem.getNumberElements(); ++ele) {
        int start = ele * numberQuadPointsPerElement, end = (ele + 1) * numberQuadPointsPerElement;
        for (int iNode = 0; iNode < numberNodesPerElement; ++iNode) {
            int iGlobal = fem.getElementConnectivity()[ele][iNode];
            for (int iAtom = 0; iAtom < numberAtoms; ++iAtom) {
                for (int lmcomp = 0; lmcomp < lmCount; ++lmcomp) {
                    for (int iquad = start; iquad < end; ++iquad) {
                        CNloc[iAtom][iGlobal * lmCount + lmcomp] += vnloc[iNode][iAtom][lmcomp][iquad];
                    }
                }
            }
        }
    }

    std::vector<double> invDiagClm(C_lm.size() * C_lm.size(),
                                   0.0);
    for (int i = 0; i < C_lm.size(); ++i) {
        invDiagClm[i + i * C_lm.size()] = 1.0 / C_lm[i];
    }

    std::vector<double> CTransposetimesC(numberTotalNodes * numberTotalNodes,
                                         0.0);
    for (int iAtom = 0; iAtom < numberAtoms; ++iAtom) {
        // invDiagClm * CxNloc[iAtom]
        char transa = 'N', transb = 'N';
        int m = C_lm.size(), k = C_lm.size(), n = numberTotalNodes;
        double alpha = 1.0, beta = 0.0;
        int lda = m, ldb = k, ldc = m;
        std::vector<double> temp(C_lm.size() * numberTotalNodes,
                                 0.0);
        clinalg::dgemm_('N',
                        'N',
                        m,
                        n,
                        k,
                        1.0,
                        invDiagClm.data(),
                        lda,
                        CNloc[iAtom].data(),
                        ldb,
                        0.0,
                        temp.data(),
                        ldc);
//    dgemm_(&transa,
//           &transb,
//           &m,
//           &n,
//           &k,
//           &alpha,
//           invDiagClm.data(),
//           &lda,
//           CNloc[iAtom].data(),
//           &ldb,
//           &beta,
//           temp.data(),
//           &ldc);

        // CTransposetimesC += CNloc' * temp
        transa = 'T';
        transb = 'N';
        m = numberTotalNodes;
        k = C_lm.size();
        n = numberTotalNodes;
        lda = k;
        ldb = k;
        ldc = m;
        beta = 1.0;
//    dgemm_(&transa,
//           &transb,
//           &m,
//           &n,
//           &k,
//           &alpha,
//           CNloc[iAtom].data(),
//           &lda,
//           temp.data(),
//           &ldb,
//           &beta,
//           CTransposetimesC.data(),
//           &ldc);
        clinalg::dgemm_('T',
                        'N',
                        m,
                        n,
                        k,
                        1.0,
                        CNloc[iAtom].data(),
                        lda,
                        temp.data(),
                        ldb,
                        1.0,
                        CTransposetimesC.data(),
                        ldc);
    }

    std::vector<double> condensedCTransposetimesC((numberTotalNodes - 2) * (numberTotalNodes - 2),
                                                  0.0);
    int cnt = 0;
    for (int j = 1; j < numberTotalNodes - 1; ++j) {
        for (int i = 1; i < numberTotalNodes - 1; ++i) {
            condensedCTransposetimesC[cnt++] = potConst * CTransposetimesC[i + j * numberTotalNodes];
        }
    }

    //fixdebug for debugging use, remove when optimizing
    PetscInt Hm, Hn;
    MatGetSize(H,
               &Hm,
               &Hn);
    assert(Hm * Hn == condensedCTransposetimesC.size());
    MatType type;
    MatGetType(H,
               &type);
//    assert(type == MATSEQDENSE);

    double *Hdata;
    MatDenseGetArray(H,
                     &Hdata);
    for (int i = 0; i < condensedCTransposetimesC.size(); ++i) {
        Hdata[i] += condensedCTransposetimesC[i];
    }
    MatDenseRestoreArray(H,
                         &Hdata);
    MatAssemblyBegin(H,
                     MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(H,
                   MAT_FINAL_ASSEMBLY);
}

