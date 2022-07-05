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

#include "NonLocalPSPData.h"
#include "../dft/KSDFTEnergyFunctional.h"
#include "../tensor/TensorUtils.h"
#include <boost/math/special_functions/spherical_harmonic.hpp>

namespace {
    extern "C" {
    int daxpy_(int *n,
               double *sa,
               double *sx,
               int *incx,
               double *sy,
               int *incy);
    }
}

NonLocalPSPData::NonLocalPSPData(const FEM &femNonLocX,
                                 const FEM &femNonLocY,
                                 const FEM &femNonLocZ,
                                 const int numberNonLocalAtoms,
                                 const int lMax,
                                 const int rankNloc,
                                 const std::string &localPSPFilename,
                                 const std::string &VlFilename,
                                 const std::string &RlFilename) :
        numberNonLocalAtoms(numberNonLocalAtoms),
        rankNloc(rankNloc),
        lMax(lMax),
        C_lm(lMax * lMax,
             0.0),
        sigmaQuad(lMax * lMax),
        umatQuad(lMax * lMax),
        vmatQuad(lMax * lMax),
        wmatQuad(lMax * lMax),
        sigmaNode(lMax * lMax),
        umatNode(lMax * lMax),
        vmatNode(lMax * lMax),
        wmatNode(lMax * lMax),
        umatNodeInterpolatedQuad(lMax * lMax),
        vmatNodeInterpolatedQuad(lMax * lMax),
        wmatNodeInterpolatedQuad(lMax * lMax) {

    // compute the non-local data, DeltaVl, Rl, phi_lm, required to compute Tucker decomposition of nonlocal data & C_lm
    // since the tensors are not actually used in the following code, they are not stored after the constructor is finished
    int numberTotalNodesNonLocX = femNonLocX.getTotalNumberNodes();
    int numberTotalNodesNonLocY = femNonLocY.getTotalNumberNodes();
    int numberTotalNodesNonLocZ = femNonLocZ.getTotalNumberNodes();
    int numberTotalQuadNonLocalX = femNonLocX.getTotalNumberQuadPoints();
    int numberTotalQuadNonLocalY = femNonLocY.getTotalNumberQuadPoints();
    int numberTotalQuadNonLocalZ = femNonLocZ.getTotalNumberQuadPoints();

    Tensor3DMPI
            distanceFromAtomNodal(numberTotalNodesNonLocX,
                                  numberTotalNodesNonLocY,
                                  numberTotalNodesNonLocZ,
                                  MPI_COMM_WORLD);
    Tensor3DMPI distanceFromAtomQuad
            (numberTotalQuadNonLocalX,
             numberTotalQuadNonLocalY,
             numberTotalQuadNonLocalZ,
             MPI_COMM_WORLD);
    generateNodalDistance(femNonLocX,
                          femNonLocY,
                          femNonLocZ,
                          distanceFromAtomNodal);
    generateQuadDistance(femNonLocX,
                         femNonLocY,
                         femNonLocZ,
                         distanceFromAtomQuad);

    std::vector<Tensor3DMPI> DeltaVlQuad
            (lMax,
             Tensor3DMPI(numberTotalQuadNonLocalX,
                         numberTotalQuadNonLocalY,
                         numberTotalQuadNonLocalZ,
                         MPI_COMM_WORLD));
    std::vector<Tensor3DMPI> RlQuad
            (lMax,
             Tensor3DMPI(numberTotalQuadNonLocalX,
                         numberTotalQuadNonLocalY,
                         numberTotalQuadNonLocalZ,
                         MPI_COMM_WORLD));
    std::vector<Tensor3DMPI> DeltaVlNode
            (lMax,
             Tensor3DMPI(numberTotalNodesNonLocX,
                         numberTotalNodesNonLocY,
                         numberTotalNodesNonLocZ,
                         MPI_COMM_WORLD));
    std::vector<Tensor3DMPI> RlNode
            (lMax,
             Tensor3DMPI(numberTotalNodesNonLocX,
                         numberTotalNodesNonLocY,
                         numberTotalNodesNonLocZ,
                         MPI_COMM_WORLD));

    // generate spline function for nonlocal PSP fitting
    alglib::spline1dinterpolant funcLocalPSP;
    std::vector<alglib::spline1dinterpolant> funcDeltaVl(lMax), funcRl(lMax);
    createNonlinearSplineObjects(localPSPFilename,
                                 VlFilename,
                                 RlFilename,
                                 funcLocalPSP,
                                 funcDeltaVl,
                                 funcRl);

    for (int l = 0; l < lMax; ++l) {
        computeFieldFromSpline(distanceFromAtomNodal,
                               funcDeltaVl[l],
                               DeltaVlNode[l]);
        computeFieldFromSpline(distanceFromAtomNodal,
                               funcRl[l],
                               RlNode[l]);
        computeFieldFromSpline(distanceFromAtomQuad,
                               funcDeltaVl[l],
                               DeltaVlQuad[l]);
        computeFieldFromSpline(distanceFromAtomQuad,
                               funcRl[l],
                               RlQuad[l]);
    }

    std::vector<Tensor3DMPI> phi_lmQuad(lMax * lMax,
                                        Tensor3DMPI(numberTotalQuadNonLocalX,
                                                    numberTotalQuadNonLocalY,
                                                    numberTotalQuadNonLocalZ,
                                                    MPI_COMM_WORLD));
    std::vector<Tensor3DMPI> phi_lmNode(lMax * lMax,
                                        Tensor3DMPI(numberTotalNodesNonLocX,
                                                    numberTotalNodesNonLocY,
                                                    numberTotalNodesNonLocZ,
                                                    MPI_COMM_WORLD));
    computePhi_lm(femNonLocX.getGlobalNodalCoord(),
                  femNonLocY.getGlobalNodalCoord(),
                  femNonLocZ.getGlobalNodalCoord(),
                  lMax,
                  distanceFromAtomNodal,
                  RlNode,
                  phi_lmNode);
    computePhi_lm(femNonLocX.getPositionQuadPointValues(),
                  femNonLocY.getPositionQuadPointValues(),
                  femNonLocZ.getPositionQuadPointValues(),
                  lMax,
                  distanceFromAtomQuad,
                  RlQuad,
                  phi_lmQuad);

    // compute C_lm
    computeC_lm(lMax,
                rankNloc,
                femNonLocX,
                femNonLocY,
                femNonLocZ,
                DeltaVlQuad,
                phi_lmQuad,
                C_lm);
    // compute the Tucker decomposition of phi_lm * DeltaVl
    computeDecomposedNonlocalData(DeltaVlNode,
                                  phi_lmNode,
                                  lMax,
                                  rankNloc,
                                  sigmaNode,
                                  umatNode,
                                  vmatNode,
                                  wmatNode);
    computeDecomposedNonlocalData(DeltaVlQuad,
                                  phi_lmQuad,
                                  lMax,
                                  rankNloc,
                                  sigmaQuad,
                                  umatQuad,
                                  vmatQuad,
                                  wmatQuad);
    // interpolate node factor matrices of phi_lm * Delta_Vl onto quadrature points
    projectNodeOntoQuad(umatNode,
                        femNonLocX,
                        umatNodeInterpolatedQuad);
    projectNodeOntoQuad(vmatNode,
                        femNonLocY,
                        vmatNodeInterpolatedQuad);
    projectNodeOntoQuad(wmatNode,
                        femNonLocZ,
                        wmatNodeInterpolatedQuad);

}

void NonLocalPSPData::computeLocalPart(const FEM &femX,
                                       const FEM &femY,
                                       const FEM &femZ,
                                       const std::vector<std::vector<double> > &nuclei,
                                       const std::string &localPSPFilename,
                                       Tensor3DMPI &localPSP) {
    int numberQuadPointsX = femX.getTotalNumberQuadPoints();
    int numberQuadPointsY = femY.getTotalNumberQuadPoints();
    int numberQuadPointsZ = femZ.getTotalNumberQuadPoints();

    //todo this part is for debug only, remove when tested thoroughly
    int dimx = localPSP.getGlobalDimension(0), dimy = localPSP.getGlobalDimension(1),
            dimz = localPSP.getGlobalDimension(2);
    assert(dimx == numberQuadPointsX);
    assert(dimy == numberQuadPointsY);
    assert(dimz == numberQuadPointsZ);

    alglib::spline1dinterpolant funcLocalPSP;
    createSplineObject(localPSPFilename,
                       funcLocalPSP);

    int taskId;
    MPI_Comm_rank(MPI_COMM_WORLD,
                  &taskId);

    Tensor3DMPI distance(numberQuadPointsX,
                         numberQuadPointsY,
                         numberQuadPointsZ,
                         MPI_COMM_WORLD);

    Tensor3DMPI temp(numberQuadPointsX,
                     numberQuadPointsY,
                     numberQuadPointsZ,
                     MPI_COMM_WORLD);


    for (int iAtom = 0; iAtom < nuclei.size(); ++iAtom) {
        double x = nuclei[iAtom][1], y = nuclei[iAtom][2], z = nuclei[iAtom][3];
        double atomCharge = nuclei[iAtom][0];

        distance.setEntriesZero();
        temp.setEntriesZero();

        generateQuadDistanceFromAtom(femX,
                                     femY,
                                     femZ,
                                     x,
                                     y,
                                     z,
                                     distance);
        computeFieldFromSpline(distance,
                               funcLocalPSP,
                               temp);
        int numLocalEntires = localPSP.getLocalNumberEntries();
        double *distanceData = distance.getLocalData();
        double *tempData = temp.getLocalData();
        double *localPSPData = localPSP.getLocalData();
        for (int i = 0; i < numLocalEntires; ++i) {
            double d = distanceData[i] + 1.0e-12;
            double potlocZbyr = -atomCharge / d;
            double fact1 = 0.5 * (1.0 + (d - 75.001) / std::sqrt((d - 75.001) * (d - 75.001)));
            double fact2 = 0.5 * (1.0 - (d - 75.001) / std::sqrt((d - 75.001) * (d - 75.001)));
            localPSPData[i] += fact1 * potlocZbyr + fact2 * tempData[i];
        }
    }
}

void NonLocalPSPData::generateNodalDistance(const FEM &femX,
                                            const FEM &femY,
                                            const FEM &femZ,
                                            Tensor3DMPI &distanceNodal) {
    int numberLocalEntries = distanceNodal.getLocalNumberEntries();
    if (numberLocalEntries == 0)
        return;

    const std::vector<double> &nodalValuesX = femX.getGlobalNodalCoord();
    const std::vector<double> &nodalValuesY = femY.getGlobalNodalCoord();
    const std::vector<double> &nodalValuesZ = femZ.getGlobalNodalCoord();

    std::array<int, 6> nodalIdx;
    double *nodalData = distanceNodal.getLocalData(nodalIdx);

    auto computeNodalDistance = [&nodalValuesX, &nodalValuesY, &nodalValuesZ]
            (int i,
             int j,
             int k) -> double {
        return std::sqrt(
                nodalValuesX[i] * nodalValuesX[i] + nodalValuesY[j] * nodalValuesY[j] +
                nodalValuesZ[k] * nodalValuesZ[k]);
    };

    int cnt = 0;
    for (int k = nodalIdx[4]; k < nodalIdx[5]; ++k) {
        for (int j = nodalIdx[2]; j < nodalIdx[3]; ++j) {
            for (int i = nodalIdx[0]; i < nodalIdx[1]; ++i) {
                nodalData[cnt++] = computeNodalDistance(i,
                                                        j,
                                                        k);
            }
        }
    }
}

void NonLocalPSPData::generateQuadDistance(const FEM &femX,
                                           const FEM &femY,
                                           const FEM &femZ,
                                           Tensor3DMPI &distanceQuad) {
    int numberLocalEntries = distanceQuad.getLocalNumberEntries();
    if (numberLocalEntries == 0)
        return;

    const std::vector<double> &quadValuesX = femX.getPositionQuadPointValues();
    const std::vector<double> &quadValuesY = femY.getPositionQuadPointValues();
    const std::vector<double> &quadValuesZ = femZ.getPositionQuadPointValues();

    std::array<int, 6> quadIdx;
    double *quadData = distanceQuad.getLocalData(quadIdx);

    auto computeQuadDistance = [&quadValuesX, &quadValuesY, &quadValuesZ]
            (int i,
             int j,
             int k) -> double {
        return std::sqrt(
                quadValuesX[i] * quadValuesX[i] + quadValuesY[j] * quadValuesY[j] + quadValuesZ[k] * quadValuesZ[k]);
    };

    int cnt = 0;
    for (int k = quadIdx[4]; k < quadIdx[5]; ++k) {
        for (int j = quadIdx[2]; j < quadIdx[3]; ++j) {
            for (int i = quadIdx[0]; i < quadIdx[1]; ++i) {
                quadData[cnt++] = computeQuadDistance(i,
                                                      j,
                                                      k);
            }
        }
    }
}

void NonLocalPSPData::generateQuadDistanceFromAtom(const FEM &femX,
                                                   const FEM &femY,
                                                   const FEM &femZ,
                                                   const double x,
                                                   const double y,
                                                   const double z,
                                                   Tensor3DMPI &distanceQuad) {
    int numberLocalEntries = distanceQuad.getLocalNumberEntries();
    if (numberLocalEntries == 0)
        return;

    const std::vector<double> &quadValuesX = femX.getPositionQuadPointValues();
    const std::vector<double> &quadValuesY = femY.getPositionQuadPointValues();
    const std::vector<double> &quadValuesZ = femZ.getPositionQuadPointValues();

    std::array<int, 6> quadIdx;
    double *quadData = distanceQuad.getLocalData(quadIdx);

    auto computeQuadDistance = [&quadValuesX, &quadValuesY, &quadValuesZ, &x, &y, &z]
            (int i,
             int j,
             int k) -> double {
        return std::sqrt((quadValuesX[i] - x) * (quadValuesX[i] - x) +
                         (quadValuesY[j] - y) * (quadValuesY[j] - y) +
                         (quadValuesZ[k] - z) * (quadValuesZ[k] - z));
    };

    int cnt = 0;
    for (int k = quadIdx[4]; k < quadIdx[5]; ++k) {
        for (int j = quadIdx[2]; j < quadIdx[3]; ++j) {
            for (int i = quadIdx[0]; i < quadIdx[1]; ++i) {
                quadData[cnt++] = computeQuadDistance(i,
                                                      j,
                                                      k);
            }
        }
    }
}

// used when the file only contains one data set, e.g. localPotential
void NonLocalPSPData::createSplineObject(const std::string &filename,
                                         alglib::spline1dinterpolant &func) {
    alglib::real_2d_array temp;
    // construct spline objects for local part of non-local PSP
    alglib::read_csv(filename.c_str(),
                     '\t',
                     0,
                     temp);

    int n = temp.rows(), incx = temp.getstride(), incy = 1;
    alglib::real_1d_array datax, datay;
    datax.setlength(n);
    datay.setlength(n);
    dcopy_(&n,
           &temp[0][0],
           &incx,
           datax.getcontent(),
           &incy);
    dcopy_(&n,
           &temp[0][1],
           &incx,
           datay.getcontent(),
           &incy);
    alglib::spline1dbuildcubic(datax,
                               datay,
                               func);
}

void NonLocalPSPData::createNonlinearSplineObjects(const std::string &localPSPFilename,
                                                   const std::string &VlFilename,
                                                   const std::string &RlFilename,
                                                   alglib::spline1dinterpolant &funcLocalPSP,
                                                   std::vector<alglib::spline1dinterpolant> &funcDeltaVl,
                                                   std::vector<alglib::spline1dinterpolant> &funcRl) {

    alglib::real_2d_array temp;
    // construct spline objects for local part of non-local PSP
    alglib::read_csv(localPSPFilename.c_str(),
                     '\t',
                     0,
                     temp);

    int n = temp.rows(), incx = temp.getstride(), incy = 1;
    alglib::real_1d_array localPotx, localPoty;
    localPotx.setlength(n);
    localPoty.setlength(n);
    dcopy_(&n,
           &temp[0][0],
           &incx,
           localPotx.getcontent(),
           &incy);
    dcopy_(&n,
           &temp[0][1],
           &incx,
           localPoty.getcontent(),
           &incy);
    alglib::spline1dbuildcubic(localPotx,
                               localPoty,
                               funcLocalPSP);

    // construct spline objects for deltaVl
    alglib::read_csv(VlFilename.c_str(),
                     '\t',
                     0,
                     temp);
    n = temp.rows(), incx = temp.getstride(), incy = 1;
    alglib::real_1d_array deltaVlx, deltaVly;
    deltaVlx.setlength(n);
    deltaVly.setlength(n);
    dcopy_(&n,
           &temp[0][0],
           &incx,
           deltaVlx.getcontent(),
           &incy);
    for (int l = 0; l < lMax; ++l) {
        double minusone = -1.0;
        dcopy_(&n,
               &temp[0][l + 1],
               &incx,
               deltaVly.getcontent(),
               &incy);
        daxpy_(&n,
               &minusone,
               localPoty.getcontent(),
               &incy,
               deltaVly.getcontent(),
               &incy);
        alglib::spline1dbuildcubic(deltaVlx,
                                   deltaVly,
                                   funcDeltaVl[l]);
    }

    // construct spline objects for Rl
    alglib::read_csv(RlFilename.c_str(),
                     '\t',
                     0,
                     temp);
    n = temp.rows(), incx = temp.getstride(), incy = 1;
    alglib::real_1d_array Rlx, Rly;
    Rlx.setlength(n);
    Rly.setlength(n);
    dcopy_(&n,
           &temp[0][0],
           &incx,
           Rlx.getcontent(),
           &incy);
    for (int l = 0; l < lMax; ++l) {
        dcopy_(&n,
               &temp[0][l + 1],
               &incx,
               Rly.getcontent(),
               &incy);
        alglib::spline1dbuildcubic(Rlx,
                                   Rly,
                                   funcRl[l]);
    }
}

void NonLocalPSPData::computeFieldFromSpline(const Tensor3DMPI &distance,
                                             alglib::spline1dinterpolant &func,
                                             Tensor3DMPI &field) {
    int numLocalEntries = distance.getLocalNumberEntries();
    if (numLocalEntries == 0)
        return;

    const double *distanceData = distance.getLocalData();
    int dimx = distance.getGlobalDimension(0);
    int dimy = distance.getGlobalDimension(1);
    int dimz = distance.getGlobalDimension(2);
    double *fieldData = field.getLocalData();
    for (int i = 0; i < numLocalEntries; ++i) {
        fieldData[i] = alglib::spline1dcalc(func,
                                            distanceData[i]);
    }
}

void NonLocalPSPData::computePhi_lm(const std::vector<double> &x,
                                    const std::vector<double> &y,
                                    const std::vector<double> &z,
                                    const int lMax,
                                    const Tensor3DMPI &distance,
                                    const std::vector<Tensor3DMPI> &R_l,
                                    std::vector<Tensor3DMPI> &phi_lm) {
    Tensor3DMPI phi(distance);
    int numPhiLocalEntries = phi.getLocalNumberEntries();
    int cnt = 0;
    double *phiData;
    if (numPhiLocalEntries != 0) {
        std::array<int, 6> phiIdx;
        phiData = phi.getLocalData(phiIdx);
        for (int k = phiIdx[4]; k < phiIdx[5]; ++k) {
            double zvalue = z[k];
            for (int j = phiIdx[2]; j < phiIdx[3]; ++j) {
                for (int i = phiIdx[0]; i < phiIdx[1]; ++i) {
                    phiData[cnt] = std::acos(zvalue / (phiData[cnt] + 1.0e-12));
                    cnt++;
                }
            }
        }
    }

    Tensor3DMPI theta(x.size(),
                      y.size(),
                      z.size(),
                      MPI_COMM_WORLD);
    int numThetaLocalEntries = theta.getLocalNumberEntries();
    double *thetaData;
    if (numThetaLocalEntries != 0) {
        std::array<int, 6> thetaIdx;
        thetaData = theta.getLocalData(thetaIdx);
        cnt = 0;
        for (int j = thetaIdx[2]; j < thetaIdx[3]; ++j) {
            for (int i = thetaIdx[0]; i < thetaIdx[1]; ++i) {
                thetaData[cnt] = std::atan2(y[j],
                                            x[i]);
                cnt++;
            }
        }
        int numElementsPerSlice = cnt;
        int numKOwnedByProc = thetaIdx[5] - thetaIdx[4];
        for (int k = 1; k < numKOwnedByProc; ++k) {
            std::copy(thetaData,
                      thetaData + numElementsPerSlice,
                      thetaData + k * numElementsPerSlice);
        }
    }

    int lcount = -1;
    int numLocalEntries = phi.getLocalNumberEntries();
    if (numLocalEntries != 0) {
        for (int l = 0; l < lMax; ++l) {
            const double *R_lData = R_l[l].getLocalData();
            for (int m = -l; m <= l; ++m) {
                lcount += 1;
                double *phi_lmData = phi_lm[lcount].getLocalData();
                if (m < 0) {
                    for (int i = 0; i < numLocalEntries; ++i) {
                        phi_lmData[i] = R_lData[i] * std::sqrt(2.0) *
                                        boost::math::spherical_harmonic_i(l,
                                                                          -m,
                                                                          phiData[i],
                                                                          thetaData[i]);
                    }
                } else if (m > 0) {
                    for (int i = 0; i < numLocalEntries; ++i) {
                        phi_lmData[i] = R_lData[i] * std::sqrt(2.0) *
                                        boost::math::spherical_harmonic_r(l,
                                                                          m,
                                                                          phiData[i],
                                                                          thetaData[i]);
                    }
                } else {
                    for (int i = 0; i < numLocalEntries; ++i) {
                        phi_lmData[i] =
                                R_lData[i] * boost::math::spherical_harmonic_r(l,
                                                                               0,
                                                                               phiData[i],
                                                                               thetaData[i]);
                    }
                }
            }
        }
    }

}

void NonLocalPSPData::computeC_lm(const int lMax,
                                  const int rankNloc,
                                  const FEM &femNonLocX,
                                  const FEM &femNonLocY,
                                  const FEM &femNonLocZ,
                                  const std::vector<Tensor3DMPI> &deltaVlQuad,
                                  const std::vector<Tensor3DMPI> &phi_lmQuad,
                                  std::vector<double> &C_lm) {
    int taskId;
    MPI_Comm_rank(MPI_COMM_WORLD,
                  &taskId);

    int lcount = -1;
    for (int l = 0; l < lMax; ++l) {
        int numLocalEntries = deltaVlQuad[l].getLocalNumberEntries();
        for (int m = -l; m <= l; ++m) {
            lcount += 1;
            Tensor3DMPI temp(phi_lmQuad[lcount].getGlobalDimension(0),
                             phi_lmQuad[lcount].getGlobalDimension(1),
                             phi_lmQuad[lcount].getGlobalDimension(2),
                             MPI_COMM_WORLD);
            if (numLocalEntries != 0) {
                const double *delta_VlData = deltaVlQuad[l].getLocalData();
                const double *phi_lmData = phi_lmQuad[lcount].getLocalData();
                double *tempData = temp.getLocalData();
                for (int i = 0; i < numLocalEntries; ++i) {
                    tempData[i] = phi_lmData[i] * delta_VlData[i] * phi_lmData[i];
                }
            }
            C_lm[lcount] = KSDFTEnergyFunctional::compute3DIntegralTuckerCuboid(temp,
                                                                                rankNloc,
                                                                                rankNloc,
                                                                                rankNloc,
                                                                                femNonLocX,
                                                                                femNonLocY,
                                                                                femNonLocZ);
            if (taskId == 0) {
                std::cout << "C_lm: " << std::setprecision(16) << 1.0 / C_lm[lcount] << std::endl;
            }
        }
    }
}

void NonLocalPSPData::computeDecomposedNonlocalData(const std::vector<Tensor3DMPI> &DeltaVl,
                                                    const std::vector<Tensor3DMPI> &phi_lm,
                                                    const int lMax,
                                                    const int rankNloc,
                                                    std::vector<Tucker::Tensor *> &sigma,
                                                    std::vector<Tucker::Matrix *> &umat,
                                                    std::vector<Tucker::Matrix *> &vmat,
                                                    std::vector<Tucker::Matrix *> &wmat) {
    int taskId;
    MPI_Comm_rank(MPI_COMM_WORLD,
                  &taskId);

    Tucker::SizeArray rank(3);
    rank[0] = rankNloc;
    rank[1] = rankNloc;
    rank[2] = rankNloc;

    int numberNodesX = DeltaVl[0].getGlobalDimension(0);
    int numberNodesY = DeltaVl[0].getGlobalDimension(1);
    int numberNodesZ = DeltaVl[0].getGlobalDimension(2);

    Tensor3DMPI DeltaVlTimesPhilm(numberNodesX,
                                  numberNodesY,
                                  numberNodesZ,
                                  MPI_COMM_WORLD);
    int numberLocalEntries = DeltaVlTimesPhilm.getLocalNumberEntries();

    int nrowsX, ncolsX, nrowsY, ncolsY, nrowsZ, ncolsZ;

    int lcount = -1;
    for (int l = 0; l < lMax; ++l) {
        for (int m = -l; m <= l; ++m) {
            lcount += 1;
            // compute DeltaVl * phi_lm
            const double *deltaVlData;
            const double *philmData;
            double *DeltaVlTimesPhilmData;
            if (numberLocalEntries != 0) {
                deltaVlData = DeltaVl[l].getLocalData();
                philmData = phi_lm[lcount].getLocalData();
                DeltaVlTimesPhilmData = DeltaVlTimesPhilm.getLocalData();
                for (int i = 0; i < numberLocalEntries; ++i) {
                    DeltaVlTimesPhilmData[i] = philmData[i] * deltaVlData[i];
                }
            }

            // compuete the Tucker decomposition of DeltaVl * phi_lm
            const TuckerMPI::TuckerTensor *ttensor = TuckerMPI::STHOSVD(DeltaVlTimesPhilm.getTensor(),
                                                                        &rank,
                                                                        true,
                                                                        false);

            // allgather the distributed core tensor to sigma
            sigma[lcount] = Tucker::MemoryManager::safe_new<Tucker::Tensor>(rank);
            TensorUtils::allreduce_tensor(ttensor->G,
                                          sigma[lcount]);

            // copy the factor matrices on 3 directions
            nrowsX = ttensor->U[0]->nrows(), ncolsX = ttensor->U[0]->ncols();
            umat[lcount] = Tucker::MemoryManager::safe_new<Tucker::Matrix>(nrowsX,
                                                                           ncolsX);
            std::copy(ttensor->U[0]->data(),
                      ttensor->U[0]->data() + nrowsX * ncolsX,
                      umat[lcount]->data());

            nrowsY = ttensor->U[1]->nrows(), ncolsY = ttensor->U[1]->ncols();
            vmat[lcount] = Tucker::MemoryManager::safe_new<Tucker::Matrix>(nrowsY,
                                                                           ncolsY);
            std::copy(ttensor->U[1]->data(),
                      ttensor->U[1]->data() + nrowsY * ncolsY,
                      vmat[lcount]->data());

            nrowsZ = ttensor->U[2]->nrows(), ncolsZ = ttensor->U[2]->ncols();
            wmat[lcount] = Tucker::MemoryManager::safe_new<Tucker::Matrix>(nrowsZ,
                                                                           ncolsZ);
            std::copy(ttensor->U[2]->data(),
                      ttensor->U[2]->data() + nrowsZ * ncolsZ,
                      wmat[lcount]->data());

            // clean the memory
            Tucker::MemoryManager::safe_delete(ttensor);
        }
    }

}

void NonLocalPSPData::projectNodeOntoQuad(const std::vector<Tucker::Matrix *> &matNode,
                                          const FEM &fem,
                                          std::vector<Tucker::Matrix *> &matQuad) {
    // do some checks, assertion will not be executed in release mode
    int nrowsNode, nrowsQuad, ncols;
    int numberNodes = fem.getTotalNumberNodes();
    int numberQuads = fem.getTotalNumberQuadPoints();

    assert(matNode.size() == matQuad.size());

    for (int i = 0; i < matNode.size(); ++i) {
        assert(matNode[i]->nrows() == numberNodes);
        nrowsNode = numberNodes;
        nrowsQuad = numberQuads;
        ncols = matNode[i]->ncols();
        matQuad[i] = Tucker::MemoryManager::safe_new<Tucker::Matrix>(nrowsQuad,
                                                                     ncols);
        for (int icols = 0; icols < ncols; ++icols) {
            std::vector<double> tempNode(nrowsNode), tempQuad, tempDiffQuad;
            std::copy(matNode[i]->data() + icols * nrowsNode,
                      matNode[i]->data() + (icols + 1) * nrowsNode,
                      tempNode.begin());
            fem.computeFieldAndDiffFieldAtAllQuadPoints(tempNode,
                                                        tempQuad,
                                                        tempDiffQuad);
            std::copy(tempQuad.begin(),
                      tempQuad.end(),
                      matQuad[i]->data() + icols * nrowsQuad);
        }
    }
}

int NonLocalPSPData::getLMax() const {
    return lMax;
}

int NonLocalPSPData::getRankNloc() const {
    return rankNloc;
}

const std::vector<Tucker::Tensor *> &NonLocalPSPData::getSigmaQuad() const {
    return sigmaQuad;
}

const std::vector<Tucker::Matrix *> &NonLocalPSPData::getUmatQuad() const {
    return umatQuad;
}

const std::vector<Tucker::Matrix *> &NonLocalPSPData::getVmatQuad() const {
    return vmatQuad;
}

const std::vector<Tucker::Matrix *> &NonLocalPSPData::getWmatQuad() const {
    return wmatQuad;
}

const std::vector<Tucker::Tensor *> &NonLocalPSPData::getSigmaNode() const {
    return sigmaNode;
}

const std::vector<Tucker::Matrix *> &NonLocalPSPData::getUmatNode() const {
    return umatNode;
}

const std::vector<Tucker::Matrix *> &NonLocalPSPData::getVmatNode() const {
    return vmatNode;
}

const std::vector<Tucker::Matrix *> &NonLocalPSPData::getWmatNode() const {
    return wmatNode;
}

const std::vector<Tucker::Matrix *> &NonLocalPSPData::getUmatNodeInterpolatedQuad() const {
    return umatNodeInterpolatedQuad;
}

const std::vector<Tucker::Matrix *> &NonLocalPSPData::getVmatNodeInterpolatedQuad() const {
    return vmatNodeInterpolatedQuad;
}

const std::vector<Tucker::Matrix *> &NonLocalPSPData::getWmatNodeInterpolatedQuad() const {
    return wmatNodeInterpolatedQuad;
}

const std::vector<double> &NonLocalPSPData::getC_lm() const {
    return C_lm;
}

int NonLocalPSPData::getNumberNonLocalAtoms() const {
    return numberNonLocalAtoms;
}

NonLocalPSPData::~NonLocalPSPData() {
    for (auto &i: sigmaQuad) {
        if (i) {
            Tucker::MemoryManager::safe_delete(i);
        }
    }
    for (auto &i: umatQuad) {
        if (i) {
            Tucker::MemoryManager::safe_delete(i);
        }
    }
    for (auto &i: vmatQuad) {
        if (i) {
            Tucker::MemoryManager::safe_delete(i);
        }
    }
    for (auto &i: wmatQuad) {
        if (i) {
            Tucker::MemoryManager::safe_delete(i);
        }
    }
    for (auto &i: sigmaNode) {
        if (i) {
            Tucker::MemoryManager::safe_delete(i);
        }
    }
    for (auto &i: umatNode) {
        if (i) {
            Tucker::MemoryManager::safe_delete(i);
        }
    }
    for (auto &i: vmatNode) {
        if (i) {
            Tucker::MemoryManager::safe_delete(i);
        }
    }
    for (auto &i: wmatNode) {
        if (i) {
            Tucker::MemoryManager::safe_delete(i);
        }
    }
    for (auto &i: umatNodeInterpolatedQuad) {
        if (i) {
            Tucker::MemoryManager::safe_delete(i);
        }
    }
    for (auto &i: vmatNodeInterpolatedQuad) {
        if (i) {
            Tucker::MemoryManager::safe_delete(i);
        }
    }
    for (auto &i: wmatNodeInterpolatedQuad) {
        if (i) {
            Tucker::MemoryManager::safe_delete(i);

        }
    }
}
