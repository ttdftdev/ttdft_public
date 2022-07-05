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

#include <cmath>
#include <vector>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include "KSDFTPotential.h"
#include "../tensor/TensorUtils.h"
#include "../utils/FileReader.h"

namespace {
    extern "C" {
    int daxpy_(int *n,
               double *sa,
               double *sx,
               int *incx,
               double *sy,
               int *incy);
    }

    void expansionIntegral(const int &mode,
                           const int &numberTermsExpansion,
                           const int &rank,
                           const int &squeezed,
                           const FEM &fem,
                           const FEM &femElectro,
                           const std::vector<double> &alpha,
                           const Tucker::Matrix *U,
                           const TuckerMPI::Tensor *effPot,
                           std::vector<Tucker::Matrix *> &integralFunc);

    void expansionIntegral(const int &mode,
                           const int &numberTermsExpansion,
                           const int &rank,
                           const int &localSize,
                           const int &squeezed,
                           const FEM &fem,
                           const FEM &femElectro,
                           const std::vector<double> &alpha,
                           const Tucker::Matrix *U,
                           const TuckerMPI::Tensor *effPot,
                           std::vector<std::vector<std::vector<double> > > &integralFunc);

    void computeExpansionIntegral(const int &numberTermsExpansion,
                                  const int rank[],
                                  const int &squeezed,
                                  const FEM &femX,
                                  const FEM &femY,
                                  const FEM &femZ,
                                  const FEM &femElectroX,
                                  const FEM &femElectroY,
                                  const FEM &femElectroZ,
                                  const std::vector<double> &alpha,
                                  Tucker::Matrix *U[],
                                  const TuckerMPI::Tensor *effPot,
                                  std::vector<Tucker::Matrix *> &integralFuncX,
                                  std::vector<Tucker::Matrix *> &integralFuncY,
                                  std::vector<Tucker::Matrix *> &integralFuncZ);

    void computeHatreePotFromExpansionIntegral(const int &numberTermsExpansion,
                                               const int *localSize,
                                               const int *rankRho,
                                               const double *localCoreCopyData,
                                               const std::vector<double> &omega,
                                               const std::vector<std::vector<std::vector<double> > > &localFuncX,
                                               const std::vector<std::vector<std::vector<double> > > &localFuncY,
                                               const std::vector<std::vector<std::vector<double> > > &localFuncZ,
                                               double *localTensorData);

    void computeHatreePotFromExpansionIntegral(const int &numberTermsExpansion,
                                               const std::vector<double> &omega,
                                               TuckerMPI::Tensor *core,
                                               const std::vector<Tucker::Matrix *> &localFuncX,
                                               const std::vector<Tucker::Matrix *> &localFuncY,
                                               const std::vector<Tucker::Matrix *> &localFuncZ,
                                               TuckerMPI::Tensor *effPot);

    void computeHatreePotFromExpansionIntegralNew(const int &numberTermsExpansion,
                                                  const std::vector<double> &omega,
                                                  TuckerMPI::Tensor *core,
                                                  const std::vector<Tucker::Matrix *> &localFuncX,
                                                  const std::vector<Tucker::Matrix *> &localFuncY,
                                                  const std::vector<Tucker::Matrix *> &localFuncZ,
                                                  TuckerMPI::Tensor *effPot);
}

KSDFTPotential::KSDFTPotential(const FEM &femX,
                               const FEM &femY,
                               const FEM &femZ,
                               const FEM &femElectroX,
                               const FEM &femElectroY,
                               const FEM &femElectroZ,
                               const std::string &alphafile,
                               const std::string &omegafile,
                               const double Asquare
) :
        femX(femX),
        femY(femY),
        femZ(femZ),
        femElectroX(femElectroX),
        femElectroY(femElectroY),
        femElectroZ(femElectroZ),
        alphafile(alphafile),
        omegafile(omegafile),
        Asquare(Asquare) {}

void KSDFTPotential::computeLDAExchangePot(const TuckerMPI::Tensor *rhoGrid,
                                           TuckerMPI::Tensor *effPot,
                                           InsertMode addv) {
    // try-catch are the tensors initialized properly
    try {
        if (rhoGrid->getGlobalSize(0) != femX.getNumberQuadPointsPerElement() * femX.getNumberElements()) {
            const std::string message("The x-dimension of rhoGrid in computeLDAExchangePot does not match.");
            throw std::out_of_range(message);
        } else if (rhoGrid->getGlobalSize(1) != femY.getNumberQuadPointsPerElement() * femY.getNumberElements()) {
            const std::string message("The y-dimension of rhoGrid in computeLDAExchangePot does not match.");
            throw std::out_of_range(message);
        } else if (rhoGrid->getGlobalSize(2) != femZ.getNumberQuadPointsPerElement() * femZ.getNumberElements()) {
            const std::string message("The z-dimension of rhoGrid in computeLDAExchangePot does not match.");
            throw std::out_of_range(message);
        } else if (effPot->getGlobalSize(0) != femX.getNumberQuadPointsPerElement() * femX.getNumberElements()) {
            const std::string message("The x-dimension of effPot in computeLDAExchangePot does not match.");
            throw std::out_of_range(message);
        } else if (effPot->getGlobalSize(1) != femY.getNumberQuadPointsPerElement() * femY.getNumberElements()) {
            const std::string message("The y-dimension of effPot in computeLDAExchangePot does not match.");
            throw std::out_of_range(message);
        } else if (effPot->getGlobalSize(2) != femZ.getNumberQuadPointsPerElement() * femZ.getNumberElements()) {
            const std::string message("The z-dimension of effPot in computeLDAExchangePot does not match.");
            throw std::out_of_range(message);
        } else if (addv != INSERT_VALUES && addv != ADD_VALUES) {
            const std::string message("Not supported insertmode, only accept INSERT_VALUES or ADD_VALUES");
            throw std::out_of_range(message);
        }
    } catch (std::exception &e) {
        std::cerr << e.what();
        std::terminate();
    }

    // create pointers pointing to the local tensors of the distributed rhoGrid and effPot
    const Tucker::Tensor *localRhoGrid = rhoGrid->getLocalTensor();
    Tucker::Tensor *localEffPot = effPot->getLocalTensor();


    // If the mode is set to be INSERT_VALUES, zero all the elements in the local-stored tensor for insert mode.
    if (addv == INSERT_VALUES) {
        localEffPot->initialize();
    }

    // check if the local distribution is the same for rhoGrid tensor and effPot tensor before calculating the exchange part of LDA potential.
    try {
        if (localRhoGrid->getNumElements() != localEffPot->getNumElements()) {
            const std::string message(
                    "The number of elements in local rhoGrid tensor does not match the number in local effPot in computeLDAExchangePot.");
            throw std::logic_error(message);
        }
    } catch (std::exception &e) {
        std::cerr << e.what();
        std::terminate();
    }

    // precompute some constant for the use of the exchange part of LDA potential calculation
    const double exchConst = -(3.0 / 4.0) * std::cbrt(3.0 / M_PI);

    // calculate the the exchange part of LDA potential for each element in the effPot
    for (int i = 0; i < localRhoGrid->getNumElements(); ++i) {
        localEffPot->data()[i] += exchConst * (4.0 / 3.0) * std::pow(localRhoGrid->data()[i],
                                                                     1.0 / 3.0);
    }
}

void KSDFTPotential::computeLDACorrelationPot(const TuckerMPI::Tensor *rhoGrid,
                                              TuckerMPI::Tensor *effPot,
                                              InsertMode addv) {
    // try-catch are the tensors initialized properly
    try {
        if (rhoGrid->getGlobalSize(0) != femX.getNumberQuadPointsPerElement() * femX.getNumberElements()) {
            const std::string message("The x-dimension of rhoGrid in computeLDACorrelationPot does not match.");
            throw std::out_of_range(message);
        } else if (rhoGrid->getGlobalSize(1) != femY.getNumberQuadPointsPerElement() * femY.getNumberElements()) {
            const std::string message("The y-dimension of rhoGrid in computeLDACorrelationPot does not match.");
            throw std::out_of_range(message);
        } else if (rhoGrid->getGlobalSize(2) != femZ.getNumberQuadPointsPerElement() * femZ.getNumberElements()) {
            const std::string message("The z-dimension of rhoGrid in computeLDACorrelationPot does not match.");
            throw std::out_of_range(message);
        } else if (effPot->getGlobalSize(0) != femX.getNumberQuadPointsPerElement() * femX.getNumberElements()) {
            const std::string message("The x-dimension of effPot in computeLDACorrelationPot does not match.");
            throw std::out_of_range(message);
        } else if (effPot->getGlobalSize(1) != femY.getNumberQuadPointsPerElement() * femY.getNumberElements()) {
            const std::string message("The y-dimension of effPot in computeLDACorrelationPot does not match.");
            throw std::out_of_range(message);
        } else if (effPot->getGlobalSize(2) != femZ.getNumberQuadPointsPerElement() * femZ.getNumberElements()) {
            const std::string message("The z-dimension of effPot in computeLDACorrelationPot does not match.");
            throw std::out_of_range(message);
        } else if (addv != INSERT_VALUES && addv != ADD_VALUES) {
            const std::string message("Not supported insertmode, only accept INSERT_VALUES or ADD_VALUES");
            throw std::out_of_range(message);
        }
    } catch (std::exception &e) {
        std::cerr << e.what();
        std::terminate();
    }

    // create pointers pointing to the local tensors of the distributed rhoGrid and effPot
    const Tucker::Tensor *localRhoGrid = rhoGrid->getLocalTensor();
    Tucker::Tensor *localEffPot = effPot->getLocalTensor();


    // If the mode is set to be INSERT_VALUES, zero all the elements in the local-stored tensor for insert mode.
    if (addv == INSERT_VALUES) {
        localEffPot->initialize();
    }

    // check if the local distribution is the same for rhoGrid tensor and effPot tensor before calculating the correlational part of LDA potential.
    try {
        if (localRhoGrid->getNumElements() != localEffPot->getNumElements()) {
            const std::string message(
                    "The number of elements in local rhoGrid tensor does not match the number in local effPot in computeLDACorrelationPot.");
            throw std::logic_error(message);
        }
    } catch (std::exception &e) {
        std::cerr << e.what();
        std::terminate();
    }

    // precompute some constant for the use of the correlational part of LDA potential calculation
    const double beta1 = 1.0529;
    const double beta2 = 0.3334;
    const double A = 0.0311;
    const double B = -0.048;
    const double C = 0.002;
    const double D = -0.0116;
    const double gamma = -0.1423;

    // compute the correlational part of the LDA potential for eache element in the local tensor;
    for (unsigned i = 0; i < localRhoGrid->getNumElements(); ++i) {
        double rs = std::cbrt(3.0 / (4.0 * M_PI * localRhoGrid->data()[i]));
        if (rs >= 1.0) {
            double num = gamma * (1 + 7.0 / 6.0 * beta1 * std::sqrt(rs) + 4.0 / 3.0 * beta2 * rs);
            double denom = std::pow(1 + beta1 * std::sqrt(rs) + beta2 * rs,
                                    2.0);
            localEffPot->data()[i] += num / denom;
        } else {
            localEffPot->data()[i] += A * std::log(rs) + (B - 1.0 / 3.0 * A) + 2.0 / 3.0 * C * rs * std::log(rs) +
                                      1.0 / 33.0 * (2 * D - C) * rs;
        }
    }
}

void KSDFTPotential::computeHartreePotNew(const TuckerMPI::TuckerTensor *rhoGridTT,
                                          TuckerMPI::Tensor *effPot,
                                          InsertMode addv) {
    std::vector<double> alpha, omega;
    Utils::ReadSingleColumnFile(alphafile,
                                alpha);
    Utils::ReadSingleColumnFile(omegafile,
                                omega);

    int numberExpansion = alpha.size();

    assert(alpha.size() == omega.size());
    for (auto i = 0; i != numberExpansion; ++i) {
        omega[i] *= (1.0 / std::sqrt(Asquare));
        alpha[i] *= (1.0 / Asquare);
    }

    Tucker::Tensor *localEffPot = effPot->getLocalTensor();

    // If the mode is set to be INSERT_VALUES, zero all the elements in the local-stored tensor for insert mode.
    if (addv == INSERT_VALUES) {
        localEffPot->initialize();
    }

    bool squeezed = false;

    int taskId, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD,
                  &taskId);
    MPI_Comm_size(MPI_COMM_WORLD,
                  &nprocs);

    int rankRho[3] = {rhoGridTT->G->getGlobalSize(0),
                      rhoGridTT->G->getGlobalSize(1),
                      rhoGridTT->G->getGlobalSize(2)};

    unsigned numberTermsExpansion = omega.size();

    std::vector<Tucker::Matrix *> localFuncX(numberExpansion);
    std::vector<Tucker::Matrix *> localFuncY(numberExpansion);
    std::vector<Tucker::Matrix *> localFuncZ(numberExpansion);

    computeExpansionIntegral(numberExpansion,
                             rankRho,
                             squeezed,
                             femX,
                             femY,
                             femZ,
                             femElectroX,
                             femElectroY,
                             femElectroZ,
                             alpha,
                             rhoGridTT->U,
                             effPot,
                             localFuncX,
                             localFuncY,
                             localFuncZ);

    computeHatreePotFromExpansionIntegralNew(numberTermsExpansion,
                                             omega,
                                             rhoGridTT->G,
                                             localFuncX,
                                             localFuncY,
                                             localFuncZ,
                                             effPot);

    for (int i = 0; i < numberExpansion; ++i) {
        Tucker::MemoryManager::safe_delete(localFuncX[i]);
        Tucker::MemoryManager::safe_delete(localFuncY[i]);
        Tucker::MemoryManager::safe_delete(localFuncZ[i]);
    }
}

void KSDFTPotential::computeEvanescentPSPOnGrid(const std::vector<std::vector<double> > &atomInformation,
                                                const double R,
                                                const double alpha,
                                                Tensor3DMPI &effPot,
                                                TensorOperation op) {

    assert((op == TENSOR_INSERT) || (op == TENSOR_ADD));
    // If the mode is set to be INSERT_VALUES, zero all the elements in the local-stored tensor for insert mode.
    if (op == TENSOR_INSERT) {
        effPot.setEntriesZero();
    }

    const std::vector<double> quadCoordX = femX.getPositionQuadPointValues();
    const std::vector<double> quadCoordY = femY.getPositionQuadPointValues();
    const std::vector<double> quadCoordZ = femZ.getPositionQuadPointValues();

    std::array<int, 6> effPotGlobalIdx;
    double *effPotLocalData = effPot.getLocalData(effPotGlobalIdx);

    int numAtoms = atomInformation.size();

    for (int iAtom = 0; iAtom < numAtoms; ++iAtom) {
        double atomCharge = atomInformation[iAtom][0];
        double atomCoordX = atomInformation[iAtom][1];
        double atomCoordY = atomInformation[iAtom][2];
        double atomCoordZ = atomInformation[iAtom][3];

        //TODO modify this part to make it general to different atoms!!!
        //Currently only support system with only one atom type
        double beta = (alpha * alpha * alpha - 2.0 * alpha) / (4.0 * (alpha * alpha - 1.0));
        double A = 0.5 * alpha * (alpha - 2.0 * beta);
        double constZoverR = -atomCharge / R;

        int cnt = 0;
        for (int k = effPotGlobalIdx[4]; k < effPotGlobalIdx[5]; ++k) {
            for (int j = effPotGlobalIdx[2]; j < effPotGlobalIdx[3]; ++j) {
                for (int i = effPotGlobalIdx[0]; i < effPotGlobalIdx[1]; ++i) {
                    double r = std::sqrt((quadCoordX[i] - atomCoordX) * (quadCoordX[i] - atomCoordX) +
                                         (quadCoordY[j] - atomCoordY) * (quadCoordY[j] - atomCoordY) +
                                         (quadCoordZ[k] - atomCoordZ) * (quadCoordZ[k] - atomCoordZ));
                    double x = r / R;
                    effPotLocalData[cnt++] +=
                            constZoverR * ((1.0 / x) * (1 - (1 + beta * x) * std::exp(-alpha * x)) - A * std::exp(-x));
                }
            }
        }
    }
}

void
KSDFTPotential::computeHartreePotNew(const TuckerMPI::TuckerTensor *rhoGridTT,
                                     Tensor3DMPI &effPot,
                                     TensorOperation addv) {
    assert((addv == TENSOR_INSERT) || (addv == TENSOR_ADD));
    InsertMode insertMode;
    if (addv == TENSOR_INSERT) {
        insertMode = INSERT_VALUES;
    } else if (addv == TENSOR_ADD) {
        insertMode = ADD_VALUES;
    }
    computeHartreePotNew(rhoGridTT,
                         effPot.getTensor(),
                         insertMode);
}

void KSDFTPotential::computeLDAExchangePot(const Tensor3DMPI &rhoGrid,
                                           Tensor3DMPI &effPot,
                                           TensorOperation addv) {
    assert((addv == TENSOR_INSERT) || (addv == TENSOR_ADD));
    InsertMode insertMode;
    if (addv == TENSOR_INSERT) {
        insertMode = INSERT_VALUES;
    } else if (addv == TENSOR_ADD) {
        insertMode = ADD_VALUES;
    }
    computeLDAExchangePot(rhoGrid.getTensor(),
                          effPot.getTensor(),
                          insertMode);

}

void KSDFTPotential::computeLDACorrelationPot(const Tensor3DMPI &rhoGrid,
                                              Tensor3DMPI &effPot,
                                              TensorOperation addv) {
    assert((addv == TENSOR_INSERT) || (addv == TENSOR_ADD));
    InsertMode insertMode;
    if (addv == TENSOR_INSERT) {
        insertMode = INSERT_VALUES;
    } else if (addv == TENSOR_ADD) {
        insertMode = ADD_VALUES;
    }
    computeLDACorrelationPot(rhoGrid.getTensor(),
                             effPot.getTensor(),
                             insertMode);
}

// Definition of functions used in this source file
namespace {
    void computeExpansionIntegral(const int &numberTermsExpansion,
                                  const int rank[],
                                  const int &squeezed,
                                  const FEM &femX,
                                  const FEM &femY,
                                  const FEM &femZ,
                                  const FEM &femElectroX,
                                  const FEM &femElectroY,
                                  const FEM &femElectroZ,
                                  const std::vector<double> &alpha,
                                  Tucker::Matrix *U[],
                                  const TuckerMPI::Tensor *effPot,
                                  std::vector<Tucker::Matrix *> &integralFuncX,
                                  std::vector<Tucker::Matrix *> &integralFuncY,
                                  std::vector<Tucker::Matrix *> &integralFuncZ) {
        int taskId, nprocs;
        MPI_Comm_rank(MPI_COMM_WORLD,
                      &taskId);
        MPI_Comm_size(MPI_COMM_WORLD,
                      &nprocs);

        const std::vector<double> &positionQuadPointValuesX = femX.getPositionQuadPointValues();
        const std::vector<double> &positionQuadPointValuesElectroX = femElectroX.getPositionQuadPointValues();
        const std::vector<double> &positionQuadPointValuesY = femY.getPositionQuadPointValues();
        const std::vector<double> &positionQuadPointValuesElectroY = femElectroY.getPositionQuadPointValues();
        const std::vector<double> &positionQuadPointValuesZ = femZ.getPositionQuadPointValues();
        const std::vector<double> &positionQuadPointValuesElectroZ = femElectroZ.getPositionQuadPointValues();

        int rankX = rank[0];
        int rankY = rank[1];
        int rankZ = rank[2];
        const Tucker::Matrix *UX = U[0];
        const Tucker::Matrix *UY = U[1];
        const Tucker::Matrix *UZ = U[2];

        const TuckerMPI::Distribution *distribution = effPot->getDistribution();
        const Tucker::SizeArray *offsetX = distribution->getMap(0,
                                                                squeezed)->getOffsets();
        const Tucker::SizeArray *offsetY = distribution->getMap(1,
                                                                squeezed)->getOffsets();
        const Tucker::SizeArray *offsetZ = distribution->getMap(2,
                                                                squeezed)->getOffsets();
        int procGrid[3];
        distribution->getProcessorGrid()->getCoordinates(procGrid);
        int istartGlobal = (*offsetX)[procGrid[0]];
        int iendGlobal = (*offsetX)[procGrid[0] + 1];
        int jstartGlobal = (*offsetY)[procGrid[1]];
        int jendGlobal = (*offsetY)[procGrid[1] + 1];
        int kstartGlobal = (*offsetZ)[procGrid[2]];
        int kendGlobal = (*offsetZ)[procGrid[2] + 1];

        /**
         * Compute UX
         * Notice: because of the way Tensor3DMPI is stored, UZ only requires the local portion corresponding to the tensor reconstruction
         */
        // initialize matrices in integralFuncY
        // nrowsMatZ = # of grids owned by the processor along Z direction

        int nrowsMatX = iendGlobal - istartGlobal, ncolsMatX = rankX;
        for (int i = 0; i < numberTermsExpansion; ++i) {
            if (integralFuncX[i] != 0) {
                Tucker::MemoryManager::safe_delete(integralFuncX[i]);
            }
            integralFuncX[i] = Tucker::MemoryManager::safe_new<Tucker::Matrix>(nrowsMatX,
                                                                               ncolsMatX);
        }
        int numberFemElectroXQuadPoints = positionQuadPointValuesElectroX.size();
        if (nrowsMatX != 0 && ncolsMatX != 0) {
            const double *Ux_data = UX->data();
            for (auto iterm = 0; iterm != numberTermsExpansion; ++iterm) {
                int cnt = 0;
                double *integralFuncXData = integralFuncX[iterm]->data();
                for (auto irank = 0; irank < rankX; ++irank) {
                    const double *rhoEig = Ux_data + irank * numberFemElectroXQuadPoints;
                    for (auto inode = istartGlobal; inode < iendGlobal; ++inode) {
                        std::vector<double> temp(numberFemElectroXQuadPoints,
                                                 positionQuadPointValuesX[inode]);
                        for (auto irho = 0; irho < numberFemElectroXQuadPoints; ++irho) {
                            temp[irho] -= positionQuadPointValuesElectroX[irho];
                            temp[irho] = std::exp(-alpha[iterm] * temp[irho] * temp[irho]) * rhoEig[irho];
                        }
                        integralFuncXData[cnt++] = femElectroX.integrate_by_quad_values(temp);
                    }
                }
            }
        }

        /**
         * Compute UY
         * Notice: because of the way Tensor3DMPI is stored, UY only requires the local portion corresponding to the tensor reconstruction
         */
        // initialize matrices in integralFuncY
        // nrowsMatY = # of grids owned by the processor along Y direction
        int nrowsMatY = jendGlobal - jstartGlobal, ncolsMatY = rankY;
        for (int i = 0; i < numberTermsExpansion; ++i) {
            if (integralFuncY[i] != 0) {
                Tucker::MemoryManager::safe_delete(integralFuncY[i]);
            }
            integralFuncY[i] = Tucker::MemoryManager::safe_new<Tucker::Matrix>(nrowsMatY,
                                                                               ncolsMatY);
        }
        int numberFemElectroYQuadPoints = positionQuadPointValuesElectroY.size();

        if (nrowsMatY != 0 && ncolsMatY != 0) {
            const double *Uy_data = UY->data();
            for (auto iterm = 0; iterm != numberTermsExpansion; ++iterm) {
                int cnt = 0;
                double *integralFuncYData = integralFuncY[iterm]->data();
                for (auto irank = 0; irank < rankY; ++irank) {
                    const double *rhoEig = Uy_data + irank * numberFemElectroYQuadPoints;
                    for (auto inode = jstartGlobal; inode < jendGlobal; ++inode) {
                        std::vector<double> temp(numberFemElectroYQuadPoints,
                                                 positionQuadPointValuesY[inode]);
                        for (auto irho = 0; irho < numberFemElectroYQuadPoints; ++irho) {
                            temp[irho] -= positionQuadPointValuesElectroY[irho];
                            temp[irho] = std::exp(-alpha[iterm] * temp[irho] * temp[irho]) * rhoEig[irho];
                        }
                        integralFuncYData[cnt++] = femElectroY.integrate_by_quad_values(temp);
                    }
                }
            }
        }


        /**
         * Compute UZ
         * Notice: because of the way Tensor3DMPI is stored, UZ only requires the local portion corresponding to the tensor reconstruction
         */
        // initialize matrices in integralFuncY
        // nrowsMatZ = # of grids owned by the processor along Z direction
        int nrowsMatZ = kendGlobal - kstartGlobal, ncolsMatZ = rankZ;
        for (int i = 0; i < numberTermsExpansion; ++i) {
            if (integralFuncZ[i] != 0) {
                Tucker::MemoryManager::safe_delete(integralFuncZ[i]);
            }
            integralFuncZ[i] = Tucker::MemoryManager::safe_new<Tucker::Matrix>(nrowsMatZ,
                                                                               ncolsMatZ);
        }
        int numberFemElectroZQuadPoints = positionQuadPointValuesElectroZ.size();

        if (nrowsMatZ != 0 && ncolsMatZ != 0) {
            const double *Uz_data = UZ->data();
            for (auto iterm = 0; iterm != numberTermsExpansion; ++iterm) {
                int cnt = 0;
                double *integralFuncZData = integralFuncZ[iterm]->data();
                for (auto irank = 0; irank < rankZ; ++irank) {
                    const double *rhoEig = Uz_data + irank * numberFemElectroZQuadPoints;
                    for (auto inode = kstartGlobal; inode < kendGlobal; ++inode) {
                        std::vector<double> temp(numberFemElectroZQuadPoints,
                                                 positionQuadPointValuesZ[inode]);
                        for (auto irho = 0; irho < numberFemElectroZQuadPoints; ++irho) {
                            temp[irho] -= positionQuadPointValuesElectroZ[irho];
                            temp[irho] = std::exp(-alpha[iterm] * temp[irho] * temp[irho]) * rhoEig[irho];
                        }
                        integralFuncZData[cnt++] = femElectroZ.integrate_by_quad_values(temp);
                    }
                }
            }
        }
    }

    void computeHatreePotFromExpansionIntegralNew(const int &numberTermsExpansion,
                                                  const std::vector<double> &omega,
                                                  TuckerMPI::Tensor *core,
                                                  const std::vector<Tucker::Matrix *> &localFuncX,
                                                  const std::vector<Tucker::Matrix *> &localFuncY,
                                                  const std::vector<Tucker::Matrix *> &localFuncZ,
                                                  TuckerMPI::Tensor *effPot) {

        int taskId;
        MPI_Comm_rank(MPI_COMM_WORLD,
                      &taskId);

        Tucker::Tensor *seqcore = Tucker::MemoryManager::safe_new<Tucker::Tensor>(core->getGlobalSize());
        TensorUtils::allreduce_tensor(core,
                                      seqcore);

        for (auto iterm = 0; iterm != numberTermsExpansion; ++iterm) {
            if (localFuncX[iterm]->getNumElements() != 0 &&
                localFuncY[iterm]->getNumElements() != 0 &&
                localFuncZ[iterm]->getNumElements() != 0) {
                Tucker::Tensor *temp;
                Tucker::Tensor *reconstructedTensor;
                temp = seqcore;
                reconstructedTensor = Tucker::ttm(temp,
                                                  0,
                                                  localFuncX[iterm]);
                temp = reconstructedTensor;
                reconstructedTensor = Tucker::ttm(temp,
                                                  1,
                                                  localFuncY[iterm]);
                Tucker::MemoryManager::safe_delete(temp);
                temp = reconstructedTensor;
                reconstructedTensor = Tucker::ttm(temp,
                                                  2,
                                                  localFuncZ[iterm]);
                Tucker::MemoryManager::safe_delete(temp);

                int localNumEntries = effPot->getLocalNumEntries();
                if (localNumEntries != 0) {
                    double *effPotLocalData = effPot->getLocalTensor()->data();
                    double *reconstructedTensorData = reconstructedTensor->data();
                    double omegai = omega[iterm];
                    int incx = 1;
                    int incy = 1;

                    assert(reconstructedTensor->getNumElements() == localNumEntries);
                    daxpy_(&localNumEntries,
                           &omegai,
                           reconstructedTensorData,
                           &incx,
                           effPotLocalData,
                           &incy);
                }
                Tucker::MemoryManager::safe_delete(reconstructedTensor);
            }
        }
        Tucker::MemoryManager::safe_delete(seqcore);
    }
}