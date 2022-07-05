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
#include <iomanip>
#include <TuckerMPI.hpp>
#include "KSDFTEnergyFunctional.h"

void KSDFTEnergyFunctional::computeLDAExchangeCorrelationEnergy(const Tensor3DMPI &rhoGrid,
                                                                Tensor3DMPI &exchGrid,
                                                                TensorOperation op) {
    assert(op == TENSOR_ADD || op == TENSOR_INSERT);

    if (op == TENSOR_INSERT)
        exchGrid.setEntriesZero();

    double exchConst = -(3.0 / 4.0) * std::cbrt(3.0 / M_PI);
    double C0 = std::cbrt(3.0 / (4.0 * M_PI));

    std::array<int, 6> gridGlobalIdx;
    const double *rhoGridLocal = rhoGrid.getLocalData();
    double *exchGridLocal = exchGrid.getLocalData();

    int localNumEntries = rhoGrid.getLocalNumberEntries();

    for (int i = 0; i < localNumEntries; ++i) {
        double exchange = exchConst * std::pow(rhoGridLocal[i],
                                               4.0 / 3.0);
        double rs = C0 / (std::cbrt(rhoGridLocal[i]) + 1.0e-12);
        double correlation;
        if (rs >= 1.0) {
            correlation = (gamma / (1.0 + beta1 * std::sqrt(rs) + beta2 * rs)) * rhoGridLocal[i];
        } else {
            correlation = (A * std::log(rs) + B + C * rs * std::log(rs) + D * rs) * rhoGridLocal[i];
        }
        exchGridLocal[i] += exchange + correlation;
    }
}

void KSDFTEnergyFunctional::computeHartreeEnergy(const Tensor3DMPI &rhoGrid,
                                                 const Tensor3DMPI &potHartGrid,
                                                 Tensor3DMPI &potHartEnergyGrid,
                                                 TensorOperation op) {
    assert(op == TENSOR_ADD || op == TENSOR_INSERT);
    if (op == TENSOR_INSERT)
        potHartEnergyGrid.setEntriesZero();

    const double *rhoGridLocal = rhoGrid.getLocalData();
    const double *potHartGridLocal = potHartGrid.getLocalData();
    double *potHartEnergyGridLocal = potHartEnergyGrid.getLocalData();

    int localNumEntries = rhoGrid.getLocalNumberEntries();
    for (int i = 0; i < localNumEntries; ++i) {
        potHartEnergyGridLocal[i] += 0.5 * rhoGridLocal[i] * potHartGridLocal[i];
    }
}

double KSDFTEnergyFunctional::computeFermiEnergy(const double *eigenValues,
                                                 int numberEigenValues,
                                                 int numberElectrons,
                                                 double kb,
                                                 double T,
                                                 int maxFermiIter) {

    // compute Fermi Energy
    int occNum = numberElectrons / 2;
    if (numberElectrons % 2 == 1) {
        occNum += 1;
    }

    double FermiEnergy = eigenValues[occNum - 1];

#ifndef NDEBUG
    std::cout << "Initial Fermi Energy: " << FermiEnergy << std::endl;
#endif
    int numIter = 0;
    for (; numIter < maxFermiIter; ++numIter) {
        double func = 0, funcd = 0, sumfi = 0;
        for (int idx = 0; idx < numberEigenValues; ++idx) {
            double arg = (eigenValues[idx] - FermiEnergy) / (kb * T);
            if (arg <= 0.0) {
                double term = 1.0 / (1.0 + std::exp(arg));
                sumfi = sumfi + 2.0 * term;
                funcd = funcd + 2.0 * (std::exp(arg) / (kb * T)) * term * term;
            } else {
                double term = 1.0 / (1.0 + std::exp(-arg));
                sumfi = sumfi + 2.0 * std::exp(-arg) * term;
                funcd = funcd + 2.0 * (std::exp(-arg) / (kb * T)) * term * term;
            }
        }
        func = sumfi - numberElectrons;
        FermiEnergy = FermiEnergy - func / funcd;
#ifndef NDEBUG
        std::cout << "Iter " << numIter << " FuncValue " << std::setprecision(16) << func << std::endl;
#endif
        if (std::abs(func) <= 1.0e-8) {
            std::cout << "Fermi Energy computation is converged." << std::endl;
            break;
        }
    }
    if (numIter == maxFermiIter) {
        std::cout << "Fermi Energy computation has reached the maximum iteration steps." << std::endl;
    }

    return FermiEnergy;
}

double KSDFTEnergyFunctional::computeRepulsiveEnergy(const std::vector<std::vector<double> > &nuclei) {
    double repEnergy = 0.0;
    int numNuclei = nuclei.size();
    for (int I = 0; I < numNuclei; ++I) {
        double atomChargeI = nuclei[I][0];
        for (int J = 0; J < I; ++J) {
            double atomChargeJ = nuclei[J][0];
            double xdiff = nuclei[I][1] - nuclei[J][1];
            double ydiff = nuclei[I][2] - nuclei[J][2];
            double zdiff = nuclei[I][3] - nuclei[J][3];
            double OnebyDistanceIJ = 1.0 / std::sqrt(xdiff * xdiff + ydiff * ydiff + zdiff * zdiff);
            repEnergy += atomChargeI * atomChargeJ * OnebyDistanceIJ;
        }
    }
    return repEnergy;
}

void KSDFTEnergyFunctional::computeEffMinusPSP(const Tensor3DMPI &rhoGrid,
                                               const Tensor3DMPI &potEffIn,
                                               const Tensor3DMPI &potLocIn,
                                               Tensor3DMPI &potEnergyGrid,
                                               TensorOperation op) {
    assert(op == TENSOR_SUBTRACT || op == TENSOR_INSERT);
    if (op == TENSOR_INSERT)
        potEnergyGrid.setEntriesZero();

    const double *rhoGridLocal = rhoGrid.getLocalData();
    const double *potEffGridLocal = potEffIn.getLocalData();
    const double *potLocGridLocal = potLocIn.getLocalData();
    double *potEnergyGridLocal = potEnergyGrid.getLocalData();

    int localNumEntries = rhoGrid.getLocalNumberEntries();
    if (op == TENSOR_INSERT) {
        for (int i = 0; i < localNumEntries; ++i) {
            potEnergyGridLocal[i] = (potEffGridLocal[i] - potLocGridLocal[i]) * rhoGridLocal[i];
        }
    } else {
        for (int i = 0; i < localNumEntries; ++i) {
            potEnergyGridLocal[i] -= (potEffGridLocal[i] - potLocGridLocal[i]) * rhoGridLocal[i];
        }
    }
}

double KSDFTEnergyFunctional::compute3DIntegral(const Tensor3DMPI &fieldGridValue,
                                                const Tensor3DMPI &jacob3DMat,
                                                const Tensor3DMPI &weight3DMat) {
    assert(fieldGridValue.getLocalNumberEntries() == jacob3DMat.getLocalNumberEntries() &&
           fieldGridValue.getLocalNumberEntries() == weight3DMat.getLocalNumberEntries());

    double integral = 0.0;
    int numLocalEntries = fieldGridValue.getLocalNumberEntries();
    const double *fieldGridValueLocal = fieldGridValue.getLocalData();
    const double *jacob3DMatLocal = jacob3DMat.getLocalData();
    const double *weight3DMatLocal = weight3DMat.getLocalData();

    for (int i = 0; i < numLocalEntries; ++i) {
        integral += fieldGridValueLocal[i] * jacob3DMatLocal[i] * weight3DMatLocal[i];
    }
    double recv = 0.0;
    MPI_Allreduce(&integral,
                  &recv,
                  1,
                  MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    return recv;
}

double KSDFTEnergyFunctional::compute3DIntegralTuckerCuboid(const Tensor3DMPI &fieldGridValue,
                                                            const int rankX,
                                                            const int rankY,
                                                            const int rankZ,
                                                            const FEM &femX,
                                                            const FEM &femY,
                                                            const FEM &femZ) {
    Tucker::SizeArray rank(3);
    rank[0] = rankX;
    rank[1] = rankY;
    rank[2] = rankZ;
    const TuckerMPI::TuckerTensor *decompFieldGrid = TuckerMPI::STHOSVD(fieldGridValue.getTensor(),
                                                                        &rank,
                                                                        true,
                                                                        false);

    TuckerMPI::Tensor *core = decompFieldGrid->G;
    int localRankXStart, localRankXEnd, localRankYStart, localRankYEnd, localRankZStart, localRankZEnd;
    const TuckerMPI::Distribution *distribution = core->getDistribution();
    bool squeezed = false;
    const Tucker::SizeArray *offsetX = distribution->getMap(0,
                                                            squeezed)->getOffsets();
    const Tucker::SizeArray *offsetY = distribution->getMap(1,
                                                            squeezed)->getOffsets();
    const Tucker::SizeArray *offsetZ = distribution->getMap(2,
                                                            squeezed)->getOffsets();
    int procGrid[3];
    distribution->getProcessorGrid()->getCoordinates(procGrid);
    localRankXStart = (*offsetX)[procGrid[0]];
    localRankXEnd = (*offsetX)[procGrid[0] + 1];
    localRankYStart = (*offsetY)[procGrid[1]];
    localRankYEnd = (*offsetY)[procGrid[1] + 1];
    localRankZStart = (*offsetZ)[procGrid[2]];
    localRankZEnd = (*offsetZ)[procGrid[2] + 1];

    const std::vector<double> &weightQuadPointValuesX = femX.getWeightQuadPointValues();
    const std::vector<double> &weightQuadPointValuesY = femY.getWeightQuadPointValues();
    const std::vector<double> &weightQuadPointValuesZ = femZ.getWeightQuadPointValues();
    const std::vector<double> &jacobianQuadPointValuesX = femX.getJacobQuadPointValues();
    const std::vector<double> &jacobianQuadPointValuesY = femY.getJacobQuadPointValues();
    const std::vector<double> &jacobianQuadPointValuesZ = femZ.getJacobQuadPointValues();

    std::vector<double> integx(localRankXEnd - localRankXStart,
                               0.0);
    assert(decompFieldGrid->U[0]->nrows() == femX.getTotalNumberQuadPoints());
    int cnt = 0;
    for (int irank = localRankXStart; irank < localRankXEnd; ++irank) {
        double *uData = decompFieldGrid->U[0]->data() + irank * femX.getTotalNumberQuadPoints();
        for (int i = 0; i < femX.getTotalNumberQuadPoints(); ++i) {
            integx[cnt] += uData[i] * weightQuadPointValuesX[i] * jacobianQuadPointValuesX[i];
        }
        cnt++;
    }

    std::vector<double> integy(localRankYEnd - localRankYStart,
                               0.0);
    assert(decompFieldGrid->U[1]->nrows() == femY.getTotalNumberQuadPoints());
    cnt = 0;
    for (int irank = localRankYStart; irank < localRankYEnd; ++irank) {
        double *vData = decompFieldGrid->U[1]->data() + irank * femY.getTotalNumberQuadPoints();
        for (int i = 0; i < femY.getTotalNumberQuadPoints(); ++i) {
            integy[cnt] += vData[i] * weightQuadPointValuesY[i] * jacobianQuadPointValuesY[i];
        }
        cnt++;
    }

    std::vector<double> integz(localRankZEnd - localRankZStart,
                               0.0);
    assert(decompFieldGrid->U[2]->nrows() == femZ.getTotalNumberQuadPoints());
    cnt = 0;
    for (int irank = localRankZStart; irank < localRankZEnd; ++irank) {
        double *wData = decompFieldGrid->U[2]->data() + irank * femZ.getTotalNumberQuadPoints();
        for (int i = 0; i < femZ.getTotalNumberQuadPoints(); ++i) {
            integz[cnt] += wData[i] * weightQuadPointValuesZ[i] * jacobianQuadPointValuesZ[i];
        }
        cnt++;
    }

    cnt = 0;
    double val = 0.0;
    if (core->getLocalNumEntries() != 0) {
        double *coreData = core->getLocalTensor()->data();
        for (int k = 0; k < integz.size(); ++k) {
            for (int j = 0; j < integy.size(); ++j) {
                for (int i = 0; i < integx.size(); ++i) {
                    val += coreData[cnt] * integx[i] * integy[j] * integz[k];
                    cnt++;
                }
            }
        }
    }

    MPI_Allreduce(MPI_IN_PLACE,
                  &val,
                  1,
                  MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD);


    Tucker::MemoryManager::safe_delete(decompFieldGrid);
    return val;
}

double KSDFTEnergyFunctional::compute3DIntegralTuckerCuboidNodal(const Tensor3DMPI &fieldNodalValue,
                                                                 const int rankX,
                                                                 const int rankY,
                                                                 const int rankZ,
                                                                 const FEM &femX,
                                                                 const FEM &femY,
                                                                 const FEM &femZ) {
    int taskId;
    MPI_Comm_rank(MPI_COMM_WORLD,
                  &taskId);

    Tucker::SizeArray rank(3);
    rank[0] = rankX;
    rank[1] = rankY;
    rank[2] = rankZ;
    const TuckerMPI::TuckerTensor *decompFieldGrid = TuckerMPI::STHOSVD(fieldNodalValue.getTensor(),
                                                                        &rank,
                                                                        true,
                                                                        true);

    TuckerMPI::Tensor *core = decompFieldGrid->G;
    int localRankXStart, localRankXEnd, localRankYStart, localRankYEnd, localRankZStart, localRankZEnd;
    const TuckerMPI::Distribution *distribution = core->getDistribution();
    bool squeezed = false;
    const Tucker::SizeArray *offsetX = distribution->getMap(0,
                                                            squeezed)->getOffsets();
    const Tucker::SizeArray *offsetY = distribution->getMap(1,
                                                            squeezed)->getOffsets();
    const Tucker::SizeArray *offsetZ = distribution->getMap(2,
                                                            squeezed)->getOffsets();
    int procGrid[3];
    distribution->getProcessorGrid()->getCoordinates(procGrid);
    localRankXStart = (*offsetX)[procGrid[0]];
    localRankXEnd = (*offsetX)[procGrid[0] + 1];
    localRankYStart = (*offsetY)[procGrid[1]];
    localRankYEnd = (*offsetY)[procGrid[1] + 1];
    localRankZStart = (*offsetZ)[procGrid[2]];
    localRankZEnd = (*offsetZ)[procGrid[2] + 1];

    int nrows = femX.getTotalNumberQuadPoints(), ncols = decompFieldGrid->U[0]->ncols();
    Tucker::Matrix *u = Tucker::MemoryManager::safe_new<Tucker::Matrix>(nrows,
                                                                        ncols);
    for (int j = 0; j < ncols; ++j) {
        std::vector<double> tempNodal(femX.getTotalNumberNodes());
        std::vector<double> tempQuad(femX.getTotalNumberQuadPoints());
        std::vector<double> tempQuadDiff(femX.getTotalNumberQuadPoints());
        std::copy(decompFieldGrid->U[0]->data() + j * femX.getTotalNumberNodes(),
                  decompFieldGrid->U[0]->data() + (j + 1) * femX.getTotalNumberNodes(),
                  tempNodal.begin());
        femX.computeFieldAndDiffFieldAtAllQuadPoints(tempNodal,
                                                     tempQuad,
                                                     tempQuadDiff);
        std::copy(tempQuad.begin(),
                  tempQuad.end(),
                  u->data() + j * nrows);
    }

    nrows = femY.getTotalNumberQuadPoints(), ncols = decompFieldGrid->U[1]->ncols();
    Tucker::Matrix *v = Tucker::MemoryManager::safe_new<Tucker::Matrix>(nrows,
                                                                        ncols);
    for (int j = 0; j < ncols; ++j) {
        std::vector<double> tempNodal(femY.getTotalNumberNodes());
        std::vector<double> tempQuad(femY.getTotalNumberQuadPoints());
        std::vector<double> tempQuadDiff(femY.getTotalNumberQuadPoints());
        std::copy(decompFieldGrid->U[1]->data() + j * femY.getTotalNumberNodes(),
                  decompFieldGrid->U[1]->data() + (j + 1) * femY.getTotalNumberNodes(),
                  tempNodal.begin());
        femY.computeFieldAndDiffFieldAtAllQuadPoints(tempNodal,
                                                     tempQuad,
                                                     tempQuadDiff);
        std::copy(tempQuad.begin(),
                  tempQuad.end(),
                  v->data() + j * nrows);
    }

    nrows = femZ.getTotalNumberQuadPoints(), ncols = decompFieldGrid->U[2]->ncols();
    Tucker::Matrix *w = Tucker::MemoryManager::safe_new<Tucker::Matrix>(nrows,
                                                                        ncols);
    for (int j = 0; j < ncols; ++j) {
        std::vector<double> tempNodal(femZ.getTotalNumberNodes());
        std::vector<double> tempQuad(femZ.getTotalNumberQuadPoints());
        std::vector<double> tempQuadDiff(femZ.getTotalNumberQuadPoints());
        std::copy(decompFieldGrid->U[2]->data() + j * femZ.getTotalNumberNodes(),
                  decompFieldGrid->U[2]->data() + (j + 1) * femZ.getTotalNumberNodes(),
                  tempNodal.begin());
        femY.computeFieldAndDiffFieldAtAllQuadPoints(tempNodal,
                                                     tempQuad,
                                                     tempQuadDiff);
        std::copy(tempQuad.begin(),
                  tempQuad.end(),
                  w->data() + j * nrows);
    }

    const std::vector<double> &weightQuadPointValuesX = femX.getWeightQuadPointValues();
    const std::vector<double> &weightQuadPointValuesY = femY.getWeightQuadPointValues();
    const std::vector<double> &weightQuadPointValuesZ = femZ.getWeightQuadPointValues();
    const std::vector<double> &jacobianQuadPointValuesX = femX.getJacobQuadPointValues();
    const std::vector<double> &jacobianQuadPointValuesY = femY.getJacobQuadPointValues();
    const std::vector<double> &jacobianQuadPointValuesZ = femZ.getJacobQuadPointValues();

    std::vector<double> integx(localRankXEnd - localRankXStart,
                               0.0);
    int cnt = 0;
    for (int irank = localRankXStart; irank < localRankXEnd; ++irank) {
        double *uData = u->data() + irank * femX.getTotalNumberQuadPoints();
        for (int i = 0; i < femX.getTotalNumberQuadPoints(); ++i) {
            integx[cnt] += uData[i] * weightQuadPointValuesX[i] * jacobianQuadPointValuesX[i];
        }
        cnt++;
    }

    std::vector<double> integy(localRankYEnd - localRankYStart,
                               0.0);
    cnt = 0;
    for (int irank = localRankYStart; irank < localRankYEnd; ++irank) {
        double *vData = v->data() + irank * femY.getTotalNumberQuadPoints();
        for (int i = 0; i < femY.getTotalNumberQuadPoints(); ++i) {
            integy[cnt] += vData[i] * weightQuadPointValuesY[i] * jacobianQuadPointValuesY[i];
        }
        cnt++;
    }

    std::vector<double> integz(localRankZEnd - localRankZStart,
                               0.0);
    cnt = 0;
    for (int irank = localRankZStart; irank < localRankZEnd; ++irank) {
        double *wData = w->data() + irank * femZ.getTotalNumberQuadPoints();
        for (int i = 0; i < femZ.getTotalNumberQuadPoints(); ++i) {
            integz[cnt] += wData[i] * weightQuadPointValuesZ[i] * jacobianQuadPointValuesZ[i];
        }
        cnt++;
    }
    cnt = 0;
    double val = 0.0;
    if (core->getLocalNumEntries() != 0) {
        double *coreData = core->getLocalTensor()->data();
        for (int k = 0; k < integz.size(); ++k) {
            for (int j = 0; j < integy.size(); ++j) {
                for (int i = 0; i < integx.size(); ++i) {
                    val += coreData[cnt] * integx[i] * integy[j] * integz[k];
                    cnt++;
                }
            }
        }
    }

    MPI_Allreduce(MPI_IN_PLACE,
                  &val,
                  1,
                  MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD);

    Tucker::MemoryManager::safe_delete(decompFieldGrid);
    Tucker::MemoryManager::safe_delete(u);
    Tucker::MemoryManager::safe_delete(v);
    Tucker::MemoryManager::safe_delete(w);
    return val;
}


