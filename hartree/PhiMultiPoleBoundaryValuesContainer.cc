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

#include "PhiMultiPoleBoundaryValuesContainer.h"
#include "../tensor/TensorUtils.h"

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

PhiMultiPoleBoundaryValuesContainer::PhiMultiPoleBoundaryValuesContainer(const std::array<int, 6> &owned_index,
                                                                         const FEM &fem_x,
                                                                         const FEM &fem_y,
                                                                         const FEM &fem_z,
                                                                         const FEM &fem_electro_x,
                                                                         const FEM &fem_electro_y,
                                                                         const FEM &fem_electro_z,
                                                                         std::string omega_file,
                                                                         std::string alpha_file,
                                                                         double Asquare,
                                                                         const unsigned rho_decomposed_rank_x,
                                                                         const unsigned rho_decomposed_rank_y,
                                                                         const unsigned rho_decomposed_rank_z) :
        PhiBoundaryValuesContainer(owned_index,
                                   fem_x,
                                   fem_y,
                                   fem_z),
        fem_electro_x_(fem_electro_x),
        fem_electro_y_(fem_electro_y),
        fem_electro_z_(fem_electro_z),
        rho_decomposed_rank_x_(rho_decomposed_rank_x),
        rho_decomposed_rank_y_(rho_decomposed_rank_y),
        rho_decomposed_rank_z_(rho_decomposed_rank_z),
        Asquare_(Asquare) {
    auto readFile = [](std::string filename,
                       std::vector<double> &data) {
        std::fstream fin;
        fin.open(filename.c_str());
        double readintemp;
        data.clear();
        while (fin >> readintemp) {
            data.push_back(readintemp);
        }
        data.shrink_to_fit();
        fin.close();
    };

    readFile(alpha_file,
             alpha_);
    readFile(omega_file,
             omega_);
    int numberExpansion = alpha_.size();
    for (auto i = 0; i != numberExpansion; ++i) {
        omega_[i] *= (1.0 / std::sqrt(Asquare));
        alpha_[i] *= (1.0 / Asquare);
    }

    ComputeBoundaryIndices();
    PhiMultiPoleBoundaryValuesContainer::rho_ = nullptr;
}

void PhiMultiPoleBoundaryValuesContainer::SetRho(Tensor3DMPI *rho) {
    PhiMultiPoleBoundaryValuesContainer::rho_ = rho;
}

void PhiMultiPoleBoundaryValuesContainer::ComputeBoundaryValues() {
    const TuckerMPI::TuckerTensor *decomposedRhoGrid = TensorUtils::computeSTHOSVDonQuadMPI(fem_electro_x_,
                                                                                            fem_electro_y_,
                                                                                            fem_electro_z_,
                                                                                            rho_decomposed_rank_x_,
                                                                                            rho_decomposed_rank_y_,
                                                                                            rho_decomposed_rank_z_,
                                                                                            *rho_);
    const std::vector<double> &nodes_x = fem_x_.getGlobalNodalCoord();
    const std::vector<double> &nodes_y = fem_y_.getGlobalNodalCoord();
    const std::vector<double> &nodes_z = fem_z_.getGlobalNodalCoord();

    Tucker::Tensor *seqcore = Tucker::MemoryManager::safe_new<Tucker::Tensor>(decomposedRhoGrid->G->getGlobalSize());
    TensorUtils::allgatherTensor(decomposedRhoGrid->G,
                                 seqcore);

    unsigned numberExpansion = alpha_.size();
    std::vector<Tucker::Matrix *> localFuncX(numberExpansion);
    std::vector<Tucker::Matrix *> localFuncY(numberExpansion);
    std::vector<Tucker::Matrix *> localFuncZ(numberExpansion);

    const std::vector<double> &positionQuadPointValuesElectroX = fem_electro_x_.getPositionQuadPointValues();
    const std::vector<double> &positionQuadPointValuesElectroY = fem_electro_y_.getPositionQuadPointValues();
    const std::vector<double> &positionQuadPointValuesElectroZ = fem_electro_z_.getPositionQuadPointValues();

    const Tucker::Matrix *UX = decomposedRhoGrid->U[0];
    const Tucker::Matrix *UY = decomposedRhoGrid->U[1];
    const Tucker::Matrix *UZ = decomposedRhoGrid->U[2];

    Tensor3DMPI temp_hartree_node(fem_x_.getTotalNumberNodes(),
                                  fem_y_.getTotalNumberNodes(),
                                  fem_z_.getTotalNumberNodes(),
                                  MPI_COMM_WORLD);
    temp_hartree_node.setEntriesZero();
    std::array<int, 6> temp_hartree_node_idx = temp_hartree_node.getGlobalIndex();

    int nrowsMatX = temp_hartree_node_idx[1] - temp_hartree_node_idx[0], ncolsMatX = rho_decomposed_rank_x_;
    for (int i = 0; i < numberExpansion; ++i) {
        localFuncX[i] = Tucker::MemoryManager::safe_new<Tucker::Matrix>(nrowsMatX,
                                                                        ncolsMatX);
    }
    int numberFemElectroXQuadPoints = positionQuadPointValuesElectroX.size();
    for (auto iterm = 0; iterm != numberExpansion; ++iterm) {
        int cnt = 0;
        double *integralFuncXData = localFuncX[iterm]->data();
        for (auto irank = 0; irank < rho_decomposed_rank_x_; ++irank) {
            const double *rhoEig = UX->data() + irank * numberFemElectroXQuadPoints;
            for (auto inode = temp_hartree_node_idx[0]; inode < temp_hartree_node_idx[1]; ++inode) {
                std::vector<double> temp(numberFemElectroXQuadPoints,
                                         nodes_x[inode]);
                for (auto irho = 0; irho < numberFemElectroXQuadPoints; ++irho) {
                    temp[irho] -= positionQuadPointValuesElectroX[irho];
                    temp[irho] = std::exp(-alpha_[iterm] * temp[irho] * temp[irho]) * rhoEig[irho];
                }
                integralFuncXData[cnt++] = fem_electro_x_.integrate_by_quad_values(temp);
            }
        }
    }

    int nrowsMatY = temp_hartree_node_idx[3] - temp_hartree_node_idx[2], ncolsMatY = rho_decomposed_rank_y_;
    for (int i = 0; i < numberExpansion; ++i) {
        localFuncY[i] = Tucker::MemoryManager::safe_new<Tucker::Matrix>(nrowsMatY,
                                                                        ncolsMatY);
    }
    int numberFemElectroYQuadPoints = positionQuadPointValuesElectroY.size();
    for (auto iterm = 0; iterm != numberExpansion; ++iterm) {
        int cnt = 0;
        double *integralFuncYData = localFuncY[iterm]->data();
        for (auto irank = 0; irank < rho_decomposed_rank_y_; ++irank) {
            const double *rhoEig = UY->data() + irank * numberFemElectroYQuadPoints;
            for (auto inode = temp_hartree_node_idx[2]; inode < temp_hartree_node_idx[3]; ++inode) {
                std::vector<double> temp(numberFemElectroYQuadPoints,
                                         nodes_y[inode]);
                for (auto irho = 0; irho < numberFemElectroYQuadPoints; ++irho) {
                    temp[irho] -= positionQuadPointValuesElectroY[irho];
                    temp[irho] = std::exp(-alpha_[iterm] * temp[irho] * temp[irho]) * rhoEig[irho];
                }
                integralFuncYData[cnt++] = fem_electro_y_.integrate_by_quad_values(temp);
            }
        }
    }

    int nrowsMatZ = temp_hartree_node_idx[5] - temp_hartree_node_idx[4], ncolsMatZ = rho_decomposed_rank_z_;
    for (int i = 0; i < numberExpansion; ++i) {
        localFuncZ[i] = Tucker::MemoryManager::safe_new<Tucker::Matrix>(nrowsMatZ,
                                                                        ncolsMatZ);
    }
    int numberFemElectroZQuadPoints = positionQuadPointValuesElectroZ.size();
    for (auto iterm = 0; iterm != numberExpansion; ++iterm) {
        int cnt = 0;
        double *integralFuncZData = localFuncZ[iterm]->data();
        for (auto irank = 0; irank < rho_decomposed_rank_z_; ++irank) {
            const double *rhoEig = UZ->data() + irank * numberFemElectroZQuadPoints;
            for (auto inode = temp_hartree_node_idx[4]; inode < temp_hartree_node_idx[5]; ++inode) {
                std::vector<double> temp(numberFemElectroZQuadPoints,
                                         nodes_z[inode]);
                for (auto irho = 0; irho < numberFemElectroZQuadPoints; ++irho) {
                    temp[irho] -= positionQuadPointValuesElectroZ[irho];
                    temp[irho] = std::exp(-alpha_[iterm] * temp[irho] * temp[irho]) * rhoEig[irho];
                }
                integralFuncZData[cnt++] = fem_electro_z_.integrate_by_quad_values(temp);
            }
        }
    }


    // could be used to initialize hartree potential
    for (auto iterm = 0; iterm != numberExpansion; ++iterm) {
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

        double *temp_hartree_node_data = temp_hartree_node.getLocalData();
        double *reconstructedTensorData = reconstructedTensor->data();
        double omegai = omega_[iterm];
        int incx = 1;
        int incy = 1;
        int localNumEntries = temp_hartree_node.getLocalNumberEntries();
        assert(reconstructedTensor->getNumElements() == localNumEntries);
        daxpy_(&localNumEntries,
               &omegai,
               reconstructedTensorData,
               &incx,
               temp_hartree_node_data,
               &incy);

        Tucker::MemoryManager::safe_delete(reconstructedTensor);
    }
    Tucker::MemoryManager::safe_delete(seqcore);
    for (int i = 0; i < numberExpansion; ++i) {
        Tucker::MemoryManager::safe_delete(localFuncX[i]);
        Tucker::MemoryManager::safe_delete(localFuncY[i]);
        Tucker::MemoryManager::safe_delete(localFuncZ[i]);
    }

    double *temp_hartree_node_data = temp_hartree_node.getLocalData();
    local_boundary_values_ = std::vector<double>(boundary_values_local_index_.size(),
                                                 0.0);
    for (int i = 0; i < boundary_values_local_index_.size(); ++i) {
        local_boundary_values_[i] = temp_hartree_node_data[boundary_values_local_index_[i]];
    }

}