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

#include <mpi.h>
#include <iomanip>
#include "PoissonHartreePotentialSolver.h"
#include "PhiZoverRBoundaryValuesContainer.h"
#include "../tensor/TuckerTensor.h"
#include "../tensor/TensorUtils.h"
#include "HartreeCalculator.h"
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
}

PoissonHartreePotentialSolver::PoissonHartreePotentialSolver(const FEM &fem_inner_x,
                                                             const FEM &fem_inner_y,
                                                             const FEM &fem_inner_z,
                                                             const FEM &fem_inner_electro_x,
                                                             const FEM &fem_inner_electro_y,
                                                             const FEM &fem_inner_electro_z,
                                                             PETScLinearSolver::Solver solver_type,
                                                             PETScLinearSolver::Preconditioner preconditioner_type,
                                                             const std::vector<std::vector<double>> &nuclei,
                                                             const int rho_decompose_rank_x,
                                                             const int rho_decompose_rank_y,
                                                             const int rho_decompose_rank_z,
                                                             const std::string &ig_alpha_filename,
                                                             const std::string &ig_omega_filename,
                                                             const double Asquare,
                                                             const double fem_outer_domain_x_start,
                                                             const double fem_outer_domain_x_end,
                                                             const double fem_outer_domain_y_start,
                                                             const double fem_outer_domain_y_end,
                                                             const double fem_outer_domain_z_start,
                                                             const double fem_outer_domain_z_end,
                                                             const double coarsing_ratio,
                                                             const int num_additional_elements)
        : fem_inner_electro_x(fem_inner_electro_x),
          fem_inner_electro_y(fem_inner_electro_y),
          fem_inner_electro_z(fem_inner_electro_z),
          nuclei(nuclei),
          Asquare(Asquare),
          solver_type(solver_type),
          preconditioner_type(preconditioner_type),
          rho_decompose_rank_x(rho_decompose_rank_x),
          rho_decompose_rank_y(rho_decompose_rank_y),
          rho_decompose_rank_z(rho_decompose_rank_z),
          fem_inner_x(fem_inner_x),
          fem_inner_y(fem_inner_y),
          fem_inner_z(fem_inner_z),
          is_initialize_hartree(true) {

    Utils::ReadSingleColumnFile(ig_alpha_filename,
                                alpha);
    Utils::ReadSingleColumnFile(ig_omega_filename,
                                omega);
    int numberExpansion = alpha.size();
    for (auto i = 0; i != numberExpansion; ++i) {
        omega[i] *= (1.0 / std::sqrt(Asquare));
        alpha[i] *= (1.0 / Asquare);
    }

    fem_hartree_x = initialize_poisson_fem(fem_inner_x,
                                           fem_outer_domain_x_start,
                                           fem_outer_domain_x_end,
                                           num_additional_elements,
                                           coarsing_ratio);
    fem_hartree_y = initialize_poisson_fem(fem_inner_y,
                                           fem_outer_domain_y_start,
                                           fem_outer_domain_y_end,
                                           num_additional_elements,
                                           coarsing_ratio);
    fem_hartree_z = initialize_poisson_fem(fem_inner_z,
                                           fem_outer_domain_z_start,
                                           fem_outer_domain_z_end,
                                           num_additional_elements,
                                           coarsing_ratio);

    hartree_nodal = std::shared_ptr<Tensor3DMPI>(new Tensor3DMPI(fem_hartree_x->getTotalNumberNodes(),
                                                                 fem_hartree_y->getTotalNumberNodes(),
                                                                 fem_hartree_z->getTotalNumberNodes(),
                                                                 MPI_COMM_WORLD));
    hartreeNodalMap = std::shared_ptr<Tensor3DMPIMap>(new Tensor3DMPIMap(*fem_hartree_x,
                                                                         *fem_hartree_y,
                                                                         *fem_hartree_z,
                                                                         *hartree_nodal));
    phi_boundary_values =
            std::shared_ptr<PhiZoverRBoundaryValuesContainer>(new PhiZoverRBoundaryValuesContainer(hartree_nodal->getGlobalIndex(),
                                                                                                   nuclei,
                                                                                                   *fem_hartree_x,
                                                                                                   *fem_hartree_y,
                                                                                                   *fem_hartree_z));
    hartree_nodal->setEntriesZero();
}

std::shared_ptr<FEM> PoissonHartreePotentialSolver::initialize_poisson_fem(const FEM &fem,
                                                                           const double poisson_outer_domain_start,
                                                                           const double poisson_outer_domain_end,
                                                                           int num_additional_elements,
                                                                           double coarsing_ratio) {

    std::shared_ptr<FEM> fem_ptr = std::shared_ptr<FEM>(new FEM(fem,
                                                                num_additional_elements,
                                                                coarsing_ratio,
                                                                poisson_outer_domain_start,
                                                                poisson_outer_domain_end));

    int taskId;
    MPI_Comm_rank(MPI_COMM_WORLD,
                  &taskId);
    if (taskId == 0) {
        const std::vector<double> &hartree_fem_node = fem_ptr->getGlobalNodalCoord();
        std::cout << std::setprecision(5);
        std::cout << "hartree nodes: (";
        for (int i = 0; i < hartree_fem_node.size() - 1; ++i) {
            std::cout << hartree_fem_node[i] << ", ";
        }
        std::cout << hartree_fem_node.back() << ")" << std::endl;
    }

    return fem_ptr;
}

void PoissonHartreePotentialSolver::compute_kernel_expansion_values(const FEM &fem_electro_x,
                                                                    const FEM &fem_electro_y,
                                                                    const FEM &fem_electro_z,
                                                                    const FEM &fem_hartree_x,
                                                                    const FEM &fem_hartree_y,
                                                                    const FEM &fem_hartree_z,
                                                                    const std::vector<double> &alpha,
                                                                    const std::vector<double> &omega,
                                                                    const double &Asquare,
                                                                    const int rho_decompose_rank_x,
                                                                    const int rho_decompose_rank_y,
                                                                    const int rho_decompose_rank_z,
                                                                    Tensor3DMPI &rho,
                                                                    Tensor3DMPI &hartree) {
    const TuckerMPI::TuckerTensor *decomposedRhoGrid = TensorUtils::computeSTHOSVDonQuadMPI(fem_electro_x,
                                                                                            fem_electro_y,
                                                                                            fem_electro_z,
                                                                                            rho_decompose_rank_x,
                                                                                            rho_decompose_rank_y,
                                                                                            rho_decompose_rank_z,
                                                                                            rho);
    const std::vector<double> &nodes_x = fem_hartree_x.getGlobalNodalCoord();
    const std::vector<double> &nodes_y = fem_hartree_y.getGlobalNodalCoord();
    const std::vector<double> &nodes_z = fem_hartree_z.getGlobalNodalCoord();

    Tucker::Tensor *seqcore = Tucker::MemoryManager::safe_new<Tucker::Tensor>(decomposedRhoGrid->G->getGlobalSize());
    TensorUtils::allgatherTensor(decomposedRhoGrid->G,
                                 seqcore);

    if (hartree.getLocalNumberEntries() != 0) {
        unsigned numberExpansion = alpha.size();
        std::vector<Tucker::Matrix *> localFuncX(numberExpansion);
        std::vector<Tucker::Matrix *> localFuncY(numberExpansion);
        std::vector<Tucker::Matrix *> localFuncZ(numberExpansion);

        const std::vector<double> &positionQuadPointValuesElectroX = fem_electro_x.getPositionQuadPointValues();
        const std::vector<double> &positionQuadPointValuesElectroY = fem_electro_y.getPositionQuadPointValues();
        const std::vector<double> &positionQuadPointValuesElectroZ = fem_electro_z.getPositionQuadPointValues();

        const Tucker::Matrix *UX = decomposedRhoGrid->U[0];
        const Tucker::Matrix *UY = decomposedRhoGrid->U[1];
        const Tucker::Matrix *UZ = decomposedRhoGrid->U[2];

        std::array<int, 6> hartree_idx = hartree.getGlobalIndex();

        int nrowsMatX = hartree_idx[1] - hartree_idx[0], ncolsMatX = rho_decompose_rank_x;
        for (int i = 0; i < numberExpansion; ++i) {
            localFuncX[i] = Tucker::MemoryManager::safe_new<Tucker::Matrix>(nrowsMatX,
                                                                            ncolsMatX);
        }
        int numberFemElectroXQuadPoints = positionQuadPointValuesElectroX.size();
        for (auto iterm = 0; iterm != numberExpansion; ++iterm) {
            int cnt = 0;
            if (localFuncX[iterm]->getNumElements() != 0) {
                double *integralFuncXData = localFuncX[iterm]->data();
                for (auto irank = 0; irank < rho_decompose_rank_x; ++irank) {
                    if (UX->getNumElements() != 0) {
                        const double *rhoEig = UX->data() + irank * numberFemElectroXQuadPoints;
                        for (auto inode = hartree_idx[0]; inode < hartree_idx[1]; ++inode) {
                            std::vector<double> temp(numberFemElectroXQuadPoints,
                                                     nodes_x[inode]);
                            for (auto irho = 0; irho < numberFemElectroXQuadPoints; ++irho) {
                                temp[irho] -= positionQuadPointValuesElectroX[irho];
                                temp[irho] = std::exp(-alpha[iterm] * temp[irho] * temp[irho]) * rhoEig[irho];
                            }
                            integralFuncXData[cnt++] = fem_electro_x.integrate_by_quad_values(temp);
                        }
                    }
                }
            }
        }

        int nrowsMatY = hartree_idx[3] - hartree_idx[2], ncolsMatY = rho_decompose_rank_y;
        for (int i = 0; i < numberExpansion; ++i) {
            localFuncY[i] = Tucker::MemoryManager::safe_new<Tucker::Matrix>(nrowsMatY,
                                                                            ncolsMatY);
        }
        int numberFemElectroYQuadPoints = positionQuadPointValuesElectroY.size();
        for (auto iterm = 0; iterm != numberExpansion; ++iterm) {
            int cnt = 0;
            if (localFuncY[iterm]->getNumElements() != 0) {
                double *integralFuncYData = localFuncY[iterm]->data();
                for (auto irank = 0; irank < rho_decompose_rank_x; ++irank) {
                    if (UY->getNumElements() != 0) {
                        const double *rhoEig = UY->data() + irank * numberFemElectroYQuadPoints;
                        for (auto inode = hartree_idx[2]; inode < hartree_idx[3]; ++inode) {
                            std::vector<double> temp(numberFemElectroYQuadPoints,
                                                     nodes_y[inode]);
                            for (auto irho = 0; irho < numberFemElectroYQuadPoints; ++irho) {
                                temp[irho] -= positionQuadPointValuesElectroY[irho];
                                temp[irho] = std::exp(-alpha[iterm] * temp[irho] * temp[irho]) * rhoEig[irho];
                            }
                            integralFuncYData[cnt++] = fem_electro_y.integrate_by_quad_values(temp);
                        }
                    }
                }
            }
        }

        int nrowsMatZ = hartree_idx[5] - hartree_idx[4], ncolsMatZ = rho_decompose_rank_z;
        for (int i = 0; i < numberExpansion; ++i) {
            localFuncZ[i] = Tucker::MemoryManager::safe_new<Tucker::Matrix>(nrowsMatZ,
                                                                            ncolsMatZ);
        }
        int numberFemElectroZQuadPoints = positionQuadPointValuesElectroZ.size();
        for (auto iterm = 0; iterm != numberExpansion; ++iterm) {
            int cnt = 0;

            if (localFuncZ[iterm]->getNumElements() != 0) {
                double *integralFuncZData = localFuncZ[iterm]->data();
                for (auto irank = 0; irank < rho_decompose_rank_z; ++irank) {
                    if (UZ->getNumElements() != 0) {
                        const double *rhoEig = UZ->data() + irank * numberFemElectroZQuadPoints;
                        for (auto inode = hartree_idx[4]; inode < hartree_idx[5]; ++inode) {
                            std::vector<double> temp(numberFemElectroZQuadPoints,
                                                     nodes_z[inode]);
                            for (auto irho = 0; irho < numberFemElectroZQuadPoints; ++irho) {
                                temp[irho] -= positionQuadPointValuesElectroZ[irho];
                                temp[irho] = std::exp(-alpha[iterm] * temp[irho] * temp[irho]) * rhoEig[irho];
                            }
                            integralFuncZData[cnt++] = fem_electro_z.integrate_by_quad_values(temp);
                        }
                    }
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

            if (hartree.getLocalNumberEntries() != 0) {
                double *hatree_data = hartree.getLocalData();
                double *reconstructedTensorData = reconstructedTensor->data();
                double omegai = omega[iterm];
                int incx = 1;
                int incy = 1;
                int localNumEntries = hartree.getLocalNumberEntries();
                assert(reconstructedTensor->getNumElements() == localNumEntries);
                daxpy_(&localNumEntries,
                       &omegai,
                       reconstructedTensorData,
                       &incx,
                       hatree_data,
                       &incy);
            }
            Tucker::MemoryManager::safe_delete(reconstructedTensor);
        }
        int num_local_entries = hartree.getLocalNumberEntries();
        bool is_all_values_finite = true;
        if (num_local_entries != 0) {
            double *hartree_data = hartree.getLocalData();
            for (int i = 0; i < num_local_entries; ++i) {
                if (std::isfinite(hartree_data[i]) == false) is_all_values_finite = false;
            }
        }
        if (is_all_values_finite == false) {
            std::cout << "Non finite values in kerenel expansion part of constructing ig." << std::endl;
            std::terminate();
        }
        Tucker::MemoryManager::safe_delete(seqcore);
        for (int i = 0; i < numberExpansion; ++i) {
            Tucker::MemoryManager::safe_delete(localFuncX[i]);
            Tucker::MemoryManager::safe_delete(localFuncY[i]);
            Tucker::MemoryManager::safe_delete(localFuncZ[i]);
        }
    }
}

bool PoissonHartreePotentialSolver::solve(const int maxIter,
                                          const double tolerance,
                                          Tensor3DMPI &rho_node,
                                          Tensor3DMPI &hartree_quad) {
    phi_boundary_values->SetRho(&rho_node);
    phi_boundary_values->ComputeBoundaryValues();

    // initialize hartree potential using kernel expansion for poisson's solver
    if (is_initialize_hartree == true) {
        int taskId;
        MPI_Comm_rank(MPI_COMM_WORLD,
                      &taskId);
        if (taskId == 0) {
            std::cout << "hartree initialized" << std::endl;
        }
        compute_kernel_expansion_values(fem_inner_electro_x,
                                        fem_inner_electro_y,
                                        fem_inner_electro_z,
                                        *fem_hartree_x,
                                        *fem_hartree_y,
                                        *fem_hartree_z,
                                        alpha,
                                        omega,
                                        Asquare,
                                        rho_decompose_rank_x,
                                        rho_decompose_rank_y,
                                        rho_decompose_rank_z,
                                        rho_node,
                                        *hartree_nodal);
    }
    HartreeCalculator::CalculateHartreePotentialOnQuadPointsUsingLargerDomain(fem_inner_x,
                                                                              fem_inner_y,
                                                                              fem_inner_z,
                                                                              *fem_hartree_x,
                                                                              *fem_hartree_y,
                                                                              *fem_hartree_z,
                                                                              maxIter,
                                                                              tolerance,
                                                                              solver_type,
                                                                              preconditioner_type,
                                                                              rho_node,
                                                                              *hartreeNodalMap,
                                                                              *hartree_nodal,
                                                                              phi_boundary_values.get(),
                                                                              hartree_quad);
    is_initialize_hartree = false;
    bool is_all_values_finite = true;
    int hartree_quad_num_local_entries = hartree_quad.getLocalNumberEntries();
    if (hartree_quad_num_local_entries != 0) {
        double *hartree_quad_data = hartree_quad.getLocalData();
        for (int i = 0; i < hartree_quad_num_local_entries; ++i) {
            if (std::isfinite(hartree_quad_data[i]) == false) is_all_values_finite = false;
        }
    }
    if (is_all_values_finite == false) {
        std::cout << "the hartree potential computed on quadrature points has non-finite values." << std::endl;
        std::terminate();
    }
    return true;
}

void PoissonHartreePotentialSolver::turn_on_initialize_hartree() {
    is_initialize_hartree = true;
}
