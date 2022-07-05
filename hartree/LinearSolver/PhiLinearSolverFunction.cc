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

#include <set>
#include "PhiLinearSolverFunction.h"

void PhiLinearSolverFunction::ComputeNonZeroPatterns(const FEM &fem_x,
                                                     const FEM &fem_y,
                                                     const FEM &fem_z,
                                                     const std::array<int, 6> &rho_index,
                                                     std::vector<PetscInt> &diagonal_non_zeros,
                                                     std::vector<PetscInt> &offdiagonal_non_zeros) {
    int numberTotalNodesX = fem_x.getTotalNumberNodes();
    int numberTotalNodesY = fem_y.getTotalNumberNodes();

    int numberNodesPerEleX = fem_x.getNumberNodesPerElement();
    int numberNodesPerEleY = fem_y.getNumberNodesPerElement();
    int numberNodesPerEleZ = fem_z.getNumberNodesPerElement();

    std::set<int> owned_global_nodes;
    for (int k = rho_index[4]; k < rho_index[5]; ++k) {
        for (int j = rho_index[2]; j < rho_index[3]; ++j) {
            for (int i = rho_index[0]; i < rho_index[1]; ++i) {
                int globalI = i + j * numberTotalNodesX + k * numberTotalNodesX * numberTotalNodesY;
                owned_global_nodes.insert(globalI);
            }
        }
    }

    diagonal_non_zeros = std::vector<PetscInt>(owned_global_nodes.size(),
                                               0);
    offdiagonal_non_zeros = std::vector<PetscInt>(owned_global_nodes.size(),
                                                  0);

    const std::multimap<int, int> &node2ElementMapX = fem_x.getNodeToElementMap();
    const std::multimap<int, int> &node2ElementMapY = fem_y.getNodeToElementMap();
    const std::multimap<int, int> &node2ElementMapZ = fem_z.getNodeToElementMap();
    int cnt = 0;
    for (int k = rho_index[4]; k < rho_index[5]; ++k) {
        for (int j = rho_index[2]; j < rho_index[3]; ++j) {
            for (int i = rho_index[0]; i < rho_index[1]; ++i) {
                for (auto iterZ = node2ElementMapZ.lower_bound(k);
                     iterZ != node2ElementMapZ.upper_bound(k); ++iterZ) {
                    for (auto iterY = node2ElementMapY.lower_bound(j);
                         iterY != node2ElementMapY.upper_bound(j); ++iterY) {
                        for (auto iterX = node2ElementMapX.lower_bound(i);
                             iterX != node2ElementMapX.upper_bound(i); ++iterX) {
                            int eleX = iterX->second;
                            int eleY = iterY->second;
                            int eleZ = iterZ->second;
                            for (int r = 0; r < numberNodesPerEleZ; ++r) {
                                for (int q = 0; q < numberNodesPerEleY; ++q) {
                                    for (int p = 0; p < numberNodesPerEleX; ++p) {
                                        int globali = eleX * (numberNodesPerEleX - 1) + p;
                                        int globalj = eleY * (numberNodesPerEleY - 1) + q;
                                        int globalk = eleZ * (numberNodesPerEleZ - 1) + r;
                                        int globalJ = globali + globalj * numberTotalNodesX +
                                                      globalk * numberTotalNodesX * numberTotalNodesY;
                                        if (owned_global_nodes.find(globalJ) == owned_global_nodes.end()) {
                                            offdiagonal_non_zeros[cnt] += 1;
                                        } else {
                                            diagonal_non_zeros[cnt] += 1;
                                        }
                                    }
                                }
                            } // end of nodes inside of element loop
                        }
                    }
                } // end of elements loop
                cnt++;
            }
        }
    } // end of loop over owned rho
}

void PhiLinearSolverFunction::ComputeNINJWithoutBoundaryNodes(Mat &NINJ) {
    const std::vector<Cartesian<unsigned>> &rho_cartisian_map = rho_index_map.GetLocalTensorGlobalCarteisianIndex();
    const std::vector<unsigned> &rho_global_index_map = rho_index_map.GetLocalTensorGlobalIndex();

    PetscInt rho_number_local_entries = rho_global_index_map.size();
    PetscInt total_nodes = number_total_nodes_x * number_total_nodes_y * number_total_nodes_z;
    PetscInt number_nodes_per_element_x = fem_x.getNumberNodesPerElement();
    PetscInt number_nodes_per_element_y = fem_y.getNumberNodesPerElement();
    PetscInt number_nodes_per_element_z = fem_z.getNumberNodesPerElement();
    PetscInt
            total_nodes_per_element =
            number_nodes_per_element_x * number_nodes_per_element_x * number_nodes_per_element_z;

    MatCreate(PETSC_COMM_WORLD,
              &NINJ);
    MatSetType(NINJ,
               MATMPIAIJ);
    MatSetSizes(NINJ,
                rho_number_local_entries,
                rho_number_local_entries,
                total_nodes,
                total_nodes);
//  MatMPIAIJSetPreallocation(NINJ, 8 * total_nodes_per_element, PETSC_NULL, 8 * total_nodes_per_element, PETSC_NULL);
    MatMPIAIJSetPreallocation(NINJ,
                              0,
                              diagonal_non_zeros.data(),
                              0,
                              offdiagonal_non_zeros.data());

    auto &NiNp = fem_x.getShapeFunctionOverlapIntegral();
    auto &NjNq = fem_y.getShapeFunctionOverlapIntegral();
    auto &NkNr = fem_z.getShapeFunctionOverlapIntegral();

    const std::multimap<int, int> &node2ElementMapX = fem_x.getNodeToElementMap();
    const std::multimap<int, int> &node2ElementMapY = fem_y.getNodeToElementMap();
    const std::multimap<int, int> &node2ElementMapZ = fem_z.getNodeToElementMap();

    PetscPrintf(PETSC_COMM_WORLD,
                "compute NINJ\n");
    for (int iterI = 0; iterI < rho_cartisian_map.size(); ++iterI) {
        const Cartesian<unsigned> &global_idx = rho_cartisian_map[iterI];
        int i = global_idx.x;
        int j = global_idx.y;
        int k = global_idx.z;
        if (isOnDirichletBoundary(i,
                                  j,
                                  k) == false) {
            PetscInt globalI = i + j * number_total_nodes_x + k * number_total_nodes_x * number_total_nodes_y;
            for (auto iterZ = node2ElementMapZ.lower_bound(k);
                 iterZ != node2ElementMapZ.upper_bound(k); ++iterZ) {
                for (auto iterY = node2ElementMapY.lower_bound(j);
                     iterY != node2ElementMapY.upper_bound(j); ++iterY) {
                    for (auto iterX = node2ElementMapX.lower_bound(i);
                         iterX != node2ElementMapX.upper_bound(i); ++iterX) {
                        int eleX = iterX->second;
                        int eleY = iterY->second;
                        int eleZ = iterZ->second;
                        int locali = i - eleX * (number_nodes_per_element_x - 1);
                        int localj = j - eleY * (number_nodes_per_element_y - 1);
                        int localk = k - eleZ * (number_nodes_per_element_z - 1);
                        std::vector<double> localNINJ(total_nodes_per_element,
                                                      0.0);
                        std::vector<PetscInt> localIdx(total_nodes_per_element,
                                                       0);
                        int nodescnt = 0;
                        for (int r = 0; r < number_nodes_per_element_z; ++r) {
                            for (int q = 0; q < number_nodes_per_element_y; ++q) {
                                for (int p = 0; p < number_nodes_per_element_x; ++p) {
                                    int globali = eleX * (number_nodes_per_element_x - 1) + p;
                                    int globalj = eleY * (number_nodes_per_element_y - 1) + q;
                                    int globalk = eleZ * (number_nodes_per_element_z - 1) + r;
                                    int globalJ = globali + globalj * number_total_nodes_x +
                                                  globalk * number_total_nodes_x * number_total_nodes_y;
                                    localNINJ[nodescnt] = NiNp[eleX][locali][p] * NjNq[eleY][localj][q] *
                                                          NkNr[eleZ][localk][r];
                                    localIdx[nodescnt] = globalJ;
                                    nodescnt++;
                                }
                            }
                        }
                        MatSetValues(NINJ,
                                     1,
                                     &globalI,
                                     total_nodes_per_element,
                                     localIdx.data(),
                                     localNINJ.data(),
                                     ADD_VALUES);
                    }
                }
            }
        }
    }
    MatAssemblyBegin(NINJ,
                     MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(NINJ,
                   MAT_FINAL_ASSEMBLY);
}

void PhiLinearSolverFunction::ComputedNIdNJForBoundaryNodes(Mat &BoundarydNIdNJ) {
    const std::vector<Cartesian<unsigned>> rho_cartisian_map = rho_index_map.GetLocalTensorGlobalCarteisianIndex();
    const std::vector<unsigned> rho_global_index_map = rho_index_map.GetLocalTensorGlobalIndex();

    PetscInt rho_number_local_entries = rho_global_index_map.size();
    PetscInt total_nodes = number_total_nodes_x * number_total_nodes_y * number_total_nodes_z;
    PetscInt number_nodes_per_element_x = fem_x.getNumberNodesPerElement();
    PetscInt number_nodes_per_element_y = fem_y.getNumberNodesPerElement();
    PetscInt number_nodes_per_element_z = fem_z.getNumberNodesPerElement();
    PetscInt
            total_nodes_per_element =
            number_nodes_per_element_x * number_nodes_per_element_x * number_nodes_per_element_z;
    MatCreate(PETSC_COMM_WORLD,
              &BoundarydNIdNJ);
    MatSetType(BoundarydNIdNJ,
               MATMPIAIJ);
    MatSetSizes(BoundarydNIdNJ,
                rho_number_local_entries,
                rho_number_local_entries,
                total_nodes,
                total_nodes);
//  MatMPIAIJSetPreallocation(BoundarydNIdNJ, 8 * total_nodes_per_element, PETSC_NULL, 8 * total_nodes_per_element, PETSC_NULL);
    MatMPIAIJSetPreallocation(BoundarydNIdNJ,
                              0,
                              diagonal_non_zeros.data(),
                              0,
                              offdiagonal_non_zeros.data());

    PetscPrintf(PETSC_COMM_WORLD,
                "Set BoundarydNIdNJ\n");

    auto &dNidNp = fem_x.getShapeFunctionGradientIntegral();
    auto &dNjdNq = fem_y.getShapeFunctionGradientIntegral();
    auto &dNkdNr = fem_z.getShapeFunctionGradientIntegral();

    auto &NiNp = fem_x.getShapeFunctionOverlapIntegral();
    auto &NjNq = fem_y.getShapeFunctionOverlapIntegral();
    auto &NkNr = fem_z.getShapeFunctionOverlapIntegral();

    const std::multimap<int, int> &node2ElementMapX = fem_x.getNodeToElementMap();
    const std::multimap<int, int> &node2ElementMapY = fem_y.getNodeToElementMap();
    const std::multimap<int, int> &node2ElementMapZ = fem_z.getNodeToElementMap();

    for (int iterI = 0; iterI < rho_cartisian_map.size(); ++iterI) {
        const Cartesian<unsigned> &global_idx = rho_cartisian_map[iterI];
        int i = global_idx.x;
        int j = global_idx.y;
        int k = global_idx.z;
        if (isOnDirichletBoundary(i,
                                  j,
                                  k) == false) {
            PetscInt globalI = i + j * number_total_nodes_x + k * number_total_nodes_x * number_total_nodes_y;
            for (auto iterZ = node2ElementMapZ.lower_bound(k);
                 iterZ != node2ElementMapZ.upper_bound(k); ++iterZ) {
                for (auto iterY = node2ElementMapY.lower_bound(j);
                     iterY != node2ElementMapY.upper_bound(j); ++iterY) {
                    for (auto iterX = node2ElementMapX.lower_bound(i);
                         iterX != node2ElementMapX.upper_bound(i); ++iterX) {
                        int eleX = iterX->second;
                        int eleY = iterY->second;
                        int eleZ = iterZ->second;
                        int locali = i - eleX * (number_nodes_per_element_x - 1);
                        int localj = j - eleY * (number_nodes_per_element_y - 1);
                        int localk = k - eleZ * (number_nodes_per_element_z - 1);
                        std::vector<double> localBoundarydNIdNJ(total_nodes_per_element,
                                                                0.0);
                        std::vector<PetscInt> localIdx(total_nodes_per_element,
                                                       -1);
                        int nodescnt = 0;
                        for (int r = 0; r < number_nodes_per_element_z; ++r) {
                            for (int q = 0; q < number_nodes_per_element_y; ++q) {
                                for (int p = 0; p < number_nodes_per_element_x; ++p) {
                                    int globali = eleX * (number_nodes_per_element_x - 1) + p;
                                    int globalj = eleY * (number_nodes_per_element_y - 1) + q;
                                    int globalk = eleZ * (number_nodes_per_element_z - 1) + r;
                                    int globalJ = globali + globalj * number_total_nodes_x +
                                                  globalk * number_total_nodes_x * number_total_nodes_y;
                                    if (isOnDirichletBoundary(globali,
                                                              globalj,
                                                              globalk) == true) {
                                        localBoundarydNIdNJ[nodescnt] =
                                                dNidNp[eleX][locali][p] * NjNq[eleY][localj][q] *
                                                NkNr[eleZ][localk][r] +
                                                NiNp[eleX][locali][p] * dNjdNq[eleY][localj][q] *
                                                NkNr[eleZ][localk][r] +
                                                NiNp[eleX][locali][p] * NjNq[eleY][localj][q] * dNkdNr[eleZ][localk][r];
                                        localIdx[nodescnt] = globalJ;
                                    }
                                    nodescnt++;
                                }
                            }
                        }
                        MatSetValues(BoundarydNIdNJ,
                                     1,
                                     &globalI,
                                     total_nodes_per_element,
                                     localIdx.data(),
                                     localBoundarydNIdNJ.data(),
                                     ADD_VALUES);
                    }
                }
            }
        }
    }
    MatAssemblyBegin(BoundarydNIdNJ,
                     MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(BoundarydNIdNJ,
                   MAT_FINAL_ASSEMBLY);
}

PhiLinearSolverFunction::PhiLinearSolverFunction(const FEM &fem_x,
                                                 const FEM &fem_y,
                                                 const FEM &fem_z,
                                                 Tensor3DMPI &rho,
                                                 Tensor3DMPIMap &rho_index_map,
                                                 BoundaryValuesContainer *boundary_values)
        : rho(rho),
          rho_index_map(rho_index_map),
          boundary_values(boundary_values),
          fem_x(fem_x),
          fem_y(fem_y),
          fem_z(fem_z),
          nuclei(nuclei),
          number_total_nodes_x(fem_x.getTotalNumberNodes()),
          number_total_nodes_y(fem_y.getTotalNumberNodes()),
          number_total_nodes_z(fem_z.getTotalNumberNodes()) {
    LinearSolverFunction::solution = std::vector<double>(rho.getLocalNumberEntries(),
                                                         0.0);
    ComputeNonZeroPatterns(fem_x,
                           fem_y,
                           fem_z,
                           rho.getGlobalIndex(),
                           diagonal_non_zeros,
                           offdiagonal_non_zeros);
}

void PhiLinearSolverFunction::InitializeSolution(int num_local_entries,
                                                 const double *initialize_data) {
    for (int i = 0; i < num_local_entries; ++i) {
        solution[i] = initialize_data[i];
    }
}

void PhiLinearSolverFunction::ComputeA() {
    const std::vector<Cartesian<unsigned>> rho_cartisian_map = rho_index_map.GetLocalTensorGlobalCarteisianIndex();
    const std::vector<unsigned> rho_global_index_map = rho_index_map.GetLocalTensorGlobalIndex();

    int rho_number_local_entries = rho_global_index_map.size();
    int total_nodes = number_total_nodes_x * number_total_nodes_y * number_total_nodes_z;

    int number_nodes_per_element_x = fem_x.getNumberNodesPerElement();
    int number_nodes_per_element_y = fem_y.getNumberNodesPerElement();
    int number_nodes_per_element_z = fem_z.getNumberNodesPerElement();
    int total_nodes_per_element = number_nodes_per_element_x * number_nodes_per_element_y * number_nodes_per_element_z;

    const std::vector<double> &nodesX = fem_x.getGlobalNodalCoord();
    const std::vector<double> &nodesY = fem_y.getGlobalNodalCoord();
    const std::vector<double> &nodesZ = fem_z.getGlobalNodalCoord();

    MatCreate(PETSC_COMM_WORLD,
              &A);
    MatSetType(A,
               MATMPIAIJ);
    MatSetSizes(A,
                rho_number_local_entries,
                rho_number_local_entries,
                total_nodes,
                total_nodes);
//  MatMPIAIJSetPreallocation(A, 8 * total_nodes_per_element, PETSC_NULL, 8 * total_nodes_per_element, PETSC_NULL);
    MatMPIAIJSetPreallocation(A,
                              0,
                              diagonal_non_zeros.data(),
                              0,
                              offdiagonal_non_zeros.data());
    MatSetUp(A);
    MatSetFromOptions(A);

    auto &dNidNp = fem_x.getShapeFunctionGradientIntegral();
    auto &dNjdNq = fem_y.getShapeFunctionGradientIntegral();
    auto &dNkdNr = fem_z.getShapeFunctionGradientIntegral();

    auto &NiNp = fem_x.getShapeFunctionOverlapIntegral();
    auto &NjNq = fem_y.getShapeFunctionOverlapIntegral();
    auto &NkNr = fem_z.getShapeFunctionOverlapIntegral();

    const std::multimap<int, int> &node2ElementMapX = fem_x.getNodeToElementMap();
    const std::multimap<int, int> &node2ElementMapY = fem_y.getNodeToElementMap();
    const std::multimap<int, int> &node2ElementMapZ = fem_z.getNodeToElementMap();

    for (int iterI = 0; iterI < rho_cartisian_map.size(); ++iterI) {
        const Cartesian<unsigned> &global_idx = rho_cartisian_map[iterI];
        int i = global_idx.x;
        int j = global_idx.y;
        int k = global_idx.z;
        PetscInt globalI = i + j * number_total_nodes_x + k * number_total_nodes_x * number_total_nodes_y;
        if (isOnDirichletBoundary(i,
                                  j,
                                  k) == false) {
            for (auto iterZ = node2ElementMapZ.lower_bound(k);
                 iterZ != node2ElementMapZ.upper_bound(k); ++iterZ) {
                for (auto iterY = node2ElementMapY.lower_bound(j);
                     iterY != node2ElementMapY.upper_bound(j); ++iterY) {
                    for (auto iterX = node2ElementMapX.lower_bound(i);
                         iterX != node2ElementMapX.upper_bound(i); ++iterX) {
                        int eleX = iterX->second;
                        int eleY = iterY->second;
                        int eleZ = iterZ->second;
                        int locali = i - eleX * (number_nodes_per_element_x - 1);
                        int localj = j - eleY * (number_nodes_per_element_y - 1);
                        int localk = k - eleZ * (number_nodes_per_element_z - 1);
                        std::vector<double> localA(total_nodes_per_element,
                                                   0.0);
                        std::vector<PetscInt> localIdx(total_nodes_per_element,
                                                       -1);
                        int nodescnt = 0;
                        for (int r = 0; r < number_nodes_per_element_z; ++r) {
                            for (int q = 0; q < number_nodes_per_element_y; ++q) {
                                for (int p = 0; p < number_nodes_per_element_x; ++p) {
                                    int globali = eleX * (number_nodes_per_element_x - 1) + p;
                                    int globalj = eleY * (number_nodes_per_element_y - 1) + q;
                                    int globalk = eleZ * (number_nodes_per_element_z - 1) + r;
                                    int globalJ = globali + globalj * number_total_nodes_x +
                                                  globalk * number_total_nodes_x * number_total_nodes_y;
                                    if (isOnDirichletBoundary(globali,
                                                              globalj,
                                                              globalk) == false) {
                                        localA[nodescnt] = dNidNp[eleX][locali][p] * NjNq[eleY][localj][q] *
                                                           NkNr[eleZ][localk][r] +
                                                           NiNp[eleX][locali][p] * dNjdNq[eleY][localj][q] *
                                                           NkNr[eleZ][localk][r] +
                                                           NiNp[eleX][locali][p] * NjNq[eleY][localj][q] *
                                                           dNkdNr[eleZ][localk][r];
                                        localIdx[nodescnt] = globalJ;
                                    }
                                    nodescnt++;
                                }
                            }
                        }
                        MatSetValues(A,
                                     1,
                                     &globalI,
                                     total_nodes_per_element,
                                     localIdx.data(),
                                     localA.data(),
                                     ADD_VALUES);
                    }
                }
            }
        } else {
            MatSetValue(A,
                        globalI,
                        globalI,
                        1.0,
                        ADD_VALUES);
        }
    }

    MatAssemblyBegin(A,
                     MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A,
                   MAT_FINAL_ASSEMBLY);
}

void PhiLinearSolverFunction::ComputeRhs() {
    const std::vector<Cartesian<unsigned>> rho_cartisian_map = rho_index_map.GetLocalTensorGlobalCarteisianIndex();
    std::vector<PetscInt> rho_global_index_map(rho_index_map.GetLocalTensorGlobalIndex().size());
    for (int i = 0; i < rho_global_index_map.size(); ++i) {
        rho_global_index_map[i] = rho_index_map.GetLocalTensorGlobalIndex()[i];
    }
    Mat NINJ;
    ComputeNINJWithoutBoundaryNodes(NINJ);

    Vec
            FourPiRho;
    MatCreateVecs(NINJ,
                  &FourPiRho,
                  PETSC_NULL);
    VecSetOption(FourPiRho,
                 VEC_IGNORE_NEGATIVE_INDICES,
                 PETSC_TRUE);

    PetscPrintf(PETSC_COMM_WORLD,
                "Set 4 pi rho\n");
    VecSetValues(FourPiRho,
                 rho_global_index_map.size(),
                 rho_global_index_map.data(),
                 rho.getLocalData(),
                 INSERT_VALUES);
    VecAssemblyBegin(FourPiRho);
    VecAssemblyEnd(FourPiRho);
    VecScale(FourPiRho,
             4.0 * M_PI);

    PetscPrintf(PETSC_COMM_WORLD,
                "NINJ*FourPiRho\n");
    MatCreateVecs(NINJ,
                  PETSC_NULL,
                  &rhs);
    VecSetOption(rhs,
                 VEC_IGNORE_NEGATIVE_INDICES,
                 PETSC_TRUE);
    MatMult(NINJ,
            FourPiRho,
            rhs);
    VecAssemblyBegin(rhs);
    VecAssemblyEnd(rhs);

    VecDestroy(&FourPiRho);
    MatDestroy(&NINJ);

    Mat
            BoundarydNIdNJ;
    ComputedNIdNJForBoundaryNodes(BoundarydNIdNJ);

    Vec
            dNIdNJtimesBoundaryVh, boundaryVh;
    MatCreateVecs(BoundarydNIdNJ,
                  &boundaryVh,
                  &dNIdNJtimesBoundaryVh);

    VecSetOption(boundaryVh,
                 VEC_IGNORE_NEGATIVE_INDICES,
                 PETSC_TRUE);
    VecSetOption(dNIdNJtimesBoundaryVh,
                 VEC_IGNORE_NEGATIVE_INDICES,
                 PETSC_TRUE);

    VecSet(boundaryVh,
           0.0);
    std::vector<PetscInt> boundaryVhIdx(boundary_values->GetBoundaryValuesGlobalIndex().size());
    for (int j = 0; j < boundaryVhIdx.size(); ++j) {
        boundaryVhIdx[j] = boundary_values->GetBoundaryValuesGlobalIndex()[j];
    }
    const std::vector<double> &vecBoundaryVh = boundary_values->GetLocalBoundaryValues();
    VecSetValues(boundaryVh,
                 boundaryVhIdx.size(),
                 boundaryVhIdx.data(),
                 vecBoundaryVh.data(),
                 INSERT_VALUES);
    VecAssemblyBegin(boundaryVh);
    VecAssemblyEnd(boundaryVh);

    PetscPrintf(PETSC_COMM_WORLD,
                "Compute BdNIdNJ*Vh\n");
    MatMult(BoundarydNIdNJ,
            boundaryVh,
            dNIdNJtimesBoundaryVh);

    MatDestroy(&BoundarydNIdNJ);

    VecAXPBY(rhs,
             -1.0,
             1.0,
             dNIdNJtimesBoundaryVh);

    VecSetOption(rhs,
                 VEC_IGNORE_NEGATIVE_INDICES,
                 PETSC_TRUE);
    VecSetValues(rhs,
                 boundaryVhIdx.size(),
                 boundaryVhIdx.data(),
                 vecBoundaryVh.data(),
                 INSERT_VALUES);
    VecAssemblyBegin(rhs);
    VecAssemblyEnd(rhs);
    VecDestroy(&dNIdNJtimesBoundaryVh);
    VecDestroy(&boundaryVh);

}

bool PhiLinearSolverFunction::isOnDirichletBoundary(int i,
                                                    int j,
                                                    int k) {
    return ((i == 0) || (i == (number_total_nodes_x - 1) || (j == 0) || (j == (number_total_nodes_y - 1)) || (k == 0)
                         || (k == (number_total_nodes_z - 1))));
}