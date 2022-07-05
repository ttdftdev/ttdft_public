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

#include "HartreeCalculator.h"
#include "LinearSolver/PhiLinearSolverFunction.h"
#include "../tensor/TensorUtils.h"

namespace HartreeCalculator {
    void ProjectSeqNodalTensorOntoMPIQuadTensor(const FEM &femX,
                                                const FEM &femY,
                                                const FEM &femZ,
                                                const std::vector<double> &seq_tensor,
                                                Tensor3DMPI &quad_tensor) {

        int numberTotalNodesX = femX.getTotalNumberNodes();
        int numberTotalNodesY = femY.getTotalNumberNodes();
        int numberTotalNodesZ = femZ.getTotalNumberNodes();

        int numberNodesPerEleX = femX.getNumberNodesPerElement();
        int numberNodesPerEleY = femY.getNumberNodesPerElement();
        int numberNodesPerEleZ = femZ.getNumberNodesPerElement();

        int numberQuadPerEleX = femX.getNumberQuadPointsPerElement();
        int numberQuadPerEleY = femY.getNumberQuadPointsPerElement();
        int numberQuadPerEleZ = femZ.getNumberQuadPointsPerElement();

        const std::vector<std::vector<double> > &shpFuncX = femX.getShapeFunctionAtQuadPoints();
        const std::vector<std::vector<double> > &shpFuncY = femY.getShapeFunctionAtQuadPoints();
        const std::vector<std::vector<double> > &shpFuncZ = femZ.getShapeFunctionAtQuadPoints();

        std::array<int, 6> quad_tensor_idx;
        double *quad_tensor_data = quad_tensor.getLocalData(quad_tensor_idx);
        int quadcnt = 0;
        for (int kquad = quad_tensor_idx[4]; kquad < quad_tensor_idx[5]; ++kquad) {
            for (int jquad = quad_tensor_idx[2]; jquad < quad_tensor_idx[3]; ++jquad) {
                for (int iquad = quad_tensor_idx[0]; iquad < quad_tensor_idx[1]; ++iquad) {
                    int elei = std::floor(iquad / numberQuadPerEleX);
                    int elej = std::floor(jquad / numberQuadPerEleY);
                    int elek = std::floor(kquad / numberQuadPerEleZ);

                    int quadi = iquad % numberQuadPerEleX;
                    int quadj = jquad % numberQuadPerEleY;
                    int quadk = kquad % numberQuadPerEleZ;

                    int eleNodeOriginx = elei * (numberNodesPerEleX - 1);
                    int eleNodeOriginy = elej * (numberNodesPerEleY - 1);
                    int eleNodeOriginz = elek * (numberNodesPerEleZ - 1);

                    quad_tensor_data[quadcnt] = 0;
                    for (int nodek = 0; nodek < numberNodesPerEleZ; nodek++) {
                        for (int nodej = 0; nodej < numberNodesPerEleY; nodej++) {
                            for (int nodei = 0; nodei < numberNodesPerEleX; ++nodei) {
                                int nodeItr = (nodek + eleNodeOriginz) * numberTotalNodesX * numberTotalNodesY +
                                              (nodej + eleNodeOriginy) * numberTotalNodesX + (nodei + eleNodeOriginx);
                                quad_tensor_data[quadcnt] +=
                                        seq_tensor[nodeItr] * shpFuncX[nodei][quadi] * shpFuncY[nodej][quadj] *
                                        shpFuncZ[nodek][quadk];
                            }
                        }
                    }
                    quadcnt++;
                }
            }
        }
    }

    void ProjectHartreePotentialOntoQuadPoints(const FEM &femX,
                                               const FEM &femY,
                                               const FEM &femZ,
                                               Tensor3DMPI &hartreeNode,
                                               Tensor3DMPI &hartQuad) {
        std::vector<double> collectiveHartNode;
        TensorUtils::AllGather3DMPITensor(hartreeNode,
                                          collectiveHartNode);

        ProjectSeqNodalTensorOntoMPIQuadTensor(femX,
                                               femY,
                                               femZ,
                                               collectiveHartNode,
                                               hartQuad);
    }

    void SolveHartreePotential(const FEM &femX,
                               const FEM &femY,
                               const FEM &femZ,
                               const int maxIter,
                               const double tolerance,
                               PETScLinearSolver::Solver &solver_type,
                               PETScLinearSolver::Preconditioner &preconditioner_type,
                               Tensor3DMPI &rho_node,
                               Tensor3DMPIMap &hartree_index_map,
                               Tensor3DMPI &hartree_node,
                               BoundaryValuesContainer *phi_boundary_values) {
        LinearSolver *petsc_linear_solver = new PETScLinearSolver(maxIter,
                                                                  tolerance,
                                                                  solver_type,
                                                                  preconditioner_type);

        LinearSolverFunction *phi_linear_solver_function = new PhiLinearSolverFunction(femX,
                                                                                       femY,
                                                                                       femZ,
                                                                                       rho_node,
                                                                                       hartree_index_map,
                                                                                       phi_boundary_values);
        phi_linear_solver_function->ComputeRhs();
        phi_linear_solver_function->ComputeA();

        int num_local_entries_hartree = hartree_node.getLocalNumberEntries();
        double *hartree_node_data = hartree_node.getLocalData();
        phi_linear_solver_function->InitializeSolution(num_local_entries_hartree,
                                                       hartree_node_data);

        petsc_linear_solver->solve(phi_linear_solver_function);

        const std::vector<double> &phi_solution = phi_linear_solver_function->getSolution();
        for (int i = 0; i < num_local_entries_hartree; ++i) {
            hartree_node_data[i] = phi_solution[i];
        }
        delete phi_linear_solver_function;
        delete petsc_linear_solver;
    }

/*
 * @param hatree_node this is still passed instead of creating in the code because it could be used as the initial guess
 */
    void CalculateHartreePotentialOnQuadPoints(const FEM &femX,
                                               const FEM &femY,
                                               const FEM &femZ,
                                               const int maxIter,
                                               const double tolerance,
                                               Tensor3DMPI &rho_node,
                                               Tensor3DMPIMap &hartree_index_map,
                                               Tensor3DMPI &hartree_node,
                                               BoundaryValuesContainer *phi_boundary_values,
                                               Tensor3DMPI &hartree_quad) {
        PETScLinearSolver::Solver solver_type = PETScLinearSolver::Solver::BCGS;
        PETScLinearSolver::Preconditioner preconditioner_type = PETScLinearSolver::Preconditioner::BJACOBI;
        SolveHartreePotential(femX,
                              femY,
                              femZ,
                              maxIter,
                              tolerance,
                              solver_type,
                              preconditioner_type,
                              rho_node,
                              hartree_index_map,
                              hartree_node,
                              phi_boundary_values);
        ProjectHartreePotentialOntoQuadPoints(femX,
                                              femY,
                                              femZ,
                                              hartree_node,
                                              hartree_quad);
    }

// local_po_idx_map and global_ks_idx_map are a pair to map from one to another, they should always be used together
// it was not created as a std::pair to reserve some flexibilities in case of extracting all indices
    void ComputeOwnedKSToPOMap(const FEM &ks_fem_x,
                               const FEM &ks_fem_y,
                               const FEM &ks_fem_z,
                               const FEM &po_fem_x,
                               const FEM &po_fem_y,
                               const FEM &po_fem_z,
                               const int *owned_po_idx,
                               std::vector<unsigned> &local_po_idx_map,
                               std::vector<unsigned> &global_ks_idx_map) {
        int number_total_nodes_po_x = po_fem_x.getTotalNumberNodes();
        int number_total_nodes_po_y = po_fem_y.getTotalNumberNodes();
        int number_total_nodes_po_z = po_fem_z.getTotalNumberNodes();
        int number_inner_nodes_x = ks_fem_x.getTotalNumberNodes();
        int number_inner_nodes_y = ks_fem_y.getTotalNumberNodes();
        int number_inner_nodes_z = ks_fem_z.getTotalNumberNodes();
        int number_outer_nodes_one_side_x = (number_total_nodes_po_x - number_inner_nodes_x) / 2;
        int number_outer_nodes_one_side_y = (number_total_nodes_po_y - number_inner_nodes_y) / 2;
        int number_outer_nodes_one_side_z = (number_total_nodes_po_z - number_inner_nodes_z) / 2;

        int ks_domain_boundary_idx[6]; // the index of ks domain boundary in po domain
        ks_domain_boundary_idx[0] = number_outer_nodes_one_side_x;
        ks_domain_boundary_idx[2] = number_outer_nodes_one_side_y;
        ks_domain_boundary_idx[4] = number_outer_nodes_one_side_z;
        ks_domain_boundary_idx[1] = number_outer_nodes_one_side_x + number_inner_nodes_x;
        ks_domain_boundary_idx[3] = number_outer_nodes_one_side_y + number_inner_nodes_y;
        ks_domain_boundary_idx[5] = number_outer_nodes_one_side_z + number_inner_nodes_z;

        auto CheckIsInKSDomain = [ks_domain_boundary_idx](int i,
                                                          int j,
                                                          int k) {
            return ((i >= ks_domain_boundary_idx[0]) && (i < ks_domain_boundary_idx[1])
                    && (j >= ks_domain_boundary_idx[2]) && (j < ks_domain_boundary_idx[3])
                    && (k >= ks_domain_boundary_idx[4]) && (k < ks_domain_boundary_idx[5]));
        };

        auto ConvertPoIdxToGlobalKSIdx = [number_outer_nodes_one_side_x,
                number_outer_nodes_one_side_y,
                number_outer_nodes_one_side_z,
                number_inner_nodes_x,
                number_inner_nodes_y,
                number_inner_nodes_z](int i,
                                      int j,
                                      int k) {
            return (i - number_outer_nodes_one_side_x) + (j - number_outer_nodes_one_side_y) * number_inner_nodes_x
                   + (k - number_outer_nodes_one_side_z) * number_inner_nodes_x * number_inner_nodes_y;
        };

        unsigned local_po_idx_iterator = 0;
        for (int k = owned_po_idx[4]; k < owned_po_idx[5]; ++k) {
            for (int j = owned_po_idx[2]; j < owned_po_idx[3]; ++j) {
                for (int i = owned_po_idx[0]; i < owned_po_idx[1]; ++i) {
                    if (CheckIsInKSDomain(i,
                                          j,
                                          k)) {
                        int global_kx_idx = ConvertPoIdxToGlobalKSIdx(i,
                                                                      j,
                                                                      k);
                        global_ks_idx_map.emplace_back(global_kx_idx);
                        local_po_idx_map.emplace_back(local_po_idx_iterator);
                    }
                    local_po_idx_iterator++;
                }
            }
        }
    }

    void CopyRhoFromKSToPO(const std::vector<unsigned> &local_po_idx_map,
                           const std::vector<unsigned> &global_ks_idx_map,
                           const std::vector<double> &rho_ks_seq,
                           Tensor3DMPI &rho_po) {

        double *rho_po_data = rho_po.getLocalData();
        for (int i = 0; i < local_po_idx_map.size(); ++i) {
            rho_po_data[local_po_idx_map[i]] = rho_ks_seq[global_ks_idx_map[i]];
        }
    }

    void ComputeOwnedPoToKSMap(const FEM &po_fem_x,
                               const FEM &po_fem_y,
                               const FEM &po_fem_z,
                               const FEM &ks_fem_x,
                               const FEM &ks_fem_y,
                               const FEM &ks_fem_z,
                               const int *owned_ks_idx,
                               std::vector<unsigned> &global_po_idx_map) {
        int number_total_nodes_po_x = po_fem_x.getTotalNumberNodes();
        int number_total_nodes_po_y = po_fem_y.getTotalNumberNodes();
        int number_total_nodes_po_z = po_fem_z.getTotalNumberNodes();
        int number_inner_nodes_x = ks_fem_x.getTotalNumberNodes();
        int number_inner_nodes_y = ks_fem_y.getTotalNumberNodes();
        int number_inner_nodes_z = ks_fem_z.getTotalNumberNodes();
        int number_outer_nodes_one_side_x = (number_total_nodes_po_x - number_inner_nodes_x) / 2;
        int number_outer_nodes_one_side_y = (number_total_nodes_po_y - number_inner_nodes_y) / 2;
        int number_outer_nodes_one_side_z = (number_total_nodes_po_z - number_inner_nodes_z) / 2;

        auto ConvertKSIdxToGlobalPOIdx = [number_outer_nodes_one_side_x,
                number_outer_nodes_one_side_y,
                number_outer_nodes_one_side_z,
                number_total_nodes_po_x,
                number_total_nodes_po_y,
                number_total_nodes_po_z](int i,
                                         int j,
                                         int k) {
            return (i + number_outer_nodes_one_side_x) + (j + number_outer_nodes_one_side_y) * number_total_nodes_po_x
                   + (k + number_outer_nodes_one_side_z) * number_total_nodes_po_x * number_total_nodes_po_y;
        };

        for (int k = owned_ks_idx[4]; k < owned_ks_idx[5]; ++k) {
            for (int j = owned_ks_idx[2]; j < owned_ks_idx[3]; ++j) {
                for (int i = owned_ks_idx[0]; i < owned_ks_idx[1]; ++i) {
                    int global_po_idx = ConvertKSIdxToGlobalPOIdx(i,
                                                                  j,
                                                                  k);
                    global_po_idx_map.emplace_back(global_po_idx);
                }
            }
        }
    }

    void CopyRhoFromPOToKS(const std::vector<unsigned> &global_po_idx_map,
                           const double *rho_po_seq,
                           double *local_rho_ks) {
        for (int i = 0; i < global_po_idx_map.size(); ++i) {
            local_rho_ks[i] = rho_po_seq[global_po_idx_map[i]];
        }
    }

    void CalculateHartreePotentialOnQuadPointsUsingLargerDomain(const FEM &ks_fem_x,
                                                                const FEM &ks_fem_y,
                                                                const FEM &ks_fem_z,
                                                                const FEM &po_fem_x,
                                                                const FEM &po_fem_y,
                                                                const FEM &po_fem_z,
                                                                const int maxIter,
                                                                const double tolerance,
                                                                PETScLinearSolver::Solver &solver_type,
                                                                PETScLinearSolver::Preconditioner &preconditioner_type,
                                                                Tensor3DMPI &rho_node_ks_domain,
                                                                Tensor3DMPIMap &hartree_index_map_po_domain,
                                                                Tensor3DMPI &hartree_node_po_domain,
                                                                BoundaryValuesContainer *phi_boundary_values,
                                                                Tensor3DMPI &hartreeQuad) {

        int taskId;
        MPI_Comm_rank(MPI_COMM_WORLD,
                      &taskId);
        if (taskId == 0) {
            std::cout << "total elements adaptive mesh for hartree: " << po_fem_x.getNumberElements() << std::endl;
            std::cout << "coarsing factor: " << po_fem_x.getCoarsingFactor() << std::endl;
            std::cout << "domain boundary: (" << po_fem_x.get_domainStart() << ", " << po_fem_x.get_domainEnd() << ")"
                      << std::endl;
        }

        std::vector<double> seq_rho_temp;
        TensorUtils::AllGather3DMPITensor(rho_node_ks_domain,
                                          seq_rho_temp);

        int total_nodes_po_x = po_fem_x.getTotalNumberNodes();
        int total_nodes_po_y = po_fem_y.getTotalNumberNodes();
        int total_nodes_po_z = po_fem_z.getTotalNumberNodes();

        Tensor3DMPI rho_node_po_domain(total_nodes_po_x,
                                       total_nodes_po_y,
                                       total_nodes_po_z,
                                       MPI_COMM_WORLD);
        Tensor3DMPIMap rho_node_po_domain_idx_map(po_fem_x,
                                                  po_fem_y,
                                                  po_fem_z,
                                                  rho_node_po_domain);
        rho_node_po_domain.setEntriesZero();

        // Copy rho from ks domain to po domain
        std::vector<unsigned> local_po_idx_map;
        std::vector<unsigned> global_ks_idx_map;
        ComputeOwnedKSToPOMap(ks_fem_x,
                              ks_fem_y,
                              ks_fem_z,
                              po_fem_x,
                              po_fem_y,
                              po_fem_z,
                              rho_node_po_domain.getGlobalIndex().data(),
                              local_po_idx_map,
                              global_ks_idx_map);
        CopyRhoFromKSToPO(local_po_idx_map,
                          global_ks_idx_map,
                          seq_rho_temp,
                          rho_node_po_domain);

        // release memory occupied by seq_rho_temp
        seq_rho_temp.clear();
        seq_rho_temp.shrink_to_fit();

        SolveHartreePotential(po_fem_x,
                              po_fem_y,
                              po_fem_z,
                              maxIter,
                              tolerance,
                              solver_type,
                              preconditioner_type,
                              rho_node_po_domain,
                              hartree_index_map_po_domain,
                              hartree_node_po_domain,
                              phi_boundary_values);

        // Copy Hartree potential from possion domain back to ks domain
        std::vector<double> seq_hartree_temp;
        TensorUtils::AllGather3DMPITensor(hartree_node_po_domain,
                                          seq_hartree_temp);
        int total_nodes_ks_x = ks_fem_x.getTotalNumberNodes();
        int total_nodes_ks_y = ks_fem_y.getTotalNumberNodes();
        int total_nodes_ks_z = ks_fem_z.getTotalNumberNodes();
        Tensor3DMPI hartree_node_ks_domain(total_nodes_ks_x,
                                           total_nodes_ks_y,
                                           total_nodes_ks_z,
                                           MPI_COMM_WORLD);
        std::vector<unsigned> global_po_idx_map;
        ComputeOwnedPoToKSMap(po_fem_x,
                              po_fem_y,
                              po_fem_z,
                              ks_fem_x,
                              ks_fem_y,
                              ks_fem_z,
                              hartree_node_ks_domain.getGlobalIndex().data(),
                              global_po_idx_map);
        CopyRhoFromPOToKS(global_po_idx_map,
                          seq_hartree_temp.data(),
                          hartree_node_ks_domain.getLocalData());

        seq_hartree_temp.clear();
        seq_hartree_temp.shrink_to_fit();

        // Project Hartree potential from nodes in ks domain to quadrature points in ks domain
        ProjectHartreePotentialOntoQuadPoints(ks_fem_x,
                                              ks_fem_y,
                                              ks_fem_z,
                                              hartree_node_ks_domain,
                                              hartreeQuad);
    }
}