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

#include "TuckerBasis.h"
#include "../dft/solver/FunctionalRayleighQuotientSeperableNonLocal.h"
#include <TuckerMPI.hpp>

TuckerBasis::TuckerBasis(
        int rank_x,
        int rank_y,
        int rank_z,
        const FEM &fem_x,
        const FEM &fem_y,
        const FEM &fem_z,
        const FEM &fem_nonloc_x,
        const FEM &fem_nonloc_y,
        const FEM &fem_nonloc_z,
        const TuckerMPI::TuckerTensor *decomposedPotEff,
        std::vector<std::shared_ptr<NonLocalMapManager>> &nonloc_map_manager,
        std::vector<std::shared_ptr<NonLocalPSPData>> &nonloc_psp_data)
        : basis_x(rank_x,
                  std::vector<double>(fem_x.getTotalNumberNodes(),
                                      0)),
          basis_y(rank_y,
                  std::vector<double>(fem_y.getTotalNumberNodes(),
                                      0)),
          basis_z(rank_z,
                  std::vector<double>(fem_z.getTotalNumberNodes(),
                                      0)),
          rank_x(rank_x),
          rank_y(rank_y),
          rank_z(rank_z),
          fem_x(fem_x),
          fem_y(fem_y),
          fem_z(fem_z),
          fem_nonloc_x(fem_nonloc_x),
          fem_nonloc_y(fem_nonloc_y),
          fem_nonloc_z(fem_nonloc_z),
          decomposedPotEff(decomposedPotEff),
          nonloc_map_manager(nonloc_map_manager),
          nonloc_psp_data(nonloc_psp_data),
          initial_guess_x(fem_x.getTotalNumberNodes(),
                          0),
          initial_guess_y(fem_y.getTotalNumberNodes(),
                          0),
          initial_guess_z(fem_z.getTotalNumberNodes(),
                          0) {}

void TuckerBasis::initialize_ig(Tensor3DMPI &psi) {
    Tucker::SizeArray rankIG(3);
    rankIG[0] = 1;
    rankIG[1] = 1;
    rankIG[2] = 1;
    const TuckerMPI::TuckerTensor *Psi =
            TuckerMPI::STHOSVD(psi.getTensor(),
                               &rankIG,
                               true,
                               false);

    const double *guessPsix = Psi->U[0]->data();
    const double *guessPsiy = Psi->U[1]->data();
    const double *guessPsiz = Psi->U[2]->data();

    int numberNodesX = fem_x.getTotalNumberNodes();
    int numberNodesY = fem_y.getTotalNumberNodes();
    int numberNodesZ = fem_z.getTotalNumberNodes();

    std::copy(guessPsix + 1,
              guessPsix + numberNodesX - 1,
              initial_guess_x.begin() + 1);
    std::copy(guessPsiy + 1,
              guessPsiy + numberNodesY - 1,
              initial_guess_y.begin() + 1);
    std::copy(guessPsiz + 1,
              guessPsiz + numberNodesZ - 1,
              initial_guess_z.begin() + 1);
}

void TuckerBasis::basis_solver(int rx,
                               int ry,
                               int rz,
                               std::vector<std::vector<double>> &eig_x,
                               std::vector<std::vector<double>> &eig_y,
                               std::vector<std::vector<double>> &eig_z,
                               double tolerance,
                               int maxIter,
                               double alpha,
                               int number_history,
                               SeparableSCFType scf_separable_type) {
    FunctionalRayleighQuotientSeperable *functional =
            new FunctionalRayleighQuotientSeperableNonLocal(
                    fem_x,
                    fem_y,
                    fem_z,
                    *decomposedPotEff,
                    fem_nonloc_x,
                    fem_nonloc_y,
                    fem_nonloc_z,
                    nonloc_map_manager,
                    nonloc_psp_data);
    SeparableHamiltonian separableHamiltonian(functional);

    int num_nodes_x = fem_x.getTotalNumberNodes();
    int num_nodes_y = fem_y.getTotalNumberNodes();
    int num_nodes_z = fem_z.getTotalNumberNodes();

    int taskId;
    MPI_Comm_rank(MPI_COMM_WORLD,
                  &taskId);

    if (taskId == 0) {
        separableHamiltonian.solveSCF(initial_guess_x,
                                      initial_guess_y,
                                      initial_guess_z,
                                      scf_separable_type,
                                      tolerance,
                                      maxIter,
                                      alpha,
                                      number_history);
    }

    if (taskId == 0) {
        std::vector<double> nodal_field_x = separableHamiltonian.getNodalFieldX();
        std::vector<double> nodal_field_y = separableHamiltonian.getNodalFieldY();
        std::vector<double> nodal_field_z = separableHamiltonian.getNodalFieldZ();

        Mat Hx, Hy, Hz, Mx, My, Mz;
        PetscInt mx = num_nodes_x - 2, qx = num_nodes_x - 2;
        PetscInt my = num_nodes_y - 2, qy = num_nodes_y - 2;
        PetscInt mz = num_nodes_z - 2, qz = num_nodes_z - 2;
        MatCreateSeqDense(PETSC_COMM_SELF,
                          mx,
                          qx,
                          NULL,
                          &Hx);
        MatCreateSeqDense(PETSC_COMM_SELF,
                          mx,
                          qx,
                          NULL,
                          &Mx);
        MatCreateSeqDense(PETSC_COMM_SELF,
                          my,
                          qy,
                          NULL,
                          &Hy);
        MatCreateSeqDense(PETSC_COMM_SELF,
                          my,
                          qy,
                          NULL,
                          &My);
        MatCreateSeqDense(PETSC_COMM_SELF,
                          mz,
                          qz,
                          NULL,
                          &Hz);
        MatCreateSeqDense(PETSC_COMM_SELF,
                          mz,
                          qz,
                          NULL,
                          &Mz);
        functional->generateHamiltonianGenericPotential(
                nodal_field_x,
                nodal_field_y,
                nodal_field_z,
                Hx,
                Hy,
                Hz,
                Mx,
                My,
                Mz);
        functional->solveHamiltonianGenericPotential(Hx,
                                                     Hy,
                                                     Hz,
                                                     Mx,
                                                     My,
                                                     Mz,
                                                     rx,
                                                     ry,
                                                     ry,
                                                     eig_x,
                                                     eig_y,
                                                     eig_z);

        PetscViewer viewer;
        PetscViewerCreate(PETSC_COMM_SELF,
                          &viewer);
        PetscViewerSetType(viewer,
                           PETSCVIEWERASCII);
        PetscViewerPushFormat(viewer,
                              PETSC_VIEWER_ASCII_MATLAB);
        PetscViewerFileSetMode(viewer,
                               FILE_MODE_WRITE);
        PetscViewerFileSetName(viewer,
                               "Hx.m");
        MatView(Hx,
                viewer);
        PetscViewerDestroy(&viewer);
        PetscViewerCreate(PETSC_COMM_SELF,
                          &viewer);
        PetscViewerSetType(viewer,
                           PETSCVIEWERASCII);
        PetscViewerPushFormat(viewer,
                              PETSC_VIEWER_ASCII_MATLAB);
        PetscViewerFileSetMode(viewer,
                               FILE_MODE_WRITE);
        PetscViewerFileSetName(viewer,
                               "Mx.m");
        MatView(Mx,
                viewer);
        PetscViewerDestroy(&viewer);

        PetscViewerCreate(PETSC_COMM_SELF,
                          &viewer);
        PetscViewerSetType(viewer,
                           PETSCVIEWERASCII);
        PetscViewerPushFormat(viewer,
                              PETSC_VIEWER_ASCII_MATLAB);
        PetscViewerFileSetMode(viewer,
                               FILE_MODE_WRITE);
        PetscViewerFileSetName(viewer,
                               "Hy.m");
        MatView(Hy,
                viewer);
        PetscViewerDestroy(&viewer);
        PetscViewerCreate(PETSC_COMM_SELF,
                          &viewer);
        PetscViewerSetType(viewer,
                           PETSCVIEWERASCII);
        PetscViewerPushFormat(viewer,
                              PETSC_VIEWER_ASCII_MATLAB);
        PetscViewerFileSetMode(viewer,
                               FILE_MODE_WRITE);
        PetscViewerFileSetName(viewer,
                               "My.m");
        MatView(My,
                viewer);
        PetscViewerDestroy(&viewer);

        PetscViewerCreate(PETSC_COMM_SELF,
                          &viewer);
        PetscViewerSetType(viewer,
                           PETSCVIEWERASCII);
        PetscViewerPushFormat(viewer,
                              PETSC_VIEWER_ASCII_MATLAB);
        PetscViewerFileSetMode(viewer,
                               FILE_MODE_WRITE);
        PetscViewerFileSetName(viewer,
                               "Hz.m");
        MatView(Hz,
                viewer);
        PetscViewerDestroy(&viewer);
        PetscViewerCreate(PETSC_COMM_SELF,
                          &viewer);
        PetscViewerSetType(viewer,
                           PETSCVIEWERASCII);
        PetscViewerPushFormat(viewer,
                              PETSC_VIEWER_ASCII_MATLAB);
        PetscViewerFileSetMode(viewer,
                               FILE_MODE_WRITE);
        PetscViewerFileSetName(viewer,
                               "Mz.m");
        MatView(Mz,
                viewer);
        PetscViewerDestroy(&viewer);
    }
    delete functional;

    for (int i = 0; i < eig_x.size(); ++i) {
        MPI_Bcast(eig_x[i].data(),
                  eig_x[i].size(),
                  MPI_DOUBLE,
                  0,
                  MPI_COMM_WORLD);
    }
    for (int i = 0; i < eig_y.size(); ++i) {
        MPI_Bcast(eig_y[i].data(),
                  eig_y[i].size(),
                  MPI_DOUBLE,
                  0,
                  MPI_COMM_WORLD);
    }
    for (int i = 0; i < eig_z.size(); ++i) {
        MPI_Bcast(eig_z[i].data(),
                  eig_z[i].size(),
                  MPI_DOUBLE,
                  0,
                  MPI_COMM_WORLD);
    }
}

void TuckerBasis::solve_for_basis(double tolerance,
                                  int maxIter,
                                  double alpha,
                                  int number_history,
                                  SeparableSCFType scf_separable_type) {
    basis_solver(rank_x,
                 rank_y,
                 rank_z,
                 basis_x,
                 basis_y,
                 basis_z,
                 tolerance,
                 maxIter,
                 alpha,
                 number_history,
                 scf_separable_type);
}