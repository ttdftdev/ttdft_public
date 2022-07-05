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
#include <petsc.h>
#include "ProjectHamiltonianSparse.h"
#include "../tensor/TensorUtils.h"
#include "../utils/Utils.h"
#include "../atoms/NonLocalMap1D.h"

namespace {
    extern "C" {
    double ddot_(int *n,
                 const double *dx,
                 int *incx,
                 const double *dy,
                 int *incy);
    }
}

ProjectHamiltonianSparse::ProjectHamiltonianSparse(const FEM &femX,
                                                   const FEM &femY,
                                                   const FEM &femZ,
                                                   const FEM &femLinearX,
                                                   const FEM &femLinearY,
                                                   const FEM &femLinearZ,
                                                   const std::vector<std::vector<double>> &eigX,
                                                   const std::vector<std::vector<double>> &eigY,
                                                   const std::vector<std::vector<double>> &eigZ) :
        femX(femX),
        femY(femY),
        femZ(femZ),
        femLinearX(femLinearX),
        femLinearY(femLinearY),
        femLinearZ(femLinearZ),
        eigX(eigX),
        eigY(eigY),
        eigZ(eigZ),
        basisXQuadValues(eigX.size()),
        basisDXQuadValues(eigX.size()),
        basisYQuadValues(eigY.size()),
        basisDYQuadValues(eigY.size()),
        basisZQuadValues(eigZ.size()),
        basisDZQuadValues(eigZ.size()) {
    int rankTuckerBasisX = eigX.size();
    int rankTuckerBasisY = eigY.size();
    int rankTuckerBasisZ = eigZ.size();
    for (auto i = 0; i < rankTuckerBasisX; ++i) {
        femX.computeFieldAndDiffFieldAtAllQuadPoints(eigX[i],
                                                     basisXQuadValues[i],
                                                     basisDXQuadValues[i]);
    }
    for (auto i = 0; i < rankTuckerBasisY; ++i) {
        femY.computeFieldAndDiffFieldAtAllQuadPoints(eigY[i],
                                                     basisYQuadValues[i],
                                                     basisDYQuadValues[i]);
    }
    for (auto i = 0; i < rankTuckerBasisZ; ++i) {
        femZ.computeFieldAndDiffFieldAtAllQuadPoints(eigZ[i],
                                                     basisZQuadValues[i],
                                                     basisDZQuadValues[i]);
    }
}

void ProjectHamiltonianSparse::Create_Cnloc(const FEM &femNonLocX,
                                            const FEM &femNonLocY,
                                            const FEM &femNonLocZ,
                                            const FEM &femNonLocLinearX,
                                            const FEM &femNonLocLinearY,
                                            const FEM &femNonLocLinearZ,
                                            const int rankTuckerBasisX,
                                            const int rankTuckerBasisY,
                                            const int rankTuckerBasisZ,
                                            const std::vector<std::shared_ptr<NonLocalPSPData>> &nonLocalPSPData,
//                                            const std::vector<std::shared_ptr<NonLocalMapManager>> &nonLocalMapManager,
                                            const AtomInformation &atom_information,
                                            double truncation) {

    PetscInt num_global_rows, num_local_rows, owned_row_start, owned_row_end;
    std::vector<NonLocRowInfo> owned_rows_info;
    std::vector<int> owned_atom_id;
    std::map<int, int> owned_atom_id_g2l;
    std::vector<int> owned_atom_types;
    std::map<int, int> owned_atom_types_g2l;
    Create_Cnloc_mat_info(atom_information,
                          num_global_rows,
                          num_local_rows,
                          owned_row_start,
                          owned_row_end,
                          owned_rows_info,
                          owned_atom_id,
                          owned_atom_id_g2l,
                          owned_atom_types,
                          owned_atom_types_g2l);


    // construct a vector to store atom type, atom id, lm id, global row idx for  each row
    int num_owned_atom_types = owned_atom_types.size();
    std::vector<std::vector<std::vector<std::vector<double>>>> umatNlocTimesJacobTimesWeight(num_owned_atom_types),
            vmatNlocTimesJacobTimesWeight(num_owned_atom_types), wmatNlocTimesJacobTimesWeight(num_owned_atom_types);
    for (int atom_type_i = 0; atom_type_i < num_owned_atom_types; ++atom_type_i) {
        int atom_type_global = owned_atom_types[atom_type_i];
        int rankNloc = nonLocalPSPData[atom_type_global]->getRankNloc();
        int lMax = nonLocalPSPData[atom_type_global]->getLMax();
        computeFactorsProduct(femX,
                              femNonLocX,
                              eigX,
                              rankNloc,
                              lMax,
                              nonLocalPSPData[atom_type_global]->getUmatQuad(),
                              umatNlocTimesJacobTimesWeight[atom_type_i]);
        computeFactorsProduct(femY,
                              femNonLocY,
                              eigY,
                              rankNloc,
                              lMax,
                              nonLocalPSPData[atom_type_global]->getVmatQuad(),
                              vmatNlocTimesJacobTimesWeight[atom_type_i]);
        computeFactorsProduct(femZ,
                              femNonLocZ,
                              eigZ,
                              rankNloc,
                              lMax,
                              nonLocalPSPData[atom_type_global]->getWmatQuad(),
                              wmatNlocTimesJacobTimesWeight[atom_type_i]);
    }

    NonLocalMap::NonLocalMap1DFactory nonlocal_map_factory_x(femX,
                                                             femLinearX,
                                                             femNonLocX,
                                                             femNonLocLinearX);
    NonLocalMap::NonLocalMap1DFactory nonlocal_map_factory_y(femY,
                                                             femLinearY,
                                                             femNonLocY,
                                                             femNonLocLinearY);
    NonLocalMap::NonLocalMap1DFactory nonlocal_map_factory_z(femZ,
                                                             femLinearZ,
                                                             femNonLocZ,
                                                             femNonLocLinearZ);
    int num_owned_atoms = owned_atom_id.size();
//  std::vector<NonLocalMap::NonLocalMap1D> nonlocal_map_x(num_owned_atoms), nonlocal_map_y(num_owned_atoms), nonlocal_map_z(num_owned_atoms);
    std::vector<std::vector<std::vector<double>>> basis_x_nonloc(num_owned_atoms), basis_y_nonloc(num_owned_atoms),
            basis_z_nonloc(num_owned_atoms);
    for (int atom_i = 0; atom_i < num_owned_atoms; ++atom_i) {
        int global_atom_i = owned_atom_id[atom_i];
        const std::vector<double> &coord = atom_information.all_nuclei[global_atom_i];
        NonLocalMap::NonLocalMap1D nonlocal_map_x, nonlocal_map_y, nonlocal_map_z;
        nonlocal_map_factory_x.generateNonLocalMap(coord[1],
                                                   nonlocal_map_x);
        nonlocal_map_factory_y.generateNonLocalMap(coord[2],
                                                   nonlocal_map_y);
        nonlocal_map_factory_z.generateNonLocalMap(coord[3],
                                                   nonlocal_map_z);

        computeFEMNL(femX,
                     femNonLocX,
                     eigX,
                     nonlocal_map_x,
                     basis_x_nonloc[atom_i]);
        computeFEMNL(femY,
                     femNonLocY,
                     eigY,
                     nonlocal_map_y,
                     basis_y_nonloc[atom_i]);
        computeFEMNL(femZ,
                     femNonLocZ,
                     eigZ,
                     nonlocal_map_z,
                     basis_z_nonloc[atom_i]);
    }

    PetscInt num_global_cols = rankTuckerBasisX * rankTuckerBasisY * rankTuckerBasisZ;
    PetscInt dnz = num_local_rows, onz = num_global_cols * 5e-3;
    MatCreateAIJ(PETSC_COMM_WORLD,
                 num_local_rows,
                 PETSC_DECIDE,
                 num_global_rows,
                 num_global_cols,
                 dnz,
                 PETSC_NULL,
                 onz,
                 PETSC_NULL,
                 &C_nloc);
    MatSetOption(C_nloc,
                 MAT_NEW_NONZERO_ALLOCATION_ERR,
                 PETSC_FALSE);

    std::vector<double> inv_C_lm_data(num_local_rows,
                                      0.0);
    std::vector<PetscInt> inv_C_lm_idx(num_local_rows,
                                       0);

    for (int i = 0; i < num_local_rows; ++i) {
        const NonLocRowInfo &atom_id = owned_rows_info[i];
        int local_atom_i = atom_id.atom_id_local;
        int local_atom_type_i = atom_id.atom_type_local;
        int lm_i = atom_id.lm_id;

        inv_C_lm_idx[i] = atom_id.global_row_idx;
        inv_C_lm_data[i] = 1.0 / (nonLocalPSPData[atom_id.atom_type]->getC_lm()[atom_id.lm_id]);

        int rank_nonloc = nonLocalPSPData[atom_id.atom_type]->getRankNloc();
        int nonloc_compact_support_size_x = femNonLocX.getTotalNumberQuadPoints();
        int nonloc_compact_support_size_y = femNonLocY.getTotalNumberQuadPoints();
        int nonloc_compact_support_size_z = femNonLocZ.getTotalNumberQuadPoints();
        Tucker::Matrix *mat_x = Tucker::MemoryManager::safe_new<Tucker::Matrix>(rankTuckerBasisX,
                                                                                rank_nonloc);
        Tucker::Matrix *mat_y = Tucker::MemoryManager::safe_new<Tucker::Matrix>(rankTuckerBasisY,
                                                                                rank_nonloc);
        Tucker::Matrix *mat_z = Tucker::MemoryManager::safe_new<Tucker::Matrix>(rankTuckerBasisZ,
                                                                                rank_nonloc);

        computeOverlapNonLocPotentialTuckerBasis(nonloc_compact_support_size_x,
                                                 basis_x_nonloc[local_atom_i],
                                                 umatNlocTimesJacobTimesWeight[local_atom_type_i][lm_i],
                                                 mat_x);
        computeOverlapNonLocPotentialTuckerBasis(nonloc_compact_support_size_y,
                                                 basis_y_nonloc[local_atom_i],
                                                 vmatNlocTimesJacobTimesWeight[local_atom_type_i][lm_i],
                                                 mat_y);
        computeOverlapNonLocPotentialTuckerBasis(nonloc_compact_support_size_z,
                                                 basis_z_nonloc[local_atom_i],
                                                 wmatNlocTimesJacobTimesWeight[local_atom_type_i][lm_i],
                                                 mat_z);

        Tucker::Tensor *temp = nonLocalPSPData[atom_id.atom_type]->getSigmaQuad()[atom_id.lm_id];
        Tucker::Tensor *reconstructedTensor;
        reconstructedTensor = Tucker::ttm(temp,
                                          0,
                                          mat_x);
        temp = reconstructedTensor;
        reconstructedTensor = Tucker::ttm(temp,
                                          1,
                                          mat_y);
        Tucker::MemoryManager::safe_delete(temp);
        temp = reconstructedTensor;
        reconstructedTensor = Tucker::ttm(temp,
                                          2,
                                          mat_z);
        Tucker::MemoryManager::safe_delete(temp);

        std::vector<PetscInt> nloc_idx;
        std::vector<double> nloc_val;
        double *nloc_data = reconstructedTensor->data();
        for (int i = 0; i < num_global_cols; ++i) {
            if (std::abs(nloc_data[i]) > truncation) {
                nloc_idx.push_back(i);
                nloc_val.push_back(nloc_data[i]);
            }
        }
        Tucker::MemoryManager::safe_delete(reconstructedTensor);
        MatSetValues(C_nloc,
                     1,
                     &atom_id.global_row_idx,
                     nloc_idx.size(),
                     nloc_idx.data(),
                     nloc_val.data(),
                     INSERT_VALUES);

        Tucker::MemoryManager::safe_delete(mat_x);
        Tucker::MemoryManager::safe_delete(mat_y);
        Tucker::MemoryManager::safe_delete(mat_z);
    }

    MatAssemblyBegin(C_nloc,
                     MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(C_nloc,
                   MAT_FINAL_ASSEMBLY);

//  MatInfo info_max, info_sum;
//  MatGetInfo(C_nloc, MAT_GLOBAL_MAX, &info_max);
//  MatGetInfo(C_nloc, MAT_GLOBAL_SUM, &info_sum);
//  PetscPrintf(PETSC_COMM_WORLD, "Create_Cnloc::max nnz allocated across all processors: %.4e\n", info_max.nz_allocated);
//  PetscPrintf(PETSC_COMM_WORLD, "Create_Cnloc::max nnz used of all processors: %.4e\n", info_max.nz_used);
//  PetscPrintf(PETSC_COMM_WORLD, "Create_Cnloc::max memory used by Cnloc: %.4e mb\n", info_max.memory / 1024 / 1024);
//  PetscPrintf(PETSC_COMM_WORLD,
//              "Create_Cnloc::sum nnz ratio for truncation %.4e: %.4e\n",
//              truncation,
//              1.0 * info_sum.nz_used / num_global_cols / num_global_rows);

    Vec inv_C_lm_vec;
    VecCreate(PETSC_COMM_WORLD,
              &inv_C_lm_vec);
    VecSetSizes(inv_C_lm_vec,
                num_local_rows,
                PETSC_DECIDE);
    VecSetUp(inv_C_lm_vec);
    VecSetValues(inv_C_lm_vec,
                 inv_C_lm_idx.size(),
                 inv_C_lm_idx.data(),
                 inv_C_lm_data.data(),
                 INSERT_VALUES);
    VecAssemblyBegin(inv_C_lm_vec);
    VecAssemblyEnd(inv_C_lm_vec);

//  MPI_Barrier(PETSC_COMM_WORLD);
//  double time_trans = MPI_Wtime();
    MatTranspose(C_nloc,
                 MAT_INITIAL_MATRIX,
                 &C_nloc_trans);
//  MPI_Barrier(PETSC_COMM_WORLD);
//  time_trans = MPI_Wtime() - time_trans;
//  PetscPrintf(MPI_COMM_WORLD, "Create_Cnloc::time transposing C_nloc matrix: %.6e\n", time_trans);

    MatDiagonalScale(C_nloc,
                     inv_C_lm_vec,
                     PETSC_NULL);

    VecDestroy(&inv_C_lm_vec);
}

void ProjectHamiltonianSparse::Create_Hloc(const int rankVeffX,
                                           const int rankVeffY,
                                           const int rankVeffZ,
                                           const int rankTuckerBasisX,
                                           const int rankTuckerBasisY,
                                           const int rankTuckerBasisZ,
                                           const TuckerMPI::TuckerTensor *tuckerDecomposedVeff,
                                           double truncation) {
    int taskid, nproc;
    MPI_Comm_rank(PETSC_COMM_WORLD,
                  &taskid);
    MPI_Comm_size(PETSC_COMM_WORLD,
                  &nproc);
//  PetscPrintf(PETSC_COMM_WORLD, "Create_Hloc::truncation %.4e\n", truncation);

    if (ProjectHamiltonianSparse::Hloc_dnz.empty()) {
        Compute_nnz_loc(tuckerDecomposedVeff,
                        rankVeffX,
                        rankVeffY,
                        rankVeffZ,
                        rankTuckerBasisX,
                        rankTuckerBasisY,
                        rankTuckerBasisZ,
                        truncation,
                        ProjectHamiltonianSparse::Hloc_dnz,
                        ProjectHamiltonianSparse::Hloc_onz);
    }

    std::vector<std::vector<double> > MatX(rankTuckerBasisX,
                                           std::vector<double>(rankTuckerBasisX));
    std::vector<std::vector<double> > MatY(rankTuckerBasisY,
                                           std::vector<double>(rankTuckerBasisY));
    std::vector<std::vector<double> > MatZ(rankTuckerBasisZ,
                                           std::vector<double>(rankTuckerBasisZ));
    std::vector<std::vector<double> > MatDX(rankTuckerBasisX,
                                            std::vector<double>(rankTuckerBasisX));
    std::vector<std::vector<double> > MatDY(rankTuckerBasisY,
                                            std::vector<double>(rankTuckerBasisY));
    std::vector<std::vector<double> > MatDZ(rankTuckerBasisZ,
                                            std::vector<double>(rankTuckerBasisZ));
    std::vector<std::vector<std::vector<double> > >
            MatPotX(rankVeffX,
                    std::vector<std::vector<double> >(rankTuckerBasisX,
                                                      std::vector<double>(rankTuckerBasisX)));
    std::vector<std::vector<std::vector<double> > >
            MatPotY(rankVeffY,
                    std::vector<std::vector<double> >(rankTuckerBasisY,
                                                      std::vector<double>(rankTuckerBasisY)));
    std::vector<std::vector<std::vector<double> > >
            MatPotZ(rankVeffZ,
                    std::vector<std::vector<double> >(rankTuckerBasisZ,
                                                      std::vector<double>(rankTuckerBasisZ)));

    computeOverlapKineticTuckerBasis(femX,
                                     femY,
                                     femZ,
                                     rankTuckerBasisX,
                                     rankTuckerBasisY,
                                     rankTuckerBasisZ,
                                     basisXQuadValues,
                                     basisDXQuadValues,
                                     basisYQuadValues,
                                     basisDYQuadValues,
                                     basisZQuadValues,
                                     basisDZQuadValues,
                                     MatX,
                                     MatY,
                                     MatZ,
                                     MatDX,
                                     MatDY,
                                     MatDZ);

    computeOverlapPotentialTuckerBasis(femX,
                                       femY,
                                       femZ,
                                       tuckerDecomposedVeff,
                                       basisXQuadValues,
                                       basisYQuadValues,
                                       basisZQuadValues,
                                       rankVeffX,
                                       rankVeffY,
                                       rankVeffZ,
                                       rankTuckerBasisX,
                                       rankTuckerBasisY,
                                       rankTuckerBasisZ,
                                       MatPotX,
                                       MatPotY,
                                       MatPotZ);

    // create map from tensor indices ijk to bases index I
    std::vector<std::vector<int>>
            matToTucker(3,
                        std::vector<int>(rankTuckerBasisX * rankTuckerBasisY * rankTuckerBasisZ));
    int cnt = 0;
    for (int zI = 0; zI < rankTuckerBasisZ; ++zI) {
        for (int yI = 0; yI < rankTuckerBasisY; ++yI) {
            for (int xI = 0; xI < rankTuckerBasisX; ++xI) {
                matToTucker[0][cnt] = xI;
                matToTucker[1][cnt] = yI;
                matToTucker[2][cnt] = zI;
                cnt++;
            }
        }
    }

    Tucker::SizeArray coreVeffDim(3);
    coreVeffDim[0] = rankVeffX;
    coreVeffDim[1] = rankVeffY;
    coreVeffDim[2] = rankVeffZ;
    Tucker::Tensor *coreVeffSeq = Tucker::MemoryManager::safe_new<Tucker::Tensor>(coreVeffDim);
    TensorUtils::allreduce_tensor(tuckerDecomposedVeff->G,
                                  coreVeffSeq);


    // initialize H_loc
    int numLocalRows = Hloc_dnz.size();
    PetscInt basis_dimension = rankTuckerBasisX * rankTuckerBasisY * rankTuckerBasisZ;
    MatCreateAIJ(PETSC_COMM_WORLD,
                 numLocalRows,
                 numLocalRows,
                 basis_dimension,
                 basis_dimension,
                 PETSC_NULL,
                 Hloc_dnz.data(),
                 PETSC_NULL,
                 Hloc_onz.data(),
                 &H_loc);
    PetscInt Hloc_istart, Hloc_iend;
    MatGetOwnershipRange(H_loc,
                         &Hloc_istart,
                         &Hloc_iend);


    // construct H_loc
    for (int irow = 0; irow < numLocalRows; ++irow) {
        PetscInt row = irow + Hloc_istart;
        int xI = matToTucker[0][row];
        int yI = matToTucker[1][row];
        int zI = matToTucker[2][row];

        std::vector<double> &localMatX = MatX[xI];
        std::vector<double> &localMatY = MatY[yI];
        std::vector<double> &localMatZ = MatZ[zI];
        std::vector<double> &localMatDX = MatDX[xI];
        std::vector<double> &localMatDY = MatDY[yI];
        std::vector<double> &localMatDZ = MatDZ[zI];

        // compute falttented ((a x b) x c), x stands for tensor product
        std::vector<double> dyad(rankTuckerBasisX * rankTuckerBasisY * rankTuckerBasisZ,
                                 0.0);
        int cnt = 0;
        for (int zrank = 0; zrank < rankTuckerBasisZ; ++zrank) {
            for (int yrank = 0; yrank < rankTuckerBasisY; ++yrank) {
                for (int xrank = 0; xrank < rankTuckerBasisX; ++xrank) {
                    dyad[cnt] = localMatDX[xrank] * localMatY[yrank] * localMatZ[zrank]
                                + localMatX[xrank] * localMatDY[yrank] * localMatZ[zrank]
                                + localMatX[xrank] * localMatY[yrank] * localMatDZ[zrank];
                    dyad[cnt] = 0.5 * dyad[cnt];
                    cnt++;
                }
            }
        }// end of x,y,zrank
        Tucker::Matrix *MatPotXLocal = Tucker::MemoryManager::safe_new<Tucker::Matrix>(rankTuckerBasisX,
                                                                                       rankVeffX);
        Tucker::Matrix *MatPotYLocal = Tucker::MemoryManager::safe_new<Tucker::Matrix>(rankTuckerBasisY,
                                                                                       rankVeffY);
        Tucker::Matrix *MatPotZLocal = Tucker::MemoryManager::safe_new<Tucker::Matrix>(rankTuckerBasisZ,
                                                                                       rankVeffZ);
        for (int i = 0; i < rankVeffX; ++i) {
            std::copy(MatPotX[i][xI].begin(),
                      MatPotX[i][xI].end(),
                      MatPotXLocal->data() + i * rankTuckerBasisX);
        }
        for (int i = 0; i < rankVeffY; ++i) {
            std::copy(MatPotY[i][yI].begin(),
                      MatPotY[i][yI].end(),
                      MatPotYLocal->data() + i * rankTuckerBasisY);
        }
        for (int i = 0; i < rankVeffZ; ++i) {
            std::copy(MatPotZ[i][zI].begin(),
                      MatPotZ[i][zI].end(),
                      MatPotZLocal->data() + i * rankTuckerBasisZ);
        }
        // recontruct tensor, reconstructedTensor will be the final constructed tensor
        Tucker::Tensor *temp;
        Tucker::Tensor *reconstructedTensor;
        temp = coreVeffSeq;
        reconstructedTensor = Tucker::ttm(temp,
                                          0,
                                          MatPotXLocal);
        temp = reconstructedTensor;
        reconstructedTensor = Tucker::ttm(temp,
                                          1,
                                          MatPotYLocal);
        Tucker::MemoryManager::safe_delete(temp);
        temp = reconstructedTensor;
        reconstructedTensor = Tucker::ttm(temp,
                                          2,
                                          MatPotZLocal);
        Tucker::MemoryManager::safe_delete(temp);

        int numTensorElements = reconstructedTensor->getNumElements();
        double *reconstructedTensorData = reconstructedTensor->data();
        for (int i = 0; i < numTensorElements; ++i) {
            dyad[i] += reconstructedTensorData[i];
        }
        Tucker::MemoryManager::safe_delete(reconstructedTensor);
        Tucker::MemoryManager::safe_delete(MatPotXLocal);
        Tucker::MemoryManager::safe_delete(MatPotYLocal);
        Tucker::MemoryManager::safe_delete(MatPotZLocal);

        std::vector<PetscInt> col_idx;
        std::vector<double> val;
        for (int j = 0; j < dyad.size(); ++j) {
            if (std::abs(dyad[j]) > truncation) {
                val.emplace_back(dyad[j]);
                col_idx.emplace_back(j);
            }
        }
        MatSetValues(H_loc,
                     1,
                     &row,
                     col_idx.size(),
                     &col_idx[0],
                     &val[0],
                     INSERT_VALUES);
    }// end of row

    MatAssemblyBegin(H_loc,
                     MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(H_loc,
                   MAT_FINAL_ASSEMBLY);

//  MatInfo info_max, info_sum;
//  MatGetInfo(H_loc, MAT_GLOBAL_MAX, &info_max);
//  MatGetInfo(H_loc, MAT_GLOBAL_SUM, &info_sum);
//  PetscPrintf(PETSC_COMM_WORLD, "Create_Hloc::max nnz allocated across all processors: %.4e\n", info_max.nz_allocated);
//  PetscPrintf(PETSC_COMM_WORLD, "Create_Hloc::max nnz used of all processors: %.4e\n", info_max.nz_used);
//  PetscPrintf(PETSC_COMM_WORLD, "Create_Hloc::max memory used by H_loc: %.4e mb\n", info_max.memory / 1024 / 1024);
//  PetscPrintf(PETSC_COMM_WORLD,
//              "Create_Hloc::sum nnz ratio for truncation %.4e: %.4e\n",
//              truncation,
//              1.0 * info_sum.nz_used / basis_dimension / basis_dimension);

    Tucker::MemoryManager::safe_delete(coreVeffSeq);
}

void ProjectHamiltonianSparse::computeOverlapKineticTuckerBasis(const FEM &femX,
                                                                const FEM &femY,
                                                                const FEM &femZ,
                                                                int rankTuckerBasisX,
                                                                int rankTuckerBasisY,
                                                                int rankTuckerBasisZ,
                                                                const std::vector<std::vector<double>> &basisXQuadValues,
                                                                const std::vector<std::vector<double>> &basisDXQuadValues,
                                                                const std::vector<std::vector<double>> &basisYQuadValues,
                                                                const std::vector<std::vector<double>> &basisDYQuadValues,
                                                                const std::vector<std::vector<double>> &basisZQuadValues,
                                                                const std::vector<std::vector<double>> &basisDZQuadValues,
                                                                std::vector<std::vector<double> > &MatX,
                                                                std::vector<std::vector<double> > &MatY,
                                                                std::vector<std::vector<double> > &MatZ,
                                                                std::vector<std::vector<double> > &MatDX,
                                                                std::vector<std::vector<double> > &MatDY,
                                                                std::vector<std::vector<double> > &MatDZ) {
    for (auto I = 0; I < rankTuckerBasisX; ++I) {
        //here
        const std::vector<double> &fieldxI = basisXQuadValues[I];
        const std::vector<double> &fieldDxI = basisDXQuadValues[I];
        for (auto J = 0; J < rankTuckerBasisX; ++J) {
            if (I <= J) {
                const std::vector<double> &fieldxJ = basisXQuadValues[J];
                const std::vector<double> &fieldDxJ = basisDXQuadValues[J];
                std::vector<double> overlapTemp(basisXQuadValues[I].size());
                std::transform(fieldxI.begin(),
                               fieldxI.end(),
                               fieldxJ.begin(),
                               overlapTemp.begin(),
                               std::multiplies<double>());
                MatX[I][J] = femX.integrate_by_quad_values(overlapTemp);
                std::transform(fieldDxI.begin(),
                               fieldDxI.end(),
                               fieldDxJ.begin(),
                               overlapTemp.begin(),
                               std::multiplies<double>());
                MatDX[I][J] = femX.integrate_inv_by_quad_values(overlapTemp);
            } else {
                MatX[I][J] = MatX[J][I];
                MatDX[I][J] = MatDX[J][I];
            }
        }
    }

    for (auto I = 0; I < rankTuckerBasisY; ++I) {
        const std::vector<double> &fieldyI = basisYQuadValues[I];
        const std::vector<double> &fieldDyI = basisDYQuadValues[I];
        for (auto J = 0; J < rankTuckerBasisY; ++J) {
            if (I <= J) {
                const std::vector<double> &fieldyJ = basisYQuadValues[J];
                const std::vector<double> &fieldDyJ = basisDYQuadValues[J];
                std::vector<double> overlapTemp(basisYQuadValues[I].size());
                std::transform(fieldyI.begin(),
                               fieldyI.end(),
                               fieldyJ.begin(),
                               overlapTemp.begin(),
                               std::multiplies<double>());
                MatY[I][J] = femY.integrate_by_quad_values(overlapTemp);
                std::transform(fieldDyI.begin(),
                               fieldDyI.end(),
                               fieldDyJ.begin(),
                               overlapTemp.begin(),
                               std::multiplies<double>());
                MatDY[I][J] = femY.integrate_inv_by_quad_values(overlapTemp);
            } else {
                MatY[I][J] = MatY[J][I];
                MatDY[I][J] = MatDY[J][I];
            }
        }
    }

    for (auto I = 0; I < rankTuckerBasisZ; ++I) {
        const std::vector<double> &fieldzI = basisZQuadValues[I];
        const std::vector<double> &fieldDzI = basisDZQuadValues[I];
        for (auto J = 0; J < rankTuckerBasisZ; ++J) {
            if (I <= J) {
                const std::vector<double> &fieldzJ = basisZQuadValues[J];
                const std::vector<double> &fieldDzJ = basisDZQuadValues[J];
                std::vector<double> overlapTemp(basisZQuadValues[I].size());
                std::transform(fieldzI.begin(),
                               fieldzI.end(),
                               fieldzJ.begin(),
                               overlapTemp.begin(),
                               std::multiplies<double>());
                MatZ[I][J] = femZ.integrate_by_quad_values(overlapTemp);
                std::transform(fieldDzI.begin(),
                               fieldDzI.end(),
                               fieldDzJ.begin(),
                               overlapTemp.begin(),
                               std::multiplies<double>());
                MatDZ[I][J] = femZ.integrate_inv_by_quad_values(overlapTemp);
            } else {
                MatZ[I][J] = MatZ[J][I];
                MatDZ[I][J] = MatDZ[J][I];
            }
        }
    }
}

void ProjectHamiltonianSparse::computeOverlapPotentialTuckerBasis(const FEM &femX,
                                                                  const FEM &femY,
                                                                  const FEM &femZ,
                                                                  const TuckerMPI::TuckerTensor *tuckerDecomposedVeff,
                                                                  const std::vector<std::vector<double>> &basisXQuadValues,
                                                                  const std::vector<std::vector<double>> &basisYQuadValues,
                                                                  const std::vector<std::vector<double>> &basisZQuadValues,
                                                                  int rankVeffX,
                                                                  int rankVeffY,
                                                                  int rankVeffZ,
                                                                  int rankTuckerBasisX,
                                                                  int rankTuckerBasisY,
                                                                  int rankTuckerBasisZ,
                                                                  std::vector<std::vector<std::vector<double> > > &MatPotX,
                                                                  std::vector<std::vector<std::vector<double> > > &MatPotY,
                                                                  std::vector<std::vector<std::vector<double> > > &MatPotZ) {
    double *umat = &(tuckerDecomposedVeff->U[0]->data()[0]);
    int umatLength = tuckerDecomposedVeff->U[0]->nrows();
    for (auto irank = 0; irank < rankVeffX; ++irank, umat = umat + umatLength) {
        for (auto I = 0; I < rankTuckerBasisX; ++I) {
            const std::vector<double> &fieldxI = basisXQuadValues[I];
            for (auto J = 0; J < rankTuckerBasisX; ++J) {
                if (I <= J) {
                    const std::vector<double> &fieldxJ = basisXQuadValues[J];
                    std::vector<double> overlapTemp(basisXQuadValues[I].size());
                    std::transform(fieldxI.begin(),
                                   fieldxI.end(),
                                   fieldxJ.begin(),
                                   overlapTemp.begin(),
                                   std::multiplies<double>());
                    std::transform(umat,
                                   umat + umatLength,
                                   overlapTemp.begin(),
                                   overlapTemp.begin(),
                                   std::multiplies<double>());
                    MatPotX[irank][I][J] = femX.integrate_by_quad_values(overlapTemp);
                } else {
                    MatPotX[irank][I][J] = MatPotX[irank][J][I];
                }
            }
        }
    }

    double *vmat = &(tuckerDecomposedVeff->U[1]->data()[0]);
    int vmatLength = tuckerDecomposedVeff->U[1]->nrows();
    for (auto irank = 0; irank < rankVeffY; ++irank, vmat = vmat + vmatLength) {
        for (auto I = 0; I < rankTuckerBasisY; ++I) {
            const std::vector<double> &fieldxI = basisYQuadValues[I];
            for (auto J = 0; J < rankTuckerBasisY; ++J) {
                if (I <= J) {
                    const std::vector<double> &fieldxJ = basisYQuadValues[J];
                    std::vector<double> overlapTemp(basisYQuadValues[I].size());
                    std::transform(fieldxI.begin(),
                                   fieldxI.end(),
                                   fieldxJ.begin(),
                                   overlapTemp.begin(),
                                   std::multiplies<double>());
                    std::transform(vmat,
                                   vmat + vmatLength,
                                   overlapTemp.begin(),
                                   overlapTemp.begin(),
                                   std::multiplies<double>());
                    MatPotY[irank][I][J] = femY.integrate_by_quad_values(overlapTemp);
                } else {
                    MatPotY[irank][I][J] = MatPotY[irank][J][I];
                }
            }
        }
    }

    double *wmat = &(tuckerDecomposedVeff->U[2]->data()[0]);
    int wmatLength = tuckerDecomposedVeff->U[2]->nrows();
    for (auto irank = 0; irank < rankVeffZ; ++irank, wmat = wmat + wmatLength) {
        for (auto I = 0; I < rankTuckerBasisZ; ++I) {
            const std::vector<double> &fieldxI = basisZQuadValues[I];
            for (auto J = 0; J < rankTuckerBasisZ; ++J) {
                if (I <= J) {
                    const std::vector<double> &fieldxJ = basisZQuadValues[J];
                    std::vector<double> overlapTemp(basisZQuadValues[I].size());
                    std::transform(fieldxI.begin(),
                                   fieldxI.end(),
                                   fieldxJ.begin(),
                                   overlapTemp.begin(),
                                   std::multiplies<double>());
                    std::transform(wmat,
                                   wmat + wmatLength,
                                   overlapTemp.begin(),
                                   overlapTemp.begin(),
                                   std::multiplies<double>());
                    MatPotZ[irank][I][J] = femZ.integrate_by_quad_values(overlapTemp);
                } else {
                    MatPotZ[irank][I][J] = MatPotZ[irank][J][I];
                }
            }
        }
    }

}

void ProjectHamiltonianSparse::Destroy() {
    MatDestroy(&H_loc);
}

void ProjectHamiltonianSparse::computeFactorsProduct(const FEM &fem,
                                                     const FEM &femNonLoc,
                                                     const std::vector<std::vector<double>> &basis,
                                                     const int rankNloc,
                                                     const int lMax,
                                                     const std::vector<Tucker::Matrix *> &factor_nloc,
                                                     std::vector<std::vector<std::vector<double> > > &factor_product) {
    int lmCount = lMax * lMax;
    auto &weightQuadPointValues = femNonLoc.getWeightQuadPointValues();
    auto &jacobianQuadPointValues = femNonLoc.getJacobQuadPointValues();
    int nonLocalCompactSupport = femNonLoc.getTotalNumberQuadPoints();
    auto &elementConnectivity = fem.getElementConnectivity();

    factor_product = std::vector<std::vector<std::vector<double> > >
            (lmCount,
             std::vector<std::vector<double> >(rankNloc,
                                               std::vector<double>(nonLocalCompactSupport,
                                                                   0.0)));

    for (int lmcomp = 0; lmcomp < lmCount; ++lmcomp) {
        for (int irank = 0; irank < rankNloc; ++irank) {
            double *factor_nloc_data = factor_nloc[lmcomp]->data() + irank * nonLocalCompactSupport;
            for (int iquad = 0; iquad < nonLocalCompactSupport; ++iquad) {
                factor_product[lmcomp][irank][iquad] =
                        factor_nloc_data[iquad] * jacobianQuadPointValues[iquad] * weightQuadPointValues[iquad];
            }
        }
    }
}

void ProjectHamiltonianSparse::computeFEMNL(const FEM &fem,
                                            const FEM &femNonLoc,
                                            const std::vector<std::vector<double>> &basis,
                                            const NonLocalMap::NonLocalMap1D &nonloc_map1d,
                                            std::vector<std::vector<double> > &basis_nonloc) {
    int rankTuckerBasis = basis.size();
    auto &elemNonLocGridToFullGrid = nonloc_map1d.elemNonLocGridToFullGrid;
    auto &shapeFunctionMatrixFullGrid = nonloc_map1d.shapeFunctionMatrixFullGrid;
    int nonLocalCompactSupport = femNonLoc.getTotalNumberQuadPoints();
    auto &elementConnectivity = fem.getElementConnectivity();

    // compute the field at given points
    basis_nonloc = std::vector<std::vector<double> >(rankTuckerBasis,
                                                     std::vector<double>(nonLocalCompactSupport,
                                                                         0.0));

    int inc = 1;
    for (int iPoint = 0; iPoint < nonLocalCompactSupport; ++iPoint) {
        for (int irank = 0; irank < rankTuckerBasis; ++irank) {
            int elementId = elemNonLocGridToFullGrid[iPoint];
            auto &localNodeIds = elementConnectivity[elementId];
            int n = localNodeIds.size();
            std::vector<double> localNodalValues(n,
                                                 0.0);
            for (int i = 0; i < n; ++i) {
                localNodalValues[i] = basis[irank][localNodeIds[i]];
            }
            basis_nonloc[irank][iPoint] =
                    ddot_(&n,
                          localNodalValues.data(),
                          &inc,
                          shapeFunctionMatrixFullGrid[iPoint].data(),
                          &inc);
        }
    }
}

void ProjectHamiltonianSparse::computeOverlapNonLocPotentialTuckerBasis(int nonloc_compact_support_size,
                                                                        std::vector<std::vector<double>> &basis_nonloc,
                                                                        std::vector<std::vector<double>> &matNlocTimesJacobTimesWeight,
                                                                        Tucker::Matrix *&mat_nonloc) {
    int rank_tucker = basis_nonloc.size();
    int rank_nonloc = matNlocTimesJacobTimesWeight.size();
    int inc = 1;

    mat_nonloc->initialize();
    double *mat_nonloc_data = mat_nonloc->data();
    int cnt = 0;
    for (int irank = 0; irank < rank_nonloc; ++irank) {
        for (int iTucker = 0; iTucker < rank_tucker; ++iTucker) {
            mat_nonloc_data[cnt++] = ddot_(&nonloc_compact_support_size,
                                           basis_nonloc[iTucker].data(),
                                           &inc,
                                           matNlocTimesJacobTimesWeight[irank].data(),
                                           &inc);
        }
    }
}

void computeJacobWeight3DMat(const FEM &femX,
                             const FEM &femY,
                             const FEM &femZ,
                             Tensor3DMPI &jacob3DMat,
                             Tensor3DMPI &weight3DMat) {
    jacob3DMat.setEntriesZero();
    weight3DMat.setEntriesZero();
    const std::vector<double> &jacobX = femX.getJacobQuadPointValues();
    const std::vector<double> &jacobY = femY.getJacobQuadPointValues();
    const std::vector<double> &jacobZ = femZ.getJacobQuadPointValues();
    const std::vector<double> &weightX = femX.getWeightQuadPointValues();
    const std::vector<double> &weightY = femY.getWeightQuadPointValues();
    const std::vector<double> &weightZ = femZ.getWeightQuadPointValues();

    std::array<int, 6> jacob3DMatGlobalIdx;
    std::array<int, 6> weight3DMatGlobalIdx;
    double *jacob3DMatLocal = jacob3DMat.getLocalData(jacob3DMatGlobalIdx);
    double *weight3DMatLocal = weight3DMat.getLocalData(weight3DMatGlobalIdx);

    int cnt = 0;
    for (int k = jacob3DMatGlobalIdx[4]; k < jacob3DMatGlobalIdx[5]; ++k) {
        for (int j = jacob3DMatGlobalIdx[2]; j < jacob3DMatGlobalIdx[3]; ++j) {
            for (int i = jacob3DMatGlobalIdx[0]; i < jacob3DMatGlobalIdx[1]; ++i) {
                jacob3DMatLocal[cnt] = jacobX[i] * jacobY[j] * jacobZ[k];
                cnt = cnt + 1;
            }
        }
    }

    cnt = 0;
    for (int k = weight3DMatGlobalIdx[4]; k < weight3DMatGlobalIdx[5]; ++k) {
        for (int j = weight3DMatGlobalIdx[2]; j < weight3DMatGlobalIdx[3]; ++j) {
            for (int i = weight3DMatGlobalIdx[0]; i < weight3DMatGlobalIdx[1]; ++i) {
                weight3DMatLocal[cnt] = weightX[i] * weightY[j] * weightZ[k];
                cnt = cnt + 1;
            }
        }
    }

}

void ProjectHamiltonianSparse::computeRhoOut(Mat &eigenVectorTucker,
                                             const std::vector<double> &occupancyFactor,
                                             Tensor3DMPI &rhoNodalOut,
                                             Tensor3DMPI &rhoGridOut) {
    int rankTuckerBasisX = eigX.size();
    int rankTuckerBasisY = eigY.size();
    int rankTuckerBasisZ = eigZ.size();
    int numEigenvalues = occupancyFactor.size();

    rhoNodalOut.setEntriesZero();
    rhoGridOut.setEntriesZero();

    int numNodalLocalEntries = rhoNodalOut.getLocalNumberEntries();

    int istartNodal = rhoNodalOut.getIstartGlobal(), iendNodal = rhoNodalOut.getIendGlobal();
    int jstartNodal = rhoNodalOut.getJstartGlobal(), jendNodal = rhoNodalOut.getJendGlobal();
    int kstartNodal = rhoNodalOut.getKstartGlobal(), kendNodal = rhoNodalOut.getKendGlobal();

    int nrowsMatEigX = iendNodal - istartNodal, ncolsMatEigX = eigX.size();
    int nrowsMatEigY = jendNodal - jstartNodal, ncolsMatEigY = eigY.size();
    int nrowsMatEigZ = kendNodal - kstartNodal, ncolsMatEigZ = eigZ.size();
    Tucker::Matrix *MatEigX = Tucker::MemoryManager::safe_new<Tucker::Matrix>(nrowsMatEigX,
                                                                              ncolsMatEigX);
    Tucker::Matrix *MatEigY = Tucker::MemoryManager::safe_new<Tucker::Matrix>(nrowsMatEigY,
                                                                              ncolsMatEigY);
    Tucker::Matrix *MatEigZ = Tucker::MemoryManager::safe_new<Tucker::Matrix>(nrowsMatEigZ,
                                                                              ncolsMatEigZ);
    if (numNodalLocalEntries != 0) {
        for (int i = 0; i < ncolsMatEigX; ++i) {
            std::copy(eigX[i].begin() + istartNodal,
                      eigX[i].begin() + iendNodal,
                      MatEigX->data() + i * nrowsMatEigX);
        }
        for (int i = 0; i < ncolsMatEigY; ++i) {
            std::copy(eigY[i].begin() + jstartNodal,
                      eigY[i].begin() + jendNodal,
                      MatEigY->data() + i * nrowsMatEigY);
        }
        for (int i = 0; i < ncolsMatEigZ; ++i) {
            std::copy(eigZ[i].begin() + kstartNodal,
                      eigZ[i].begin() + kendNodal,
                      MatEigZ->data() + i * nrowsMatEigZ);
        }
    }

//  int numGridLocalEntries = rhoGridOut.getLocalNumberEntries();
//  int istartGrid = rhoGridOut.getIstartGlobal(), iendGrid = rhoGridOut.getIendGlobal();
//  int jstartGrid = rhoGridOut.getJstartGlobal(), jendGrid = rhoGridOut.getJendGlobal();
//  int kstartGrid = rhoGridOut.getKstartGlobal(), kendGrid = rhoGridOut.getKendGlobal();
//  int nrowsMatEigQuadX = iendGrid - istartGrid, ncolsMatEigQuadX = basisXQuadValues.size();
//  int nrowsMatEigQuadY = jendGrid - jstartGrid, ncolsMatEigQuadY = basisYQuadValues.size();
//  int nrowsMatEigQuadZ = kendGrid - kstartGrid, ncolsMatEigQuadZ = basisZQuadValues.size();
//  Tucker::Matrix *MatEigQuadX = Tucker::MemoryManager::safe_new<Tucker::Matrix>(nrowsMatEigQuadX,
//                                                                                ncolsMatEigQuadX);
//  Tucker::Matrix *MatEigQuadY = Tucker::MemoryManager::safe_new<Tucker::Matrix>(nrowsMatEigQuadY,
//                                                                                ncolsMatEigQuadY);
//  Tucker::Matrix *MatEigQuadZ = Tucker::MemoryManager::safe_new<Tucker::Matrix>(nrowsMatEigQuadZ,
//                                                                                ncolsMatEigQuadZ);
//
//  if (numGridLocalEntries != 0) {
//    for (int i = 0; i < ncolsMatEigQuadX; ++i) {
//      std::copy(basisXQuadValues[i].begin() + istartGrid, basisXQuadValues[i].begin() + iendGrid,
//                MatEigQuadX->data() + i * nrowsMatEigQuadX);
//    }
//    for (int i = 0; i < ncolsMatEigQuadY; ++i) {
//      std::copy(basisYQuadValues[i].begin() + jstartGrid, basisYQuadValues[i].begin() + jendGrid,
//                MatEigQuadY->data() + i * nrowsMatEigQuadY);
//    }
//    for (int i = 0; i < ncolsMatEigQuadZ; ++i) {
//      std::copy(basisZQuadValues[i].begin() + kstartGrid, basisZQuadValues[i].begin() + kendGrid,
//                MatEigQuadZ->data() + i * nrowsMatEigQuadZ);
//    }
//  }

    double *eigenVectorTucker_seq_data;
    int numGlobalRows = rankTuckerBasisX * rankTuckerBasisY * rankTuckerBasisZ;

    Mat eigenVectorTucker_seq;
    MatDenseGetLocalMatrix(eigenVectorTucker,
                           &eigenVectorTucker_seq);
    MatDenseGetArray(eigenVectorTucker_seq,
                     &eigenVectorTucker_seq_data);

    int nprocs, taskid;
    PetscInt Istart, Iend;
    MPI_Comm_size(PETSC_COMM_WORLD,
                  &nprocs);
    MPI_Comm_rank(PETSC_COMM_WORLD,
                  &taskid);
    MatGetOwnershipRange(eigenVectorTucker,
                         &Istart,
                         &Iend);
    int number_local_owned_rows = Iend - Istart;
    std::vector<int> recvcounts(nprocs,
                                0), displs(nprocs + 1,
                                           0);
    MPI_Allgather(&number_local_owned_rows,
                  1,
                  MPI_INT,
                  recvcounts.data(),
                  1,
                  MPI_INT,
                  PETSC_COMM_WORLD);
    for (int k = 0; k < nprocs; ++k) {
        displs[k + 1] = displs[k] + recvcounts[k];
    }

    double time0 = 0, time_comm = 0, time_node = 0, time_quad = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    double time_all = MPI_Wtime();
    for (int i = 0; i < numEigenvalues; ++i) {
        Tucker::SizeArray core_size(3);
        core_size[0] = rankTuckerBasisX;
        core_size[1] = rankTuckerBasisY;
        core_size[2] = rankTuckerBasisZ;
        Tucker::Tensor *core = Tucker::MemoryManager::safe_new<Tucker::Tensor>(core_size);

//    MPI_Barrier(MPI_COMM_WORLD);
//    time0 = MPI_Wtime();
//    MPI_Iallgatherv(eigenVectorTucker_seq_data + i * number_local_owned_rows,
//                   number_local_owned_rows,
//                   MPI_DOUBLE,
//                   core->data(),
//                   recvcounts.data(),
//                   displs.data(),
//                   MPI_DOUBLE,
//                   PETSC_COMM_WORLD);
        MPI_Allgatherv(eigenVectorTucker_seq_data + i * number_local_owned_rows,
                       number_local_owned_rows,
                       MPI_DOUBLE,
                       core->data(),
                       recvcounts.data(),
                       displs.data(),
                       MPI_DOUBLE,
                       PETSC_COMM_WORLD);
//    MPI_Barrier(MPI_COMM_WORLD);
//    time_comm += MPI_Wtime() - time0;

//    MPI_Barrier(MPI_COMM_WORLD);
//    time0 = MPI_Wtime();
        if (numNodalLocalEntries != 0) {
            Tucker::Tensor *temp;
            Tucker::Tensor *reconstructedTensor;
            temp = core;
            reconstructedTensor = Tucker::ttm(temp,
                                              2,
                                              MatEigZ);
            temp = reconstructedTensor;
            reconstructedTensor = Tucker::ttm(temp,
                                              0,
                                              MatEigX);
            Tucker::MemoryManager::safe_delete(temp);
            temp = reconstructedTensor;
            reconstructedTensor = Tucker::ttm(temp,
                                              1,
                                              MatEigY);
            Tucker::MemoryManager::safe_delete(temp);

            double *rhoNodalOutData = rhoNodalOut.getLocalData();
            double *reconstructedTensorData = reconstructedTensor->data();
            for (int j = 0; j < numNodalLocalEntries; ++j) {
                rhoNodalOutData[j] +=
                        2.0 * occupancyFactor[i] * reconstructedTensorData[j] * reconstructedTensorData[j];
            }
            Tucker::MemoryManager::safe_delete(reconstructedTensor);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        time_node += MPI_Wtime() - time0;

//    MPI_Barrier(MPI_COMM_WORLD);
//    time0 = MPI_Wtime();
//    if (numGridLocalEntries != 0) {
//      Tucker::Tensor *temp;
//      Tucker::Tensor *reconstructedTensor;
//      temp = core;
//      reconstructedTensor = Tucker::ttm(temp, 2, MatEigQuadZ);
//      temp = reconstructedTensor;
//      reconstructedTensor = Tucker::ttm(temp, 0, MatEigQuadX);
//      Tucker::MemoryManager::safe_delete(temp);
//      temp = reconstructedTensor;
//      reconstructedTensor = Tucker::ttm(temp, 1, MatEigQuadY);
//      Tucker::MemoryManager::safe_delete(temp);
//
//      double *rhoGridOutData = rhoGridOut.getLocalData();
//      double *reconstructedTensorData = reconstructedTensor->data();
//      for (int j = 0; j < numGridLocalEntries; ++j) {
//        rhoGridOutData[j] +=
//            2.0 * occupancyFactor[i] * reconstructedTensorData[j] * reconstructedTensorData[j];
//      }
//      Tucker::MemoryManager::safe_delete(reconstructedTensor);
//    }
//    MPI_Barrier(MPI_COMM_WORLD);
//    time_quad += MPI_Wtime() - time0;

        Tucker::MemoryManager::safe_delete(core);
    }
    MatDenseRestoreArray(eigenVectorTucker,
                         &eigenVectorTucker_seq_data);
    time_all = -time_all + MPI_Wtime();

    // decompose rho nodal
    double epsilon = 1.0e-4;
    MPI_Barrier(MPI_COMM_WORLD);
    double dtime = MPI_Wtime();
    const TuckerMPI::TuckerTensor *decomposed_rho = TuckerMPI::STHOSVD(rhoNodalOut.tensor,
                                                                       epsilon,
                                                                       false);
    MPI_Barrier(MPI_COMM_WORLD);
    dtime = -dtime + MPI_Wtime();
    const auto &drank = decomposed_rho->G->getGlobalSize();

    Tucker::Matrix **U = Tucker::MemoryManager::safe_new_array<Tucker::Matrix *>(3);
    // project factor matrix x onto quad points
    dtime = MPI_Wtime();
    int rho_rank_x = drank[0];
    int num_quad_points_x = femX.getTotalNumberQuadPoints();
    int num_nodal_points_x = femX.getTotalNumberNodes();
    U[0] = Tucker::MemoryManager::safe_new<Tucker::Matrix>(num_quad_points_x,
                                                           rho_rank_x);
    double *U_nodal_ptr_x = decomposed_rho->U[0]->data();
    double *U_quad_ptr_x = U[0]->data();
    for (int i = 0; i < rho_rank_x; ++i) {
        std::vector<double> vtemp(U_nodal_ptr_x + i * num_nodal_points_x,
                                  U_nodal_ptr_x + (i + 1) * num_nodal_points_x);
        std::vector<double> vtemp_quad;
        femX.computeFieldAtAllQuadPoints(vtemp,
                                         vtemp_quad);
        std::copy(vtemp_quad.begin(),
                  vtemp_quad.end(),
                  U_quad_ptr_x + i * num_quad_points_x);
    }

    int rho_rank_y = drank[1];
    int num_quad_points_y = femY.getTotalNumberQuadPoints();
    int num_nodal_points_y = femY.getTotalNumberNodes();
    U[1] = Tucker::MemoryManager::safe_new<Tucker::Matrix>(num_quad_points_y,
                                                           rho_rank_y);
    double *U_nodal_ptr_y = decomposed_rho->U[1]->data();
    double *U_quad_ptr_y = U[1]->data();
    for (int i = 0; i < rho_rank_y; ++i) {
        std::vector<double> vtemp(U_nodal_ptr_y + i * num_nodal_points_y,
                                  U_nodal_ptr_y + (i + 1) * num_nodal_points_y);
        std::vector<double> vtemp_quad;
        femY.computeFieldAtAllQuadPoints(vtemp,
                                         vtemp_quad);
        std::copy(vtemp_quad.begin(),
                  vtemp_quad.end(),
                  U_quad_ptr_y + i * num_quad_points_y);
    }

    int rho_rank_z = drank[2];
    int num_quad_points_z = femZ.getTotalNumberQuadPoints();
    int num_nodal_points_z = femZ.getTotalNumberNodes();
    U[2] = Tucker::MemoryManager::safe_new<Tucker::Matrix>(num_quad_points_z,
                                                           rho_rank_z);
    double *U_nodal_ptr_z = decomposed_rho->U[2]->data();
    double *U_quad_ptr_z = U[2]->data();
    for (int i = 0; i < rho_rank_z; ++i) {
        std::vector<double> vtemp(U_nodal_ptr_z + i * num_nodal_points_z,
                                  U_nodal_ptr_z + (i + 1) * num_nodal_points_z);
        std::vector<double> vtemp_quad;
        femZ.computeFieldAtAllQuadPoints(vtemp,
                                         vtemp_quad);
        std::copy(vtemp_quad.begin(),
                  vtemp_quad.end(),
                  U_quad_ptr_z + i * num_quad_points_z);
    }
    dtime = -dtime + MPI_Wtime();


    for (int i = 0; i < 3; ++i) {
        Tucker::MemoryManager::safe_delete<Tucker::Matrix>(decomposed_rho->U[i]);
        decomposed_rho->U[i] = U[i];
    }


//  Tucker::MemoryManager::safe_delete(rho_quad_ptr);

//  std::cout << "max: " << rhoNodalOut.tensor->maxEntry() << ", " << rhoGridOut.tensor->maxEntry() << std::endl;
    TuckerMPI::Tensor *rho_quad_ptr = decomposed_rho->reconstructTensor();
    Tucker::MemoryManager::safe_delete(decomposed_rho);
    double *rho_quad_ptr_data = rho_quad_ptr->getLocalTensor()->data();
    for (int i = 0;
         i < rho_quad_ptr->getLocalNumEntries(); ++i) { rho_quad_ptr_data[i] = std::abs(rho_quad_ptr_data[i]); }
    std::copy(rho_quad_ptr_data,
              rho_quad_ptr_data + rho_quad_ptr->getLocalNumEntries(),
              rhoGridOut.getLocalData());


//  Tensor3DMPI jacob(num_quad_points_x, num_quad_points_y, num_quad_points_z);
//  Tensor3DMPI weight(num_quad_points_x, num_quad_points_y, num_quad_points_z);
//  computeJacobWeight3DMat(femX, femY, femZ, jacob, weight);


//  std::cout << std::setprecision(16);
//  std::cout << "max: " << rho_quad_ptr->maxEntry() << ", " << rhoNodalOut.tensor->maxEntry() << ", " << rhoGridOut.tensor->maxEntry() << ", " << rhoGridOut.tensor->minEntry() << ", " << rhoGridOut.tensor->norm2() << std::endl;
    Tucker::MemoryManager::safe_delete(rho_quad_ptr);

    Tucker::MemoryManager::safe_delete(MatEigX);
    Tucker::MemoryManager::safe_delete(MatEigY);
    Tucker::MemoryManager::safe_delete(MatEigZ);
//  Tucker::MemoryManager::safe_delete(MatEigQuadX);
//  Tucker::MemoryManager::safe_delete(MatEigQuadY);
//  Tucker::MemoryManager::safe_delete(MatEigQuadZ);

//  PetscPrintf(PETSC_COMM_WORLD, "ProjectHamiltonianSparse::computeRhoOut: all time %.4f\n", time_all);
//  PetscPrintf(PETSC_COMM_WORLD, "ProjectHamiltonianSparse::computeRhoOut: communication %.4f\n", time_comm);
//  PetscPrintf(PETSC_COMM_WORLD, "ProjectHamiltonianSparse::computeRhoOut: nodal reconstruction %.4f\n", time_node);
//  PetscPrintf(PETSC_COMM_WORLD, "ProjectHamiltonianSparse::computeRhoOut: quadrature reconstruction %.4f\n", time_quad);
}

void ProjectHamiltonianSparse::Compute_nnz_loc(const TuckerMPI::TuckerTensor *tuckerDecomposedVeff,
                                               const int rankVeffX,
                                               const int rankVeffY,
                                               const int rankVeffZ,
                                               const int rankTuckerBasisX,
                                               const int rankTuckerBasisY,
                                               const int rankTuckerBasisZ,
                                               double truncation,
                                               std::vector<PetscInt> &Hloc_dnz,
                                               std::vector<PetscInt> &Hloc_onz) {
//  PetscPrintf(PETSC_COMM_WORLD, "computing nnz pattern for Hloc\n");
    std::vector<std::vector<double> > MatX(rankTuckerBasisX,
                                           std::vector<double>(rankTuckerBasisX));
    std::vector<std::vector<double> > MatY(rankTuckerBasisY,
                                           std::vector<double>(rankTuckerBasisY));
    std::vector<std::vector<double> > MatZ(rankTuckerBasisZ,
                                           std::vector<double>(rankTuckerBasisZ));
    std::vector<std::vector<double> > MatDX(rankTuckerBasisX,
                                            std::vector<double>(rankTuckerBasisX));
    std::vector<std::vector<double> > MatDY(rankTuckerBasisY,
                                            std::vector<double>(rankTuckerBasisY));
    std::vector<std::vector<double> > MatDZ(rankTuckerBasisZ,
                                            std::vector<double>(rankTuckerBasisZ));
    std::vector<std::vector<std::vector<double> > >
            MatPotX(rankVeffX,
                    std::vector<std::vector<double> >(rankTuckerBasisX,
                                                      std::vector<double>(rankTuckerBasisX)));
    std::vector<std::vector<std::vector<double> > >
            MatPotY(rankVeffY,
                    std::vector<std::vector<double> >(rankTuckerBasisY,
                                                      std::vector<double>(rankTuckerBasisY)));
    std::vector<std::vector<std::vector<double> > >
            MatPotZ(rankVeffZ,
                    std::vector<std::vector<double> >(rankTuckerBasisZ,
                                                      std::vector<double>(rankTuckerBasisZ)));
//  PetscPrintf(PETSC_COMM_WORLD, "kinetic\n");
    computeOverlapKineticTuckerBasis(femX,
                                     femY,
                                     femZ,
                                     rankTuckerBasisX,
                                     rankTuckerBasisY,
                                     rankTuckerBasisZ,
                                     basisXQuadValues,
                                     basisDXQuadValues,
                                     basisYQuadValues,
                                     basisDYQuadValues,
                                     basisZQuadValues,
                                     basisDZQuadValues,
                                     MatX,
                                     MatY,
                                     MatZ,
                                     MatDX,
                                     MatDY,
                                     MatDZ);
//  PetscPrintf(PETSC_COMM_WORLD, "overlap\n");
    computeOverlapPotentialTuckerBasis(femX,
                                       femY,
                                       femZ,
                                       tuckerDecomposedVeff,
                                       basisXQuadValues,
                                       basisYQuadValues,
                                       basisZQuadValues,
                                       rankVeffX,
                                       rankVeffY,
                                       rankVeffZ,
                                       rankTuckerBasisX,
                                       rankTuckerBasisY,
                                       rankTuckerBasisZ,
                                       MatPotX,
                                       MatPotY,
                                       MatPotZ);
    std::vector<std::vector<int>>
            matToTucker(3,
                        std::vector<int>(rankTuckerBasisX * rankTuckerBasisY * rankTuckerBasisZ));
    int cnt = 0;
    for (int zI = 0; zI < rankTuckerBasisZ; ++zI) {
        for (int yI = 0; yI < rankTuckerBasisY; ++yI) {
            for (int xI = 0; xI < rankTuckerBasisX; ++xI) {
                matToTucker[0][cnt] = xI;
                matToTucker[1][cnt] = yI;
                matToTucker[2][cnt] = zI;
                cnt++;
            }
        }
    }

    Tucker::SizeArray coreVeffDim(3);
    coreVeffDim[0] = rankVeffX;
    coreVeffDim[1] = rankVeffY;
    coreVeffDim[2] = rankVeffZ;
    Tucker::Tensor *coreVeffSeq = Tucker::MemoryManager::safe_new<Tucker::Tensor>(coreVeffDim);
    TensorUtils::allreduce_tensor(tuckerDecomposedVeff->G,
                                  coreVeffSeq);

    // count nnc for H_loc
    PetscInt H_loc_istart, H_loc_iend;
    int basis_dimension = rankTuckerBasisX * rankTuckerBasisY * rankTuckerBasisZ;
    Mat fake_mat;
    MatCreateAIJ(PETSC_COMM_WORLD,
                 PETSC_DECIDE,
                 PETSC_DECIDE,
                 basis_dimension,
                 basis_dimension,
                 1,
                 PETSC_NULL,
                 1,
                 PETSC_NULL,
                 &fake_mat);
    MatGetOwnershipRange(fake_mat,
                         &H_loc_istart,
                         &H_loc_iend);
    MatDestroy(&fake_mat);

    int H_loc_num_local_rows = H_loc_iend - H_loc_istart;

    Hloc_dnz.resize(H_loc_num_local_rows,
                    0);
    Hloc_onz.resize(H_loc_num_local_rows,
                    0);
    std::vector<double> band_each_row(H_loc_num_local_rows,
                                      0);

//  MPI_Barrier(PETSC_COMM_WORLD);
//  double time_loc = MPI_Wtime();
    std::vector<double> num_zeros_6_to_14(9,
                                          0);
    for (int irow = 0; irow < H_loc_num_local_rows; ++irow) {
        int row = irow + H_loc_istart;
        int xI = matToTucker[0][row];
        int yI = matToTucker[1][row];
        int zI = matToTucker[2][row];

        const std::vector<double> &localMatX = MatX[xI];
        const std::vector<double> &localMatY = MatY[yI];
        const std::vector<double> &localMatZ = MatZ[zI];
        const std::vector<double> &localMatDX = MatDX[xI];
        const std::vector<double> &localMatDY = MatDY[yI];
        const std::vector<double> &localMatDZ = MatDZ[zI];

        // compute falttented ((a x b) x c), x stands for tensor product
        std::vector<double> dyad(rankTuckerBasisX * rankTuckerBasisY * rankTuckerBasisZ,
                                 0.0);
        int cnt = 0;
        for (int zrank = 0; zrank < rankTuckerBasisZ; ++zrank) {
            for (int yrank = 0; yrank < rankTuckerBasisY; ++yrank) {
                for (int xrank = 0; xrank < rankTuckerBasisX; ++xrank) {
                    dyad[cnt] = localMatDX[xrank] * localMatY[yrank] * localMatZ[zrank]
                                + localMatX[xrank] * localMatDY[yrank] * localMatZ[zrank]
                                + localMatX[xrank] * localMatY[yrank] * localMatDZ[zrank];
                    dyad[cnt] = 0.5 * dyad[cnt];
                    cnt++;
                }
            }
        }// end of x,y,zrank
        Tucker::Matrix *MatPotXLocal = Tucker::MemoryManager::safe_new<Tucker::Matrix>(rankTuckerBasisX,
                                                                                       rankVeffX);
        Tucker::Matrix *MatPotYLocal = Tucker::MemoryManager::safe_new<Tucker::Matrix>(rankTuckerBasisY,
                                                                                       rankVeffY);
        Tucker::Matrix *MatPotZLocal = Tucker::MemoryManager::safe_new<Tucker::Matrix>(rankTuckerBasisZ,
                                                                                       rankVeffZ);
        for (int i = 0; i < rankVeffX; ++i) {
            std::copy(MatPotX[i][xI].begin(),
                      MatPotX[i][xI].end(),
                      MatPotXLocal->data() + i * rankTuckerBasisX);
        }
        for (int i = 0; i < rankVeffY; ++i) {
            std::copy(MatPotY[i][yI].begin(),
                      MatPotY[i][yI].end(),
                      MatPotYLocal->data() + i * rankTuckerBasisY);
        }
        for (int i = 0; i < rankVeffZ; ++i) {
            std::copy(MatPotZ[i][zI].begin(),
                      MatPotZ[i][zI].end(),
                      MatPotZLocal->data() + i * rankTuckerBasisZ);
        }
        // recontruct tensor, reconstructedTensor will be the final constructed tensor
        Tucker::Tensor *temp;
        Tucker::Tensor *reconstructedTensor;
        temp = coreVeffSeq;
        reconstructedTensor = Tucker::ttm(temp,
                                          0,
                                          MatPotXLocal);
        temp = reconstructedTensor;
        reconstructedTensor = Tucker::ttm(temp,
                                          1,
                                          MatPotYLocal);
        Tucker::MemoryManager::safe_delete(temp);
        temp = reconstructedTensor;
        reconstructedTensor = Tucker::ttm(temp,
                                          2,
                                          MatPotZLocal);
        Tucker::MemoryManager::safe_delete(temp);

        int numTensorElements = reconstructedTensor->getNumElements();
        double *reconstructedTensorData = reconstructedTensor->data();
        for (int i = 0; i < numTensorElements; ++i) {
            dyad[i] += reconstructedTensorData[i];
        }
        Tucker::MemoryManager::safe_delete(reconstructedTensor);
        Tucker::MemoryManager::safe_delete(MatPotXLocal);
        Tucker::MemoryManager::safe_delete(MatPotYLocal);
        Tucker::MemoryManager::safe_delete(MatPotZLocal);

        for (int j = 0; j < dyad.size(); ++j) {
            double val = std::abs(dyad[j]);
            num_zeros_6_to_14[0] += val < 1.0e-6;
            num_zeros_6_to_14[1] += val < 1.0e-7;
            num_zeros_6_to_14[2] += val < 1.0e-8;
            num_zeros_6_to_14[3] += val < 1.0e-9;
            num_zeros_6_to_14[4] += val < 1.0e-10;
            num_zeros_6_to_14[5] += val < 1.0e-11;
            num_zeros_6_to_14[6] += val < 1.0e-12;
            num_zeros_6_to_14[7] += val < 1.0e-13;
            num_zeros_6_to_14[8] += val < 1.0e-14;
            if (val > truncation) {
                if (j >= H_loc_istart && j < H_loc_iend) {
                    Hloc_dnz[irow] += 1;
                } else {
                    Hloc_onz[irow] += 1;
                }
                band_each_row[irow] += 1;
            }
        }

        //std::copy(dyad.begin(), dyad.end(), localProjH + row * numColumns);
    }// end of row
    Tucker::MemoryManager::safe_delete(coreVeffSeq);
//  MPI_Barrier(PETSC_COMM_WORLD);
//  time_loc = MPI_Wtime() - time_loc;
//  PetscPrintf(MPI_COMM_WORLD, "time for compute nnzs for H_loc matrix: %.6e\n", time_loc);

    std::vector<double> num_zeros_6_to_14_sum(9,
                                              0);
    MPI_Allreduce(num_zeros_6_to_14.data(),
                  num_zeros_6_to_14_sum.data(),
                  9,
                  MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD);

    for (int k = 0; k < 9; ++k) {
        double basis_size = 1.0 * rankTuckerBasisX * rankTuckerBasisY * rankTuckerBasisZ;
        num_zeros_6_to_14_sum[k] /= (basis_size * basis_size);
//    PetscPrintf(PETSC_COMM_WORLD, "1e-%d: %.4e\n", k + 6, num_zeros_6_to_14_sum[k]);
    }

    double max_diagonal_band_loc, min_diagonal_band_loc, max_diagonal_band_all, min_diagonal_band_all;
    double max_offdiagonal_band_loc, min_offdiagonal_band_loc, max_offdiagonal_band_all, min_offdiagonal_band_all;
    double total_dnz_loc = 0.0, total_onz_loc = 0.0, max_dnz, min_dnz, max_onz, min_onz;

    double max_band_loc, min_band_loc, max_band_all, min_band_all;
    double total_elements_loc = 0, max_total_elements_all, min_total_elements_all;
    for (int l = 0; l < band_each_row.size(); ++l) {
        total_elements_loc += band_each_row[l];
        total_dnz_loc += Hloc_dnz[l];
        total_onz_loc += Hloc_onz[l];
    }
    max_diagonal_band_loc = *(std::max_element(Hloc_dnz.begin(),
                                               Hloc_dnz.end()));
    min_diagonal_band_loc = *(std::min_element(Hloc_dnz.begin(),
                                               Hloc_dnz.end()));
    max_offdiagonal_band_loc = *(std::max_element(Hloc_onz.begin(),
                                                  Hloc_onz.end()));
    min_offdiagonal_band_loc = *(std::min_element(Hloc_onz.begin(),
                                                  Hloc_onz.end()));
    max_band_loc = *(std::max_element(band_each_row.begin(),
                                      band_each_row.end()));
    min_band_loc = *(std::min_element(band_each_row.begin(),
                                      band_each_row.end()));
    MPI_Allreduce(&max_diagonal_band_loc,
                  &max_diagonal_band_all,
                  1,
                  MPI_DOUBLE,
                  MPI_MAX,
                  PETSC_COMM_WORLD);
    MPI_Allreduce(&min_diagonal_band_loc,
                  &min_diagonal_band_all,
                  1,
                  MPI_DOUBLE,
                  MPI_MIN,
                  PETSC_COMM_WORLD);
    MPI_Allreduce(&max_offdiagonal_band_loc,
                  &max_offdiagonal_band_all,
                  1,
                  MPI_DOUBLE,
                  MPI_MAX,
                  PETSC_COMM_WORLD);
    MPI_Allreduce(&min_offdiagonal_band_loc,
                  &min_offdiagonal_band_all,
                  1,
                  MPI_DOUBLE,
                  MPI_MIN,
                  PETSC_COMM_WORLD);
    MPI_Allreduce(&max_band_loc,
                  &max_band_all,
                  1,
                  MPI_DOUBLE,
                  MPI_MAX,
                  PETSC_COMM_WORLD);
    MPI_Allreduce(&min_band_loc,
                  &min_band_all,
                  1,
                  MPI_DOUBLE,
                  MPI_MIN,
                  PETSC_COMM_WORLD);
    MPI_Allreduce(&total_elements_loc,
                  &max_total_elements_all,
                  1,
                  MPI_DOUBLE,
                  MPI_MAX,
                  PETSC_COMM_WORLD);
    MPI_Allreduce(&total_elements_loc,
                  &min_total_elements_all,
                  1,
                  MPI_DOUBLE,
                  MPI_MIN,
                  PETSC_COMM_WORLD);
    MPI_Allreduce(&total_dnz_loc,
                  &max_dnz,
                  1,
                  MPI_DOUBLE,
                  MPI_MAX,
                  PETSC_COMM_WORLD);
    MPI_Allreduce(&total_dnz_loc,
                  &min_dnz,
                  1,
                  MPI_DOUBLE,
                  MPI_MIN,
                  PETSC_COMM_WORLD);
    MPI_Allreduce(&total_onz_loc,
                  &max_onz,
                  1,
                  MPI_DOUBLE,
                  MPI_MAX,
                  PETSC_COMM_WORLD);
    MPI_Allreduce(&total_onz_loc,
                  &min_onz,
                  1,
                  MPI_DOUBLE,
                  MPI_MIN,
                  PETSC_COMM_WORLD);

//  PetscPrintf(PETSC_COMM_WORLD, "digonal band max, min: %.4e, %.4e\n", max_diagonal_band_all, min_diagonal_band_all);
//  PetscPrintf(PETSC_COMM_WORLD,
//              "off digonal band max, min: %.4e, %.4e\n",
//              max_offdiagonal_band_all,
//              min_offdiagonal_band_all);
//  PetscPrintf(PETSC_COMM_WORLD, "band max, min: %.4e, %.4e\n", max_band_all, min_band_all);
//  PetscPrintf(PETSC_COMM_WORLD,
//              "total local elemeents max, min: %.4e, %.4e\n",
//              max_total_elements_all,
//              min_total_elements_all);
//  PetscPrintf(PETSC_COMM_WORLD, "total local diagonal elements max, min: %.4e, %.4e\n", max_dnz, min_dnz);
//  PetscPrintf(PETSC_COMM_WORLD, "total local off diagonal elements max, min: %.4e, %.4e\n", max_onz, min_onz);

    for (int j = 0; j < Hloc_dnz.size(); ++j) {
        if (Hloc_dnz[j] * 1.5 < H_loc_num_local_rows) {
            Hloc_dnz[j] *= 1.5;
        } else {
            Hloc_dnz[j] = H_loc_num_local_rows;
        }

        if (Hloc_onz[j] * 1.5 < (basis_dimension - H_loc_num_local_rows)) {
            Hloc_onz[j] *= 1.5;
        } else {
            Hloc_onz[j] = basis_dimension - H_loc_num_local_rows;
        }
    }

}

ProjectHamiltonianSparse::~ProjectHamiltonianSparse() {
    MatDestroy(&C_nloc);
    MatDestroy(&C_nloc_trans);
}

void ProjectHamiltonianSparse::Create_Cnloc_mat_info(const AtomInformation &atom_information,
                                                     PetscInt &num_global_rows,
                                                     PetscInt &num_local_rows,
                                                     PetscInt &owned_row_start,
                                                     PetscInt &owned_row_end,
                                                     std::vector<NonLocRowInfo> &owned_rows_info,
                                                     std::vector<int> &owned_atom_id,
                                                     std::map<int, int> &owned_atom_id_g2l,
                                                     std::vector<int> &owned_atom_types,
                                                     std::map<int, int> &owned_atom_types_g2l) {

    int num_atom_type = atom_information.numAtomType;
    std::vector<int> num_atoms_each_type(num_atom_type,
                                         0), num_lmsq_each_type(num_atom_type,
                                                                0);
    for (int atom_type_i = 0; atom_type_i < num_atom_type; ++atom_type_i) {
        num_atoms_each_type[atom_type_i] = atom_information.nuclei[atom_type_i].size();
        int lMax = atom_information.lMax[atom_type_i];
        num_lmsq_each_type[atom_type_i] = lMax * lMax;
    }
    num_global_rows = 0;
    for (int atom_type_i = 0; atom_type_i < num_atom_type; ++atom_type_i) {
        num_global_rows += num_atoms_each_type[atom_type_i] * num_lmsq_each_type[atom_type_i];\

    }
    Vec fake_vec;
    VecCreateMPI(PETSC_COMM_WORLD,
                 PETSC_DECIDE,
                 num_global_rows,
                 &fake_vec);
    VecAssemblyBegin(fake_vec);
    VecAssemblyEnd(fake_vec);
    VecGetOwnershipRange(fake_vec,
                         &owned_row_start,
                         &owned_row_end);
    VecGetLocalSize(fake_vec,
                    &num_local_rows);
    VecDestroy(&fake_vec);

    // compute owned atom id and owned atom type as a local to global map
    PetscInt row_cnt = 0;
    for (int atomtype_i = 0; atomtype_i < num_atom_type; ++atomtype_i) {
        for (int atom_i = 0; atom_i < num_atoms_each_type[atomtype_i]; ++atom_i) {
            for (int lm_i = 0; lm_i < num_lmsq_each_type[atomtype_i]; ++lm_i) {
                if (row_cnt >= owned_row_start && row_cnt < owned_row_end) {
                    if (std::find(owned_atom_types.begin(),
                                  owned_atom_types.end(),
                                  atomtype_i) == owned_atom_types.end()) {
                        owned_atom_types.emplace_back(atomtype_i);
                    }
                    if (std::find(owned_atom_id.begin(),
                                  owned_atom_id.end(),
                                  atom_i) == owned_atom_id.end()) {
                        owned_atom_id.emplace_back(atom_i);
                    }
                }
                row_cnt++;
            }
        }
    }

    for (int owned_atom_type_i = 0; owned_atom_type_i < owned_atom_types.size(); ++owned_atom_type_i) {
        owned_atom_types_g2l[owned_atom_types[owned_atom_type_i]] = owned_atom_type_i;
    }

    for (int owned_atom_i = 0; owned_atom_i < owned_atom_id.size(); ++owned_atom_i) {
        owned_atom_id_g2l[owned_atom_id[owned_atom_i]] = owned_atom_i;
    }

    row_cnt = 0;
    for (int atomtype_i = 0; atomtype_i < num_atom_type; ++atomtype_i) {
        for (int atom_i = 0; atom_i < num_atoms_each_type[atomtype_i]; ++atom_i) {
            for (int lm_i = 0; lm_i < num_lmsq_each_type[atomtype_i]; ++lm_i) {
                if (row_cnt >= owned_row_start && row_cnt < owned_row_end) {
                    owned_rows_info.emplace_back(NonLocRowInfo(atomtype_i,
                                                               owned_atom_types_g2l[atomtype_i],
                                                               atom_i,
                                                               owned_atom_id_g2l[atom_i],
                                                               lm_i,
                                                               row_cnt));
                }
                row_cnt++;
            }
        }
    }
}
