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

//
// Created by iancclin
//

#include "MultTm.h"
#include "DeviceUtils.cuh"
#include <cuda_runtime.h>
#include <iostream>

MultTM::MultTM() {

  MPI_Comm nodal_comm;
  int nodal_size, nodal_rank;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                      &nodal_comm);

  MPI_Comm_size(nodal_comm, &nodal_size);
  MPI_Comm_rank(nodal_comm, &nodal_rank);
  int max_nodal_size, min_nodal_size;
  MPI_Allreduce(&nodal_size,
                &max_nodal_size,
                1,
                MPI_INT,
                MPI_MAX,
                MPI_COMM_WORLD);
  MPI_Allreduce(&nodal_size,
                &min_nodal_size,
                1,
                MPI_INT,
                MPI_MIN,
                MPI_COMM_WORLD);

  if (max_nodal_size!=min_nodal_size) {
    std::cout << "ERROR: each node should have the same number of cpus."
              << std::endl;
    std::terminate();
  }

  device_utils::device_get_device_count(local_num_devices);
  if (local_num_devices==0) {
    std::cout << "ERROR: no gpu or gpu is more than cpus on the node "
                 "(currently not supported.)"
              << std::endl;
    std::terminate();
  }

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int num_tasks_per_device = nodal_size/local_num_devices;
  int local_device_id = nodal_rank/num_tasks_per_device;
  device_utils::device_set_device(local_device_id);
  global_num_devices = local_num_devices*(world_size/nodal_size);
  int global_device_id = world_rank/num_tasks_per_device;
  MPI_Comm_split(MPI_COMM_WORLD, global_device_id, world_rank, &comm_tm);

  MPI_Comm_size(comm_tm, &comm_tm_size);
  MPI_Comm_rank(comm_tm, &comm_tm_rank);

  cublasCreate(&cublas_handle);
  cusolverDnCreate(&cusolverdn_handle);
  cudaStreamCreateWithFlags(&cusolver_stream, cudaStreamNonBlocking);
  cusolverDnSetStream(cusolverdn_handle, cusolver_stream);
}

void MultTM::mult(Mat &X, Mat &Y, std::vector<double> &result) {

  const PetscInt *x_range, *y_range;
  MatGetOwnershipRanges(X, &x_range);
  MatGetOwnershipRanges(Y, &y_range);
  for (int i_rank = 0; i_rank < world_size; ++i_rank) {
    if (x_range[i_rank]!=y_range[i_rank]) {
      std::cout << "X and Y matrix should have same row ownerships."
                << std::endl;
      std::terminate();
    }
  }

  PetscInt global_m, global_n;
  MatGetSize(X, &global_m, &global_n);
  std::vector<PetscInt> row_t, col_t(global_n, 0);
  for (int i = 0; i < global_n; ++i)
    col_t[i] = i;
  if (comm_tm_rank==0) {
    int start = x_range[world_rank];
    int end = x_range[world_rank + comm_tm_size];
    row_t = std::vector<PetscInt>(end - start, start);
    for (int i = 0; i < row_t.size(); ++i)
      row_t[i] += i;
  }

  IS is_row, is_col;
  ISCreateGeneral(MPI_COMM_WORLD,
                  row_t.size(),
                  row_t.data(),
                  PETSC_COPY_VALUES,
                  &is_row);
  ISCreateGeneral(MPI_COMM_WORLD,
                  col_t.size(),
                  col_t.data(),
                  PETSC_COPY_VALUES,
                  &is_col);

  Mat *subX, *subY;
  MatCreateSubMatrices(X,
                       1,
                       &is_row,
                       &is_col,
                       MAT_INITIAL_MATRIX,
                       &subX);
  MatCreateSubMatrices(Y,
                       1,
                       &is_row,
                       &is_col,
                       MAT_INITIAL_MATRIX,
                       &subY);

  result = std::vector<double>(col_t.size()*col_t.size(), 0.0);

  if (comm_tm_rank==0) {
    double *x_lht, *y_lht;
    MatDenseGetArray(subX[0], &x_lht);
    MatDenseGetArray(subY[0], &y_lht);

    double *x_ld, *y_ld, *xty_ld;
    cudaError_t cudat_error1, cudat_error2, cudat_error3;
    cudat_error1 =
        cudaMalloc(&x_ld,
                   row_t.size()*col_t.size()*sizeof(double));
    cudat_error2 =
        cudaMalloc(&y_ld,
                   row_t.size()*col_t.size()*sizeof(double));
    cudat_error3 =
        cudaMalloc(&xty_ld,
                   col_t.size()*col_t.size()*sizeof(double));

    if ((cudat_error1!=cudaSuccess) || (cudat_error2!=cudaSuccess) ||
        (cudat_error3!=cudaSuccess)) {
      std::cout << "device malloc failed." << std::endl;
      std::terminate();
    }

    cudat_error1 =
        cudaMemcpy(&x_ld, &x_lht, row_t.size()*col_t.size()*sizeof(double),
                   cudaMemcpyHostToDevice);
    cudat_error2 =
        cudaMemcpy(&y_ld, &y_lht, row_t.size()*col_t.size()*sizeof(double),
                   cudaMemcpyHostToDevice);
    if ((cudat_error1!=cudaSuccess) || (cudat_error2!=cudaSuccess)) {
      std::cout << "host to device memory copy failed." << std::endl;
      std::terminate();
    }

    double double_one = 1.0;
    cublasStatus_t blas_stat;
    blas_stat =
        cublasDgemm(cublas_handle,
                    CUBLAS_OP_T,
                    CUBLAS_OP_N,
                    col_t.size(),
                    col_t.size(),
                    row_t.size(),
                    &double_one,
                    x_ld,
                    row_t.size(),
                    y_ld,
                    row_t.size(),
                    &double_one,
                    xty_ld,
                    col_t.size());
    if (blas_stat!=CUBLAS_STATUS_SUCCESS) {
      std::cout << "CUBLAS matrix-matrix multiplication failed" << std::endl;
      std::terminate();
    }

    cudat_error1 = cudaFree(&x_ld);
    cudat_error2 = cudaFree(&y_ld);

    cudaMemcpy(result.data(), xty_ld,
               col_t.size()*col_t.size()*sizeof(double),
               cudaMemcpyDeviceToHost);

    cudat_error3 = cudaFree(&xty_ld);

    if ((cudat_error1!=cudaSuccess) || (cudat_error2!=cudaSuccess) ||
        (cudat_error3!=cudaSuccess)) {
      std::cout << "device memory free failed." << std::endl;
      std::terminate();
    }

    MatDenseRestoreArray(subX[0], &x_lht);
    MatDenseRestoreArray(subY[0], &y_lht);
  }

  MatDestroySubMatrices(1, &subX);
  MatDestroySubMatrices(1, &subY);

  MPI_Allreduce(MPI_IN_PLACE,
                result.data(),
                result.size(),
                MPI_DOUBLE,
                MPI_SUM,
                MPI_COMM_WORLD);
}

void MultTM::orth(Mat &X) {
  std::vector<double> S;
  mult(X, X, S);
  PetscInt global_m, global_n;
  MatGetSize(X, &global_m, &global_n);
  int mat_m = global_m, mat_n = global_n;

  cudaError_t cudat_error1, cudat_error2;
  int *info_d = nullptr;
  int worksize_d = 0;
  double *workspace_d = nullptr;

  double *S_d = nullptr;
  cudat_error1 = cudaMalloc(&S_d, sizeof(double)*S.size());
  if ((cudat_error1!=cudaSuccess)) {
    std::cout << "device memory allocation failed." << std::endl;
    std::terminate();
  }

  cudat_error1 = cudaMemcpy(S_d,
                            S.data(),
                            sizeof(double)*S.size(),
                            cudaMemcpyHostToDevice);
  if ((cudat_error1!=cudaSuccess)) {
    std::cout << "device memory copy failed." << std::endl;
    std::terminate();
  }

  cusolverStatus_t solver_stat;
  cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
  solver_stat = cusolverDnDpotrf_bufferSize(cusolverdn_handle,
                                            uplo,
                                            mat_m,
                                            S_d,
                                            mat_m,
                                            &worksize_d);
  if (solver_stat!=CUSOLVER_STATUS_SUCCESS) {
    std::cout << "CUSOLVER buffer preparation failed" << std::endl;
    std::terminate();
  }

  cudat_error1 = cudaMalloc(&workspace_d,
                            sizeof(double)*worksize_d);
  cudat_error2 = cudaMalloc(&info_d,
                            sizeof(int));
  if ((cudat_error1!=cudaSuccess) || (cudat_error2!=cudaSuccess)) {
    std::cout << "device memory allocation failed." << std::endl;
    std::terminate();
  }
  cudat_error1 = cudaMemset(&info_d,
                            0,
                            sizeof(int));
  if ((cudat_error1!=cudaSuccess)) {
    std::cout << "device memory set failed." << std::endl;
    std::terminate();
  }

  solver_stat = cusolverDnDpotrf(cusolverdn_handle,
                                 uplo,
                                 mat_m,
                                 S_d,
                                 mat_m,
                                 workspace_d,
                                 worksize_d,
                                 info_d);
  if (solver_stat!=CUSOLVER_STATUS_SUCCESS) {
    std::cout << "CUSOLVER potrf failed" << std::endl;
    std::terminate();
  }
  cudat_error1 = cudaFree(workspace_d);
  if ((cudat_error1!=cudaSuccess)) {
    std::cout << "device memory free failed." << std::endl;
    std::terminate();
  }


  PetscInt local_m, local_n;
  MatGetSize(X, &local_m, &local_n);
  int local_mat_m = local_m, local_mat_n = local_n;

  double *X_lh, *X_ld;
  MatDenseGetArray(X, &X_lh);

#ifdef SOLVE_WITH_INVERSE
  solver_stat = cusolverDnDtrtri_bufferSize(cusolverdn_handle,
                                            uplo,
                                            CUBLAS_DIAG_NON_UNIT,
                                            mat_m,
                                            S_d,
                                            mat_m,
                                            &worksize_d);
  if (solver_stat!=CUSOLVER_STATUS_SUCCESS) {
    std::cout << "CUSOLVER trtri buffer preparation failed" << std::endl;
    std::terminate();
  }

  cudat_error1 = cudaMalloc(&workspace_d,
                            sizeof(double)*worksize_d);
  if ((cudat_error1!=cudaSuccess)) {
    std::cout << "device memory allocation failed." << std::endl;
    std::terminate();
  }

  cudat_error1 = cudaMemset(&info_d,
                            0,
                            sizeof(int));
  if ((cudat_error1!=cudaSuccess)) {
    std::cout << "device memory set failed." << std::endl;
    std::terminate();
  }

  solver_stat = cusolverDnDtrtri(cusolverdn_handle,
                                 uplo,
                                 CUBLAS_DIAG_NON_UNIT,
                                 mat_m,
                                 S_d,
                                 mat_m,
                                 workspace_d,
                                 worksize_d,
                                 info_d);
  if (solver_stat!=CUSOLVER_STATUS_SUCCESS) {
    std::cout << "CUSOLVER trtri execution failed" << std::endl;
    std::terminate();
  }
  cudat_error1 = cudaFree(workspace_d);
  if ((cudat_error1!=cudaSuccess)) {
    std::cout << "device memory free failed." << std::endl;
    std::terminate();
  }
#endif

  cudat_error1 = cudaMalloc(&X_ld,
                            local_mat_m*local_mat_n*sizeof(double));
  if ((cudat_error1!=cudaSuccess)) {
    std::cout << "device memory allocation failed." << std::endl;
    std::terminate();
  }

  cudat_error1 =
      cudaMemcpy(&X_ld, &X_lh,
                 local_mat_m*local_mat_n*sizeof(double),
                 cudaMemcpyHostToDevice);
  if ((cudat_error1!=cudaSuccess)) {
    std::cout << "device memory copy failed." << std::endl;
    std::terminate();
  }

  double double_one = 1.0;
  cublasStatus_t blas_stat;

#ifdef SOLVE_WITH_INVERSE
  double *XLt_d;
  cudat_error1 = cudaMalloc(&XLt_d,
                            local_mat_m*local_mat_n*sizeof(double));
  if ((cudat_error1!=cudaSuccess)) {
    std::cout << "device memory allocation failed." << std::endl;
    std::terminate();
  }

  blas_stat =
      cublasDtrmm(cublas_handle,
                  CUBLAS_SIDE_RIGHT,
                  uplo,
                  CUBLAS_OP_T,
                  CUBLAS_DIAG_NON_UNIT,
                  local_mat_m,
                  local_mat_n,
                  &double_one,
                  S_d,
                  local_mat_n,
                  X_ld,
                  local_mat_m,
                  XLt_d,
                  local_mat_m);
  if (blas_stat!=CUBLAS_STATUS_SUCCESS) {
    std::cout << "CUBLAS triangular matrix-matrix multiplication failed" << std::endl;
    std::terminate();
  }

  cudat_error1 =
      cudaMemcpy(&X_lh, &XLt_d,
                 local_mat_m*local_mat_n*sizeof(double),
                 cudaMemcpyDeviceToHost);
  if ((cudat_error1!=cudaSuccess)) {
    std::cout << "host memory copy failed." << std::endl;
    std::terminate();
  }


  cudat_error1 = cudaFree(&XLt_d);
  if ((cudat_error1!=cudaSuccess)) {
  std::cout << "device memory free failed." << std::endl;
  std::terminate();
  }
#else

  blas_stat =
      cublasDtrsm(cublas_handle,
                  CUBLAS_SIDE_RIGHT,
                  uplo,
                  CUBLAS_OP_T,
                  CUBLAS_DIAG_NON_UNIT,
                  local_mat_m,
                  local_mat_n,
                  &double_one,
                  S_d,
                  local_mat_n,
                  X_ld,
                  local_mat_m);
  if (blas_stat!=CUBLAS_STATUS_SUCCESS) {
    std::cout << "CUBLAS triangular matrix-matrix solve failed" << std::endl;
    std::terminate();
  }

  cudat_error1 =
      cudaMemcpy(&X_lh, &X_ld,
                 local_mat_m*local_mat_n*sizeof(double),
                 cudaMemcpyDeviceToHost);
  if ((cudat_error1!=cudaSuccess)) {
    std::cout << "host memory copy failed." << std::endl;
    std::terminate();
  }

#endif

  cudat_error1 = cudaFree(info_d);
  if ((cudat_error1!=cudaSuccess)) {
    std::cout << "device memory free failed." << std::endl;
    std::terminate();
  }

  cudat_error1 = cudaFree(&S_d);
  cudat_error2 = cudaFree(&X_ld);


  if ((cudat_error1!=cudaSuccess) || (cudat_error2!=cudaSuccess)) {
    std::cout << "device memory free failed." << std::endl;
    std::terminate();
  }

  MatDenseRestoreArray(X, &X_lh);
}

MultTM::~MultTM() {
  cusolverDnDestroy(cusolverdn_handle);
  cudaStreamDestroy(cusolver_stream);
  cublasDestroy(cublas_handle);
}
