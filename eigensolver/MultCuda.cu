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

#include <iostream>
#include <vector>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include "Mult.h"

namespace {
    __global__ void adjust_column(int owned_row_start,
                                  int owned_row_size,
                                  int n,
                                  int *col_idx);
    extern "C" {

    }
}

AX::AX(int num_bg,
       int num_wfns) : double_one(1.0),
                       double_zero(0.0),
                       num_wfns(num_wfns),
                       num_band_groups(num_bg),
                       device_comm(num_bg),
                       owned_row_start(0),
                       owned_row_end(0),
                       owned_cpu_start(0),
                       owned_cpu_end(0),
                       owned_wfn_idx(num_bg + 1,
                                     0),
                       owned_wfn_start(0),
                       owned_wfn_end(0) {

    cusparseStatus_t sparse_status;
    if (device_comm.owned_device) {
        cusparseCreate(&sparse_handle);
        sparse_status = cusparseCreateMatDescr(&sparse_descr);
        if (sparse_status != CUSPARSE_STATUS_SUCCESS) {
            std::cout << "Matrix descriptor initialization failed" << std::endl;
            std::terminate();
        }
        cusparseSetMatType(sparse_descr,
                           CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(sparse_descr,
                                CUSPARSE_INDEX_BASE_ZERO);

        cublasStatus_t blas_stat;
        blas_stat = cublasCreate(&blas_handle);
        if (blas_stat != CUBLAS_STATUS_SUCCESS) {
            std::cout << "CUBLAS Library create handle failed" << std::endl;
            std::terminate();
        }
    }

    std::vector<int> owned_wfns(num_band_groups,
                                num_wfns / num_band_groups);
    for (int i = 0; i < num_wfns % num_band_groups; ++i) {
        owned_wfns[i] += 1;
    }
    for (int i = 0; i < num_band_groups; ++i) {
        owned_wfn_idx[i + 1] = owned_wfn_idx[i] + owned_wfns[i];
    }
    if (device_comm.owned_band_block != MPI_UNDEFINED) {
        owned_wfn_start = owned_wfn_idx[device_comm.owned_band_block];
        owned_wfn_end = owned_wfn_idx[device_comm.owned_band_block + 1];
    }

    int owned_col_size = 0;
    if (device_comm.owned_device) {
        owned_col_size = owned_wfns[device_comm.owned_band_block];
    }
    std::vector<PetscInt> wfn_col_idx_temp(owned_col_size,
                                           owned_wfn_start);
    for (int i = 0; i < wfn_col_idx_temp.size(); ++i) {
        wfn_col_idx_temp[i] += i;
    }
    PetscInt wfn_col_idx_size = owned_wfn_end - owned_wfn_start;
    ISCreateGeneral(MPI_COMM_WORLD,
                    wfn_col_idx_size,
                    wfn_col_idx_temp.data(),
                    PETSC_COPY_VALUES,
                    &wfn_col_is);

}

void AX::setup_system(Mat A) {
    PetscInt global_am, global_an;
    MatGetSize(A,
               &global_am,
               &global_an);
    global_mata_m = global_am;
    global_mata_k = global_an;
    if (global_mata_m != global_mata_k) {
        std::cout << "A should be a square matrix." << std::endl;
        std::terminate();
    }

    const PetscInt *row_ownership;
    MatGetOwnershipRanges(A,
                          &row_ownership);

    int band_comm_size, band_comm_rank;
    MPI_Comm_size(device_comm.band_comm,
                  &band_comm_size);
    MPI_Comm_rank(device_comm.band_comm,
                  &band_comm_rank);
#ifndef NDEBUG
    for (int i = 0; i < device_comm.world_size; ++i) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (i == device_comm.world_rank) {
            std::cout << "rank " << i << ": (" << band_comm_size << ", " << band_comm_rank << ")" << std::endl;
        }
    }
#endif
    if (device_comm.owned_device) {
        owned_cpu_start = band_comm_rank * num_band_groups * device_comm.num_tasks_per_device;
        owned_cpu_end = (band_comm_rank + 1) * num_band_groups * device_comm.num_tasks_per_device;
        owned_row_start = row_ownership[owned_cpu_start];
        owned_row_end = row_ownership[owned_cpu_end];
    }

    local_mata_m = owned_row_end - owned_row_start;
    local_mata_k = global_mata_k;

    std::vector<PetscInt> row_idx_vec(local_mata_m,
                                      owned_row_start);
    std::vector<PetscInt> col_idx_vec(local_mata_k - local_mata_m,
                                      0);
    for (int i = 0; i < local_mata_m; ++i) {
        row_idx_vec[i] += i;
    }
    int col_idx = 0;
    for (int i = 0; i < owned_row_start; ++i) {
        col_idx_vec[col_idx] = i;
        col_idx++;
    }
    for (int i = owned_row_end; i < local_mata_k; ++i) {
        col_idx_vec[col_idx] = i;
        col_idx++;
    }
    IS row_idx_petsc_is, col_idx_petsc_is;
    ISCreateGeneral(MPI_COMM_WORLD,
                    local_mata_m,
                    &row_idx_vec[0],
                    PETSC_COPY_VALUES,
                    &row_idx_petsc_is);
    ISCreateGeneral(MPI_COMM_WORLD,
                    local_mata_m,
                    &row_idx_vec[0],
                    PETSC_COPY_VALUES,
                    &wfn_row_is);
    ISCreateGeneral(MPI_COMM_WORLD,
                    local_mata_k - local_mata_m,
                    &col_idx_vec[0],
                    PETSC_COPY_VALUES,
                    &col_idx_petsc_is);
    Mat *local_mat_dense, *local_mat_sparse;
    MatCreateSubMatrices(A,
                         1,
                         &row_idx_petsc_is,
                         &row_idx_petsc_is,
                         MAT_INITIAL_MATRIX,
                         &local_mat_dense);
    MatConvert(local_mat_dense[0],
               MATDENSE,
               MAT_INPLACE_MATRIX,
               &local_mat_dense[0]);
    MatCreateSubMatrices(A,
                         1,
                         &row_idx_petsc_is,
                         &col_idx_petsc_is,
                         MAT_INITIAL_MATRIX,
                         &local_mat_sparse);


    setup_mata_on_device(local_mat_sparse[0],
                         local_mat_dense[0]);

    is_freed = false;

    MatDestroySubMatrices(1,
                          &local_mat_dense);
    MatDestroySubMatrices(1,
                          &local_mat_sparse);
    ISDestroy(&row_idx_petsc_is);
    ISDestroy(&col_idx_petsc_is);
}

void AX::setup_mata_on_device(Mat &A_seq_sparse,
                              Mat &A_seq_dense) {
    MatType seq_type;
    MatGetType(A_seq_sparse,
               &seq_type);
    if (std::string(seq_type) != "seqaij") {
        std::cout << "wrong matrix type: " << seq_type << "passed into AX::setup_mata_on_device, should be MATSEQAIJ."
                  << std::endl;
        std::terminate();
    }

    MatGetType(A_seq_dense,
               &seq_type);
    if (std::string(seq_type) != "seqdense") {
        std::cout << "wrong matrix: " << seq_type << " passed into AX::setup_mata_on_device, should be MATSEQDENSE."
                  << std::endl;
        std::terminate();
    }

    PetscInt sam, sak;
    MatGetSize(A_seq_sparse,
               &sam,
               &sak);

    if (sam != local_mata_m) {
        std::cout << "wrong matrix size m" << std::endl;
        std::terminate();
    } else if (sak != (local_mata_k - local_mata_m)) {
        std::cout << "wrong matrix size n" << std::endl;
        std::terminate();
    }


    MatInfo A_seq_info;
    MatGetInfo(A_seq_sparse,
               MAT_LOCAL,
               &A_seq_info);
    nzs = A_seq_info.nz_used;

    PetscInt parn;
    const PetscInt *ia;
    const PetscInt *ja;
    PetscBool done;
    MatGetRowIJ(A_seq_sparse,
                0,
                PETSC_FALSE,
                PETSC_FALSE,
                &parn,
                &ia,
                &ja,
                &done);
    if (done != PETSC_TRUE) {
        std::cout << "MatGetRowIJ failed." << std::endl;
        std::terminate();
    }
    nrs = parn;
    std::vector<int> ia_h(ia,
                          ia + nrs + 1);
    std::vector<int> ja_h(ja,
                          ja + nzs);
#ifndef NDEBUG
    for (int i = 0; i < nzs; ++i) {
        ja_h[i] += (owned_row_start <= ja_h[i])*gpu_local_dense_m;
    }
#endif
    MatRestoreRowIJ(A_seq_sparse,
                    0,
                    PETSC_FALSE,
                    PETSC_FALSE,
                    &parn,
                    &ia,
                    &ja,
                    &done);
    if (done != PETSC_TRUE) {
        std::cout << "MatRestoreRowIJ failed." << std::endl;
        std::terminate();
    }

    if (device_comm.owned_device) {
        cudaError_t cudat_error1, cudat_error2;
        cudat_error1 = cudaMalloc(&ia_d,
                                  (nrs + 1) * sizeof(int));
        cudat_error2 = cudaMalloc(&ja_d,
                                  nzs * sizeof(int));
        if ((cudat_error1 != cudaSuccess) ||
            (cudat_error2 != cudaSuccess)) {
            std::cout << "ia_d, ja_d device malloc failed" << std::endl;
            std::terminate();
        }

        cudat_error1 = cudaMemcpy(ia_d,
                                  ia_h.data(),
                                  (nrs + 1) * sizeof(int),
                                  cudaMemcpyHostToDevice);
        cudat_error2 = cudaMemcpy(ja_d,
                                  ja_h.data(),
                                  nzs * sizeof(int),
                                  cudaMemcpyHostToDevice);
        if ((cudat_error1 != cudaSuccess) ||
            (cudat_error2 != cudaSuccess)) {
            std::cout << "ia_d, ja_d device memcpy failed" << std::endl;
            std::terminate();
        }
        adjust_column<<<(nzs + 255) / 256, 256>>>(owned_row_start,
                                                  owned_row_end - owned_row_start,
                                                  nzs,
                                                  ja_d);
    }

    double *a_arr_h;
    MatSeqAIJGetArray(A_seq_sparse,
                      &a_arr_h);

    if (device_comm.owned_device) {
        cudaError_t cudat_error1, cudat_error2;
        cudat_error1 = cudaMalloc(&a_sparse_arr_d,
                                  nzs * sizeof(double));
        cudat_error2 = cudaMemcpy(a_sparse_arr_d,
                                  a_arr_h,
                                  nzs * sizeof(double),
                                  cudaMemcpyHostToDevice);
        if ((cudat_error1 != cudaSuccess) ||
            (cudat_error2 != cudaSuccess)) {
            std::cout << "A_seq_sparse array device malloc/memcpy failed" << std::endl;
            std::terminate();
        }
    }

    MatSeqAIJRestoreArray(A_seq_sparse,
                          &a_arr_h);

    double *a_seq_dense_arr;
    PetscInt dam, dak;
    MatGetSize(A_seq_dense,
               &dam,
               &dak);
    if (dam != dak) {
        std::cout << "non-square size inconsistent for dense mat in AX::setup_mata_on_device." << std::endl;
        std::terminate();
    } else if (dam != local_mata_m) {
        std::cout << "size inconsistent for dense mat m dim in AX::setup_mata_on_device." << std::endl;
        std::terminate();
    }
    MatDenseGetArray(A_seq_dense,
                     &a_seq_dense_arr);

    if (device_comm.owned_device) {
        cudaError_t cudat_error1, cudat_error2;
        cudat_error1 = cudaMalloc(&a_dense_d,
                                  local_mata_m * local_mata_m * sizeof(double));
        cudat_error2 = cudaMemcpy(a_dense_d,
                                  a_seq_dense_arr,
                                  local_mata_m * local_mata_m * sizeof(double),
                                  cudaMemcpyHostToDevice);
        if ((cudat_error1 != cudaSuccess) ||
            (cudat_error2 != cudaSuccess)) {
            std::cout << "A_dense_sparse array device malloc/memcpy failed" << std::endl;
            std::terminate();
        }
    }

    MatDenseRestoreArray(A_seq_dense,
                         &a_seq_dense_arr);
}

void AX::print_matrices() {
    if (is_freed) {
        std::cout << "empty pointer for printing." << std::endl;
        return;
    }

    for (int i = 0; i < device_comm.world_size; ++i) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (i == device_comm.world_rank) {
            if (device_comm.owned_device) {
                std::vector<int> ia(nrs + 1,
                                    0);
                std::vector<int> ja(nzs,
                                    0);
                std::vector<double> arra(nzs,
                                         0);
                std::vector<double> densea(local_mata_m * local_mata_m,
                                           0);
                cudaMemcpy(&ia[0],
                           ia_d,
                           (nrs + 1) * sizeof(int),
                           cudaMemcpyDeviceToHost);
                cudaMemcpy(&ja[0],
                           ja_d,
                           nzs * sizeof(int),
                           cudaMemcpyDeviceToHost);
                cudaMemcpy(&arra[0],
                           a_sparse_arr_d,
                           nzs * sizeof(double),
                           cudaMemcpyDeviceToHost);
                cudaMemcpy(&densea[0],
                           a_dense_d,
                           (local_mata_m * local_mata_m) * sizeof(double),
                           cudaMemcpyDeviceToHost);
                std::cout << "ia: ";
                for (int i: ia) {
                    std::cout << i << ", ";
                }
                std::cout << std::endl;
                std::cout << "ja: ";
                for (int i: ja) {
                    std::cout << i << ", ";
                }
                std::cout << std::endl;
                std::cout << "arr_a: ";
                for (double i: arra) {
                    std::cout << i << ", ";
                }
                std::cout << std::endl;

                std::cout << "dense: " << std::endl;
                for (int i = 0; i < local_mata_m; ++i) {
                    for (int j = 0; j < local_mata_m; ++j) {
                        std::cout << densea[i + j * local_mata_m] << " ";
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
            }
        }
    }
}

void AX::print_device_double(double *dev,
                             int n,
                             const std::string var_name) const {

    for (int i = 0; i < device_comm.world_size; ++i) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (i == device_comm.world_rank) {
            if (device_comm.owned_device) {
                if (dev == nullptr) {
                    std::cout << "empty pointer for printing." << std::endl;
                    std::terminate();
                }
                std::vector<double> temp(n,
                                         0);
                cudaMemcpy(&temp[0],
                           dev,
                           n * sizeof(double),
                           cudaMemcpyDeviceToHost);
                std::cout << var_name << ": ";
                for (const auto &i: temp) {
                    std::cout << i << ", ";
                }
                std::cout << std::endl;
            }
        }
    }
}

void AX::print_device_int(int *dev,
                          int n,
                          const std::string var_name) const {
    for (int i = 0; i < device_comm.world_size; ++i) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (i == device_comm.world_rank) {
            if (device_comm.owned_device) {
                if (dev == nullptr) {
                    std::cout << "empty pointer for printing." << std::endl;
                    std::terminate();
                }
                std::vector<int> temp(n,
                                      0);
                cudaMemcpy(&temp[0],
                           dev,
                           n * sizeof(int),
                           cudaMemcpyDeviceToHost);
                std::cout << var_name << ": ";
                for (const auto &i: temp) {
                    std::cout << i << ", ";
                }
                std::cout << std::endl;
            }
        }
    }
}

void AX::trans_mult(int x_m,
                    int x_n,
                    const double *x,
                    double *ax) const {

    if (x_m != local_mata_m) {
        std::cout << "wrong x dimension for trans_mult" << std::endl;
        std::terminate();
    }

    double *x_d;
    cudaError_t cuda_error;
    cuda_error = cudaMalloc(&x_d,
                            x_m * x_n * sizeof(double));
    if (cuda_error != cudaSuccess) {
        std::cout << "trans_mult malloc x_d failed" << std::endl;
        std::terminate();
    }
    cuda_error = cudaMemcpy(x_d,
                            x,
                            x_m * x_n * sizeof(double),
                            cudaMemcpyHostToDevice);
    if (cuda_error != cudaSuccess) {
        std::cout << "trans_mult memcpy x_d failed" << std::endl;
        std::terminate();
    }

    double *ax_d;
    cuda_error = cudaMalloc(&ax_d,
                            local_mata_k * x_n * sizeof(double));
    if (cuda_error != cudaSuccess) {
        std::cout << "trans_mult malloc ax_d failed" << std::endl;
        std::terminate();
    }

    cusparseStatus_t sparse_status;
#ifndef NDEBUG
    //    This part is moved to the contructor to avoid being called multiple times. Kept here for future investigation.
    //    cusparseHandle_t sparse_handle = 0;
    //    cusparseMatDescr_t sparse_descr = 0;
    //    cusparseCreate(&sparse_handle);
    //    sparse_status = cusparseCreateMatDescr(&sparse_descr);
    //    if (sparse_status != CUSPARSE_STATUS_SUCCESS) {
    //        std::cout << "Matrix descriptor initialization failed" << std::endl;
    //        std::terminate();
    //    }
    //    cusparseSetMatType(sparse_descr,
    //                       CUSPARSE_MATRIX_TYPE_GENERAL);
    //    cusparseSetMatIndexBase(sparse_descr,
    //                            CUSPARSE_INDEX_BASE_ZERO);

        print_device_int(ia_d,
                         nrs + 1,
                         "ia_d");
        print_device_int(ja_d,
                         nzs,
                         "ja_d");
        print_device_double(a_sparse_arr_d,
                            nzs,
                            "a_sparse_arr_d");
        print_device_double(x_d,
                            x_m * x_n,
                            "x_d");
        std::cout << local_mata_m << ", " << x_n << ", " << local_mata_k << std::endl;
#endif
    sparse_status = cusparseDcsrmm(sparse_handle,
                                   CUSPARSE_OPERATION_TRANSPOSE,
                                   local_mata_m,
                                   x_n,
                                   local_mata_k,
                                   nzs,
                                   &double_one,
                                   sparse_descr,
                                   a_sparse_arr_d,
                                   ia_d,
                                   ja_d,
                                   x_d,
                                   local_mata_m,
                                   &double_zero,
                                   ax_d,
                                   local_mata_k);
    if (sparse_status != CUSPARSE_STATUS_SUCCESS) {
        std::cout << "CUSPARSE matrix-matrix multiplication failed with status: " << sparse_status << std::endl;
        std::terminate();
    }

#ifndef NDEBUG
    print_device_double(ax_d,
                        local_mata_k * x_n,
                        "ax_d");


//    This part is moved to the contructor to avoid being called multiple times. Kept here for future investigation.
//    sparse_status = cusparseDestroyMatDescr(sparse_descr);
//    sparse_descr = 0;
//    if (sparse_status != CUSPARSE_STATUS_SUCCESS) {
//        std::cout << "Matrix descriptor destruction failed" << std::endl;
//        std::terminate();
//    }
//
//    sparse_status = cusparseDestroy(sparse_handle);
//    sparse_handle = 0;
//    if (sparse_status != CUSPARSE_STATUS_SUCCESS) {
//        std::cout << "CUSPARSE Library release of resources failed" << std::endl;
//        std::terminate();
//    }

//    cublasStatus_t blas_stat;
//    cublasHandle_t blas_handle = 0;
//
//    blas_stat = cublasCreate(&blas_handle);
//    if (blas_stat != CUBLAS_STATUS_SUCCESS) {
//        std::cout << "CUBLAS Library create handle failed" << std::endl;
//        std::terminate();
//    }

    print_device_double(a_dense_d,
                        local_mata_m * local_mata_m,
                        "a_dense_d");
#endif

    cublasStatus_t blas_stat;
    blas_stat = cublasDgemm(blas_handle,
                            CUBLAS_OP_T,
                            CUBLAS_OP_N,
                            local_mata_m,
                            x_n,
                            local_mata_m,
                            &double_one,
                            a_dense_d,
                            local_mata_m,
                            x_d,
                            local_mata_m,
                            &double_one,
                            &ax_d[owned_row_start],
                            local_mata_k);
    if (blas_stat != CUBLAS_STATUS_SUCCESS) {
        std::cout << "CUBLAS matrix-matrix multiplication failed" << std::endl;
        std::terminate();
    }

// code to transpose result, save for future use.
//    double *axt_d;
//    cuda_error = cudaMalloc(&axt_d,
//                            local_mata_k * x_n * sizeof(double));
//    if (cuda_error != cudaSuccess) {
//        std::cout << "trans_mult malloc axt_d failed" << std::endl;
//        std::terminate();
//    }
//
//    blas_stat = cublasDgeam(blas_handle,
//                            CUBLAS_OP_T,
//                            CUBLAS_OP_N,
//                            x_n,
//                            local_mata_k,
//                            &double_one,
//                            ax_d,
//                            local_mata_k,
//                            &double_zero,
//                            axt_d,
//                            x_n,
//                            axt_d,
//                            x_n);
//    if (blas_stat != CUBLAS_STATUS_SUCCESS) {
//        std::cout << "CUBLAS matrix dgeam failed" << std::endl;
//        std::terminate();
//    }

    cuda_error = cudaMemcpy(ax,
                            ax_d,
                            local_mata_k * x_n * sizeof(double),
                            cudaMemcpyDeviceToHost);
    if (cuda_error != cudaSuccess) {
        std::cout << "trans_mult memcpy axt_d to host failed" << std::endl;
        std::terminate();
    }

    cudaFree(x_d);
//    cudaFree(axt_d);
    cudaFree(ax_d);
}

void AX::free_mata_on_device() {
    cudaFree(ia_d);
    cudaFree(ja_d);
    cudaFree(a_sparse_arr_d);
    cudaFree(a_dense_d);
    is_freed = true;
}

AX::~AX() {

    if (is_freed != true) {
        free_mata_on_device();
    }
    if (device_comm.owned_device) {
        cusparseStatus_t sparse_status;
        sparse_status = cusparseDestroyMatDescr(sparse_descr);
        sparse_descr = 0;
        if (sparse_status != CUSPARSE_STATUS_SUCCESS) {
            std::cout << "Matrix descriptor destruction failed" << std::endl;
            std::terminate();
        }

        sparse_status = cusparseDestroy(sparse_handle);
        sparse_handle = 0;
        if (sparse_status != CUSPARSE_STATUS_SUCCESS) {
            std::cout << "CUSPARSE Library release of resources failed" << std::endl;
            std::terminate();
        }

        cublasStatus_t blas_stat;
        blas_stat = cublasDestroy(blas_handle);
        blas_handle = 0;
        if (blas_stat != CUBLAS_STATUS_SUCCESS) {
            std::cout << "CUBLAS Library release of resources failed" << std::endl;
            std::terminate();
        }
    }
    int mpi_finalized;
    MPI_Finalized(&mpi_finalized);
    if (!mpi_finalized) {
        ISDestroy(&wfn_row_is);
        ISDestroy(&wfn_col_is);
    }
    is_freed = true;
}

void AX::perform_ax(Mat &X,
                    Mat &AX) {

    Mat *sub_X;
    MatCreateSubMatrices(X,
                         1,
                         &wfn_row_is,
                         &wfn_col_is,
                         MAT_INITIAL_MATRIX,
                         &sub_X);
    int x_m = local_mata_m;
    int x_n = owned_wfn_end - owned_wfn_start;
    double *sub_X_ptr;
    MatDenseGetArray(sub_X[0],
                     &sub_X_ptr);

#ifndef NDEBUG
    for (int i = 0; i < device_comm.world_size; ++i) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (i == device_comm.world_rank) {
            std::cout << "rank " << i << ": " << std::endl;
            MatView(sub_X[0],
                    PETSC_VIEWER_STDOUT_SELF);
        }
    }

    Mat AsubX;
    MatCreateSeqDense(MPI_COMM_SELF,
                      local_mata_k,
                      x_n,
                      PETSC_NULL,
                      &AsubX);
    double *AsubX_ptr;
    MatDenseGetArray(AsubX,
                     &AsubX_ptr);
    if (device_comm.owned_device) {
        trans_mult(x_m,
                   x_n,
                   sub_X_ptr,
                   AsubX_ptr);
    }
    MatDenseRestoreArray(sub_X[0],
                         &sub_X_ptr);
    MPI_Allreduce(MPI_IN_PLACE,
                  AsubX_ptr,
                  local_mata_k * x_n,
                  MPI_DOUBLE,
                  MPI_SUM,
                  device_comm.band_comm);

    for (int i = 0; i < device_comm.world_size; ++i) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (i == device_comm.world_rank) {
            std::cout << "rank " << i << ": " << std::endl;
            MatView(AsubX,
                    PETSC_VIEWER_STDOUT_SELF);
        }
    }
    MatDenseRestoreArray(AsubX,
                         &AsubX_ptr);
#endif

    double *A_subX = (double *) malloc(local_mata_k * x_n * sizeof(double));

    if (device_comm.owned_device) {
        trans_mult(x_m,
                   x_n,
                   sub_X_ptr,
                   A_subX);
    }

    MatDenseRestoreArray(sub_X[0],
                         &sub_X_ptr);
    MatDestroySubMatrices(1,
                          &sub_X);
    MPI_Allreduce(MPI_IN_PLACE,
                  A_subX,
                  local_mata_k * x_n,
                  MPI_DOUBLE,
                  MPI_SUM,
                  device_comm.band_comm);

#ifndef NDEBUG
    for (int irank = 0; irank < device_comm.world_size; ++irank) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (irank == device_comm.world_rank && device_comm.owned_device == true) {
            std::cout << "rank: " << irank << std::endl;
            for (int i = 0; i < local_mata_k; ++i) {
                for (int j = 0; j < x_n; ++j) {
                    printf("%.6e ", A_subX[i + j*local_mata_k]);
                }
                std::cout << std::endl;
            }
        }
    }
#endif

//    double *owned_Asub_X = (double *) malloc(x_m*x_n*sizeof(double));



}

namespace {
    __global__ void adjust_column(int owned_row_start,
                                  int owned_row_size,
                                  int n,
                                  int *col_idx) {

        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for (int i = index; i < n; i += stride) {
            col_idx[i] += (owned_row_start <= col_idx[i]) * owned_row_size;
        }
    }
}