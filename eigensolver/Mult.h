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

#ifndef TTDFT_MULT_H
#define TTDFT_MULT_H

#include "DeviceCommUtils.h"
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <petscmat.h>

class AX {
public:
  /**
   * @brief constructor
   * @param num_bg number of band groups
   * @param num_wfns number of total wavefunctions for later AX multiplication
   */
  AX(int num_bg, int num_wfns);

  /**
   * @brief setting up the system and convert & copy the data to device
   * @param A the PETSc MPIAIJ matrix storing the projected KS Hamiltonian. A
   * has to be symmetric.
   */
  void setup_system(Mat A);

  /**
   * @brief perform A*X operation and return the result in PETSc MATDENSE format
   * @param[in] X matrix X to be multiplied
   * @param[out] AX the result of A*X. This matrix has to be pre-allocated
   * before passed in. This function does NOT allocate and initialize the
   * matrix.
   */
  void perform_ax(Mat &X, Mat &AX);

  /**
   * @brief use for freeing matrix A and reuse the object, internally called by
   * the destructor
   */
  void free_mata_on_device();

  ~AX();

private:
  cusparseHandle_t sparse_handle = 0;
  cusparseMatDescr_t sparse_descr = 0;
  cublasHandle_t blas_handle = 0;

  // the owned rows are [start, end)
  int owned_row_start;
  int owned_row_end;
  int owned_cpu_start;
  int owned_cpu_end;

  int global_mata_m;
  int global_mata_k;
  int local_mata_m;
  int local_mata_k;

  int num_band_groups;
  int num_wfns;
  std::vector<int> owned_wfn_idx;
  int owned_wfn_start;
  int owned_wfn_end;
  IS wfn_row_is, wfn_col_is;

  // some logistic data
  double double_one;
  double double_zero;

  // gpu sparse data
  int nzs;
  int nrs;
  int *ia_d;
  int *ja_d;
  double *a_sparse_arr_d;

  // gpu dense data
  double *a_dense_d;

  DeviceCommUtils device_comm;

  bool is_freed;

  void setup_mata_on_device(Mat &A_seq_sparse, Mat &A_seq_dense);

  void trans_mult(int x_m, int x_n, const double *x, double *ax) const;

  void print_device_double(double *dev, int n,
                           const std::string var_name) const;

  void print_device_int(int *dev, int n, const std::string var_name) const;

  void print_matrices();
};

#endif // TTDFT_MULT_H
