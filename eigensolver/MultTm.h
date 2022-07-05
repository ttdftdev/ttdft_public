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
#ifndef TTDFT_GPU_EIGENSOLVER_MULTTM_H_
#define TTDFT_GPU_EIGENSOLVER_MULTTM_H_

#include <petscmat.h>
#include <vector>
#include <cublas_v2.h>
#include <cusolverDn.h>

class MultTM {
public:
  MultTM();

  void mult(Mat &X, Mat &Y, std::vector<double> &result);

  void orth(Mat &X);

  virtual ~MultTM();

protected:
  MPI_Comm comm_tm;
  int comm_tm_rank, comm_tm_size;
  int world_rank, world_size;
  int local_num_devices, global_num_devices;

  cublasHandle_t cublas_handle = NULL;
  cusolverDnHandle_t cusolverdn_handle = NULL;
  cudaStream_t cusolver_stream = NULL;

};

#endif // TTDFT_GPU_EIGENSOLVER_MULTTM_H_
