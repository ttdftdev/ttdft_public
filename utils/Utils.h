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

#ifndef TUCKER_TENSOR_KSDFT_UTILS_H
#define TUCKER_TENSOR_KSDFT_UTILS_H

#include <string>
#include <petscmat.h>

namespace Utils {

    void print_current_max_mem(const std::string &str);

    template<typename T>
    int mpi_allgather(std::vector<T> &input,
                      std::vector<T> &output,
                      MPI_Datatype mpi_datatype,
                      MPI_Comm mpi_comm);

    void permute_matrix_rcm(Mat &A,
                            std::vector<int> &rcm_to_tensor,
                            std::vector<int> &tensor_to_rcm,
                            std::vector<int> &rcm_to_tensor_local,
                            std::vector<PetscInt> &dnz,
                            std::vector<PetscInt> &onz);

}

#endif //TUCKER_TENSOR_KSDFT_UTILS_H
