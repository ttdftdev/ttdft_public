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

#include <petscsys.h>
#include <vector>
#include "Utils.h"

namespace Utils {
    void print_current_max_mem(const std::string &str) {
#ifdef CHECK_MEM_IN_COMP
        PetscLogDouble mem;
        PetscMemoryGetCurrentUsage(&mem);
        double max_mem;
        MPI_Reduce(&mem, &max_mem, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        max_mem = max_mem / 1024.0 / 1024.0;
        PetscPrintf(MPI_COMM_WORLD, str.c_str());
        PetscPrintf(MPI_COMM_WORLD, " current max memory across all processors %.4f Mb.\n", max_mem);
#endif //CHECK_MEM_IN_COMP
    }

    template<typename T>
    int mpi_allgather(std::vector<T> &input,
                      std::vector<T> &output,
                      MPI_Datatype mpi_datatype,
                      MPI_Comm mpi_comm) {
        int nprocs, taskid;
        MPI_Comm_size(mpi_comm,
                      &nprocs);
        MPI_Comm_rank(mpi_comm,
                      &taskid);
        std::vector<int> recvs(nprocs,
                               0), displs(nprocs + 1,
                                          0);
        int local_size = input.size();
        MPI_Allgather(&local_size,
                      1,
                      MPI_INT,
                      recvs.data(),
                      1,
                      MPI_INT,
                      mpi_comm);
        for (int i = 0; i < nprocs; ++i) displs[i + 1] = displs[i] + recvs[i];

        int global_size = 0;
        for (int j = 0; j < nprocs; ++j) global_size += recvs[j];
        output.resize(global_size);
        return MPI_Allgatherv(input.data(),
                              local_size,
                              mpi_datatype,
                              output.data(),
                              recvs.data(),
                              displs.data(),
                              mpi_datatype,
                              mpi_comm);
    }

}