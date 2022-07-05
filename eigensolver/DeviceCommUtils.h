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

#ifndef TTDFT_DEVICE_COMM_UTILS_H
#define TTDFT_DEVICE_COMM_UTILS_H

#include <mpi.h>

struct DeviceCommUtils {
public:
    /**
     * @brief device information for MPI communication for mult class
     * @param n number of band groups used to partition GPUs.
     */
    explicit DeviceCommUtils(int n);

    ~DeviceCommUtils();

    int num_band_groups;
    int owned_band_block;

//    MPI_Comm band_sub_comm;
    MPI_Comm row_mat_comm;
    MPI_Comm nodal_comm;
    MPI_Comm band_comm;

    int world_rank, world_size;
    int nodal_rank, nodal_size;
    int row_mat_rank, row_mat_size;
    int band_sub_rank, band_sub_size;
    int gpu_rank, gpu_size;

    int num_tasks_per_device;
    int local_device_id, local_num_devices;
    int global_device_id, global_num_devices;
    int device_rank;

    bool owned_device;
};


#endif //TTDFT_DEVICE_COMM_UTILS_H
