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
#include "DeviceUtils.cuh"
#include "DeviceCommUtils.h"

DeviceCommUtils::DeviceCommUtils(int n) : num_band_groups(n) {

    MPI_Comm_size(MPI_COMM_WORLD,
                  &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD,
                  &world_rank);

    MPI_Comm_split_type(MPI_COMM_WORLD,
                        MPI_COMM_TYPE_SHARED,
                        0,
                        MPI_INFO_NULL,
                        &nodal_comm);

    MPI_Comm_size(nodal_comm,
                  &nodal_size);
    MPI_Comm_rank(nodal_comm,
                  &nodal_rank);
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

    if (max_nodal_size != min_nodal_size) {
        std::cout << "ERROR: each node should have the same number of cpus." << std::endl;
        std::terminate();
    }

    device_utils::device_get_device_count(local_num_devices);
    if (local_num_devices == 0 || local_num_devices > nodal_size) {
        std::cout << "ERROR: no gpu or gpu is more than cpus on the node (currently not supported.)" << std::endl;
        std::terminate();
    }

    num_tasks_per_device = nodal_size / local_num_devices;
    local_device_id = nodal_rank / num_tasks_per_device;
    device_utils::device_set_device(local_device_id);
    global_num_devices = local_num_devices * (world_size / nodal_size);
    if (global_num_devices % num_band_groups != 0) {
        std::cout << "ERROR: total number of gpus should be divisible by the specified number of bands." << std::endl;
        std::terminate();
    }

    global_device_id = world_rank / num_tasks_per_device;
    device_rank = MPI_UNDEFINED;
    owned_band_block = MPI_UNDEFINED;
    owned_device = false;
    for (int i = 0; i < global_num_devices; ++i) {
        if (world_rank == i * num_tasks_per_device) {
            device_rank = i;
            owned_band_block = i % num_band_groups;
            owned_device = true;
        }
    }


    MPI_Comm_split(MPI_COMM_WORLD,
                   owned_band_block,
                   world_rank,
                   &band_comm);
    int band_comm_size = MPI_UNDEFINED, band_comm_rank = MPI_UNDEFINED;
    MPI_Comm_rank(band_comm,
                  &band_comm_rank);
    MPI_Comm_size(band_comm,
                  &band_comm_size);

    MPI_Comm_split(MPI_COMM_WORLD,
                   world_rank / (num_tasks_per_device * num_band_groups),
                   world_rank,
                   &row_mat_comm);
    MPI_Comm_size(row_mat_comm,
                  &row_mat_size);
    MPI_Comm_rank(row_mat_comm,
                  &row_mat_rank);
#ifndef NDEBUG
    if (world_rank == 0) std::cout << "device count" << std::endl;
    for (int i = 0; i < world_size; ++i) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (i == world_rank) {
            std::cout << "rank " << i << ": (" << band_comm_rank << ", " << band_comm_size << ")" << std::endl;
        }
    }
#endif
}

DeviceCommUtils::~DeviceCommUtils() {
    int mpi_finalized;
    MPI_Finalized(&mpi_finalized);
    if (!mpi_finalized) {
        MPI_Comm_free(&row_mat_comm);
        MPI_Comm_free(&nodal_comm);
        MPI_Comm_free(&band_comm);
    }
}
