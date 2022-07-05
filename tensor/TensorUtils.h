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

#ifndef TUCKER_TENSOR_KSDFT_TUCKERUTILS_H
#define TUCKER_TENSOR_KSDFT_TUCKERUTILS_H

#include <TuckerMPI.hpp>
#include "../fem/FEM.h"
#include "Tensor3DMPI.h"

namespace TensorUtils {
    void allgatherTensor(TuckerMPI::Tensor *mpitensor,
                         Tucker::Tensor *seqtensor);

    void allreduce_tensor(TuckerMPI::Tensor *mpitensor,
                          Tucker::Tensor *seqtensor);

    void factorMatGrid2Quad(const FEM &femX,
                            const FEM &femY,
                            const FEM &femZ,
                            const TuckerMPI::TuckerTensor *gridTTensor,
                            TuckerMPI::TuckerTensor *quadTTensor);

    const TuckerMPI::TuckerTensor *computeSTHOSVDonQuadMPI(const FEM &quadFemX,
                                                           const FEM &quadFemY,
                                                           const FEM &quadFemZ,
                                                           const int decompRankX,
                                                           const int decompRankY,
                                                           const int decompRankZ,
                                                           Tensor3DMPI &nodalField);

    void AllGather3DMPITensor(Tensor3DMPI &tensor,
                              std::vector<double> &seq_tensor_data);

    void AllGather3DMPITensor(Tensor3DMPI &tensor,
                              double *seq_tensor_data);

    void AllGatherData(int number_local_entries,
                       double *send_buffer,
                       double *receive_buffer);

    void ReconstructSeqTuckerTensor(Tucker::Tensor *core,
                                    Tucker::Matrix *Ux,
                                    Tucker::Matrix *Uy,
                                    Tucker::Matrix *Uz,
                                    Tucker::Tensor *reconstructed_tensor);
}

#endif //TUCKER_TENSOR_KSDFT_TUCKERUTILS_H
