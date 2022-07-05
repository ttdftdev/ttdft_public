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

#ifndef TUCKER_TENSOR_KSDFT_TUCKERTENSOR_H
#define TUCKER_TENSOR_KSDFT_TUCKERTENSOR_H

#include <TuckerMPI_TuckerTensor.hpp>
#include "Tensor3DMPI.h"

class TuckerTensor {
public:
    ~TuckerTensor();

    const TuckerMPI::TuckerTensor *getTTensor();

private:
    const TuckerMPI::TuckerTensor *ttensor = nullptr;
    std::vector<double> seqcore;
};

#endif //TUCKER_TENSOR_KSDFT_TUCKERTENSOR_H
