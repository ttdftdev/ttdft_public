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

#include "MatUtils.h"

void Utils::matGetOwnedRowsCols(const Mat &mat,
                                std::vector<int> &idxRow,
                                std::vector<int> &idxCol) {
    IS rows, cols;
    MatGetOwnershipIS(mat,
                      &rows,
                      &cols);

    int rowSize;
    ISGetLocalSize(rows,
                   &rowSize);
    idxRow = std::vector<int>(rowSize,
                              0);
    const int *ptr;
    ISGetIndices(rows,
                 &ptr);
    for (unsigned i = 0; i != rowSize; ++i)
        idxRow[i] = ptr[i];
    ISRestoreIndices(rows,
                     &ptr);

    int colSize;
    ISGetLocalSize(cols,
                   &colSize);
    idxCol = std::vector<int>(colSize,
                              0);
    ISGetIndices(cols,
                 &ptr);
    for (unsigned j = 0; j != colSize; ++j)
        idxCol[j] = ptr[j];
    ISRestoreIndices(cols,
                     &ptr);
}
