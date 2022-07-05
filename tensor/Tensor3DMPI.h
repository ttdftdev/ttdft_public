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

#ifndef TUCKER_TENSOR_KSDFT_TENSOR3DMPI_H
#define TUCKER_TENSOR_KSDFT_TENSOR3DMPI_H

#include <TuckerMPI_Tensor.hpp>
#include <iomanip>
#include <array>

class TuckerTensor;

typedef enum {
    TENSOR_INSERT, TENSOR_ADD, TENSOR_SUBTRACT, TENSOR_MULTIPLY, TENSOR_SQUARE
} TensorOperation;

class Tensor3DMPI {
public:

    Tensor3DMPI(int dimx,
                int dimy,
                int dimz,
                MPI_Comm comm);

    Tensor3DMPI(const Tensor3DMPI &inputTensor);

    Tensor3DMPI(int dimx,
                int dimy,
                int dimz,
                MPI_Comm comm,
                const std::vector<double> &input);

    Tensor3DMPI(int dimx,
                int dimy,
                int dimz,
                MPI_Comm comm,
                const double input[]);

    Tensor3DMPI &operator+=(const TuckerMPI::Tensor *input);

    double *getLocalData(std::array<int, 6> &globalIdx);

    const double *getLocalData(std::array<int, 6> &globalIdx) const;

    double *getLocalData();

    const double *getLocalData() const;

    int getLocalNumberEntries() const;

    int getGlobalNumberEntries() const;

    int getGlobalDimension(int order) const;

    void setEntriesZero() { tensor->getLocalTensor()->initialize(); };

    static void set_proc(int proc_x,
                         int proc_y,
                         int proc_z);

    Tensor3DMPI &operator=(const Tensor3DMPI &input);

    Tensor3DMPI &operator+(const Tensor3DMPI &input) const;

    Tensor3DMPI &operator-(const Tensor3DMPI &input) const;

    Tensor3DMPI &operator*(const Tensor3DMPI &input) const;

    Tensor3DMPI &operator/(const Tensor3DMPI &input) const;

    Tensor3DMPI &operator+=(const Tensor3DMPI &input);

    Tensor3DMPI &operator-=(const Tensor3DMPI &input);

    Tensor3DMPI &operator*=(const Tensor3DMPI &input);

    Tensor3DMPI &operator/=(const Tensor3DMPI &input);

    Tensor3DMPI &operator+(const double d) const;

    Tensor3DMPI &operator-(const double d) const;

    Tensor3DMPI &operator*(const double d) const;

    Tensor3DMPI &operator/(const double d) const;

    Tensor3DMPI &operator+=(const double d);

    Tensor3DMPI &operator-=(const double d);

    Tensor3DMPI &operator*=(const double d);

    Tensor3DMPI &operator/=(const double d);

    // delete when the tests are done
    TuckerMPI::Tensor *getTensor() { return tensor; }

    const TuckerMPI::Tensor *getTensor() const { return tensor; }

    int getIstartGlobal() const;

    int getIendGlobal() const;

    int getJstartGlobal() const;

    int getJendGlobal() const;

    int getKstartGlobal() const;

    int getKendGlobal() const;

    const std::array<int, 6> &getGlobalIndex() const;

    TuckerMPI::Tensor *tensor = nullptr;
    MPI_Comm comm;

    virtual ~Tensor3DMPI();

private:
    static int proc_x, proc_y, proc_z;
    int dimx, dimy, dimz;
    std::array<int, 6> index;
    int numLocalEntries, numGlobalEntries;

    // force the tensor pointer to be initialized
    Tensor3DMPI();
};

#endif //TUCKER_TENSOR_KSDFT_TENSOR3DMPI_H
