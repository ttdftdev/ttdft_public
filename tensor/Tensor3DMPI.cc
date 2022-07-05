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

#include <algorithm>
#include "Tensor3DMPI.h"
#include "TuckerTensor.h"

int Tensor3DMPI::proc_x, Tensor3DMPI::proc_y, Tensor3DMPI::proc_z;

Tensor3DMPI::Tensor3DMPI(int dimx,
                         int dimy,
                         int dimz,
                         MPI_Comm comm)
        : dimx(dimx),
          dimy(dimy),
          dimz(dimz),
          comm(comm) {
    Tucker::SizeArray sizeArray(3);
    sizeArray[0] = dimx;
    sizeArray[1] = dimy;
    sizeArray[2] = dimz;

    int nproc;
    MPI_Comm_size(comm,
                  &nproc);

    // TODO generalize later
    Tucker::SizeArray procArray(3);
    procArray[0] = proc_x;
    procArray[1] = proc_y;
    procArray[2] = proc_z;

    TuckerMPI::Distribution
            *distribution = Tucker::MemoryManager::safe_new<TuckerMPI::Distribution>(sizeArray,
                                                                                     procArray);
    tensor = Tucker::MemoryManager::safe_new<TuckerMPI::Tensor>(distribution);
    tensor->getLocalTensor()->initialize();

    bool squeezed = false;
    const Tucker::SizeArray *offsetX = distribution->getMap(0,
                                                            squeezed)->getOffsets();
    const Tucker::SizeArray *offsetY = distribution->getMap(1,
                                                            squeezed)->getOffsets();
    const Tucker::SizeArray *offsetZ = distribution->getMap(2,
                                                            squeezed)->getOffsets();
    int procGrid[3];
    distribution->getProcessorGrid()->getCoordinates(procGrid);
    index[0] = (*offsetX)[procGrid[0]];
    index[1] = (*offsetX)[procGrid[0] + 1];
    index[2] = (*offsetY)[procGrid[1]];
    index[3] = (*offsetY)[procGrid[1] + 1];
    index[4] = (*offsetZ)[procGrid[2]];
    index[5] = (*offsetZ)[procGrid[2] + 1];
    numLocalEntries = tensor->getLocalNumEntries();
    numGlobalEntries = tensor->getGlobalNumEntries();
}

Tensor3DMPI::Tensor3DMPI(const Tensor3DMPI &inputTensor) {
    index = inputTensor.index;
    comm = inputTensor.comm;

    dimx = inputTensor.dimx;
    dimy = inputTensor.dimy;
    dimz = inputTensor.dimz;

    Tucker::SizeArray sizeArray(3);
    sizeArray[0] = dimx;
    sizeArray[1] = dimy;
    sizeArray[2] = dimz;

    int nproc;
    MPI_Comm_size(comm,
                  &nproc);

    // TODO generalize later
    Tucker::SizeArray procArray(3);
    procArray[0] = proc_x;
    procArray[1] = proc_y;
    procArray[2] = proc_z;

    TuckerMPI::Distribution
            *distribution = Tucker::MemoryManager::safe_new<TuckerMPI::Distribution>(sizeArray,
                                                                                     procArray);
    tensor = Tucker::MemoryManager::safe_new<TuckerMPI::Tensor>(distribution);
    double *localTensor = getLocalData();
    const double *localInputTensor = inputTensor.getLocalData();
    int numEntries = tensor->getLocalNumEntries();

    for (int i = 0; i < numEntries; ++i) {
        localTensor[i] = localInputTensor[i];
    }

    numLocalEntries = tensor->getLocalNumEntries();
    numGlobalEntries = tensor->getGlobalNumEntries();
}

Tensor3DMPI::Tensor3DMPI(int dimx,
                         int dimy,
                         int dimz,
                         MPI_Comm comm,
                         const std::vector<double> &input) : dimx(dimx),
                                                             dimy(dimy),
                                                             dimz(dimz),
                                                             comm(comm) {

    Tucker::SizeArray sz(3);
    sz[0] = dimx;
    sz[1] = dimy;
    sz[2] = dimz;

    int nprocs;
    MPI_Comm_size(comm,
                  &nprocs);
    Tucker::SizeArray procArray(3);
    procArray[0] = proc_x;
    procArray[1] = proc_y;
    procArray[2] = proc_z;


    TuckerMPI::Distribution *dist = Tucker::MemoryManager::safe_new<TuckerMPI::Distribution>(sz,
                                                                                             procArray);
    tensor = Tucker::MemoryManager::safe_new<TuckerMPI::Tensor>(dist);

    bool squeezed = false;
    const Tucker::SizeArray *offsetX = dist->getMap(0,
                                                    squeezed)->getOffsets();
    const Tucker::SizeArray *offsetY = dist->getMap(1,
                                                    squeezed)->getOffsets();
    const Tucker::SizeArray *offsetZ = dist->getMap(2,
                                                    squeezed)->getOffsets();
    int procGrid[3];
    dist->getProcessorGrid()->getCoordinates(procGrid);
    index[0] = (*offsetX)[procGrid[0]];
    index[1] = (*offsetX)[procGrid[0] + 1];
    index[2] = (*offsetY)[procGrid[1]];
    index[3] = (*offsetY)[procGrid[1] + 1];
    index[4] = (*offsetZ)[procGrid[2]];
    index[5] = (*offsetZ)[procGrid[2] + 1];

    numLocalEntries = tensor->getLocalNumEntries();
    numGlobalEntries = tensor->getGlobalNumEntries();

    if (index[4] != index[5]) {
        int start = dimx * dimy * index[4], end = dimx * dimy * index[5];
        double *data = tensor->getLocalTensor()->data();
        std::copy(input.begin() + start,
                  input.begin() + end,
                  data);
    }
}

Tensor3DMPI::Tensor3DMPI(int dimx,
                         int dimy,
                         int dimz,
                         MPI_Comm comm,
                         const double input[]) : dimx(dimx),
                                                 dimy(dimy),
                                                 dimz(dimz),
                                                 comm(comm) {

    Tucker::SizeArray sz(3);
    sz[0] = dimx;
    sz[1] = dimy;
    sz[2] = dimz;

    int nprocs;
    MPI_Comm_size(comm,
                  &nprocs);
    Tucker::SizeArray procArray(3);
    procArray[0] = 1;
    procArray[1] = 1;
    procArray[2] = nprocs;

    TuckerMPI::Distribution *dist = Tucker::MemoryManager::safe_new<TuckerMPI::Distribution>(sz,
                                                                                             procArray);
    tensor = Tucker::MemoryManager::safe_new<TuckerMPI::Tensor>(dist);

    bool squeezed = false;
    const Tucker::SizeArray *offsetX = dist->getMap(0,
                                                    squeezed)->getOffsets();
    const Tucker::SizeArray *offsetY = dist->getMap(1,
                                                    squeezed)->getOffsets();
    const Tucker::SizeArray *offsetZ = dist->getMap(2,
                                                    squeezed)->getOffsets();
    int procGrid[3];
    dist->getProcessorGrid()->getCoordinates(procGrid);
    index[0] = (*offsetX)[procGrid[0]];
    index[1] = (*offsetX)[procGrid[0] + 1];
    index[2] = (*offsetY)[procGrid[1]];
    index[3] = (*offsetY)[procGrid[1] + 1];
    index[4] = (*offsetZ)[procGrid[2]];
    index[5] = (*offsetZ)[procGrid[2] + 1];

    numLocalEntries = tensor->getLocalNumEntries();
    numGlobalEntries = tensor->getGlobalNumEntries();
    if (index[4] != index[5]) {
        int start = dimx * dimy * index[4], end = dimx * dimy * index[5];
        double *data = tensor->getLocalTensor()->data();
        std::copy(input + start,
                  input + end,
                  data);
    }

}

Tensor3DMPI &Tensor3DMPI::operator+=(const TuckerMPI::Tensor *input) {
    if (index[4] != index[5]) {
        double *data = tensor->getLocalTensor()->data();
        const double *inputData = input->getLocalTensor()->data();
        for (int i = 0; i < numLocalEntries; ++i) {
            data[i] += inputData[i];
        }
    }
    return *this;
}

Tensor3DMPI::~Tensor3DMPI() {
    Tucker::MemoryManager::safe_delete(tensor);
}

//Tensor3DMPI &Tensor3DMPI::operator=(const Tensor3DMPI &inputTensor) {
//    return;
//}

double *Tensor3DMPI::getLocalData(std::array<int, 6> &globalIdx) {
    globalIdx = index;
    if (tensor->getLocalNumEntries() != 0) {
        return tensor->getLocalTensor()->data();
    } else {
        return nullptr;
    }
}

const double *Tensor3DMPI::getLocalData(std::array<int, 6> &globalIdx) const {
    globalIdx = index;
    if (tensor->getLocalNumEntries() != 0) {
        return tensor->getLocalTensor()->data();
    } else {
        return nullptr;
    }
}

double *Tensor3DMPI::getLocalData() {
    if (tensor->getLocalNumEntries() != 0) {
        return tensor->getLocalTensor()->data();
    } else {
        return nullptr;
    }
}

const double *Tensor3DMPI::getLocalData() const {
    if (index[4] != index[5]) {
        return tensor->getLocalTensor()->data();
    } else {
        return nullptr;
    }
}

Tensor3DMPI::Tensor3DMPI() {/* do nothing */}

int Tensor3DMPI::getLocalNumberEntries() const {
    return tensor->getLocalNumEntries();
}

int Tensor3DMPI::getGlobalNumberEntries() const {
    return tensor->getGlobalNumEntries();
}

int Tensor3DMPI::getGlobalDimension(int order) const {
    if (order == 0)
        return dimx;
    else if (order == 1)
        return dimy;
    else if (order == 2)
        return dimz;
}

/**
 * @warning the assign operator does not check the equivalence in sizes and distribution between tensors, be sure to initialize the tensors before using assignment operator
 */
Tensor3DMPI &Tensor3DMPI::operator=(const Tensor3DMPI &input) {

    int numLocalNumberEntries = tensor->getLocalNumEntries();

    if (numLocalNumberEntries != 0) {
        double *data = tensor->getLocalTensor()->data();
        const double *inputData = input.tensor->getLocalTensor()->data();

        std::copy(inputData,
                  inputData + numLocalNumberEntries,
                  data);
    }

    return *this;
}

Tensor3DMPI &Tensor3DMPI::operator+(const Tensor3DMPI &input) const {

    int numLocalNumberEntries = tensor->getLocalNumEntries();
    Tensor3DMPI result(dimx,
                       dimy,
                       dimz,
                       comm);

    if (numLocalNumberEntries != 0) {
        const double *data = tensor->getLocalTensor()->data();
        const double *inputData = input.tensor->getLocalTensor()->data();

        double *resultData = result.tensor->getLocalTensor()->data();

        std::transform(data,
                       data + numLocalNumberEntries,
                       inputData,
                       resultData,
                       std::plus<double>());
    }
    return result;
}

Tensor3DMPI &Tensor3DMPI::operator-(const Tensor3DMPI &input) const {

    Tensor3DMPI result(dimx,
                       dimy,
                       dimz,
                       comm);

    int numLocalNumberEntries = tensor->getLocalNumEntries();
    if (numLocalNumberEntries != 0) {
        const double *data = tensor->getLocalTensor()->data();
        const double *inputData = input.tensor->getLocalTensor()->data();

        double *resultData = result.tensor->getLocalTensor()->data();

        std::transform(data,
                       data + numLocalNumberEntries,
                       inputData,
                       resultData,
                       std::minus<double>());
    }
    return result;
}

Tensor3DMPI &Tensor3DMPI::operator*(const Tensor3DMPI &input) const {

    Tensor3DMPI result(dimx,
                       dimy,
                       dimz,
                       comm);

    int numLocalNumberEntries = tensor->getLocalNumEntries();
    if (numLocalNumberEntries != 0) {
        const double *data = tensor->getLocalTensor()->data();
        const double *inputData = input.tensor->getLocalTensor()->data();

        double *resultData = result.tensor->getLocalTensor()->data();

        std::transform(data,
                       data + numLocalNumberEntries,
                       inputData,
                       resultData,
                       std::multiplies<double>());
    }
    return result;
}

Tensor3DMPI &Tensor3DMPI::operator/(const Tensor3DMPI &input) const {

    Tensor3DMPI result(dimx,
                       dimy,
                       dimz,
                       comm);

    int numLocalNumberEntries = tensor->getLocalNumEntries();
    if (numLocalNumberEntries != 0) {
        const double *data = tensor->getLocalTensor()->data();
        const double *inputData = input.tensor->getLocalTensor()->data();

        double *resultData = result.tensor->getLocalTensor()->data();

        std::transform(data,
                       data + numLocalNumberEntries,
                       inputData,
                       resultData,
                       std::divides<double>());
    }
    return result;
}

Tensor3DMPI &Tensor3DMPI::operator+=(const Tensor3DMPI &input) {

    int numLocalNumberEntries = tensor->getLocalNumEntries();
    if (numLocalNumberEntries != 0) {
        double *data = tensor->getLocalTensor()->data();
        const double *inputData = input.tensor->getLocalTensor()->data();

        for (int i = 0; i < numLocalNumberEntries; ++i) {
            data[i] += inputData[i];
        }
    }
    return *this;
}

Tensor3DMPI &Tensor3DMPI::operator-=(const Tensor3DMPI &input) {

    int numLocalNumberEntries = tensor->getLocalNumEntries();
    if (numLocalNumberEntries != 0) {
        double *data = tensor->getLocalTensor()->data();
        const double *inputData = input.tensor->getLocalTensor()->data();

        for (int i = 0; i < numLocalNumberEntries; ++i) {
            data[i] -= inputData[i];
        }
    }
    return *this;
}

Tensor3DMPI &Tensor3DMPI::operator*=(const Tensor3DMPI &input) {

    int numLocalNumberEntries = tensor->getLocalNumEntries();
    if (numLocalNumberEntries != 0) {
        double *data = tensor->getLocalTensor()->data();
        const double *inputData = input.tensor->getLocalTensor()->data();

        for (int i = 0; i < numLocalNumberEntries; ++i) {
            data[i] *= inputData[i];
        }
    }
    return *this;
}

Tensor3DMPI &Tensor3DMPI::operator/=(const Tensor3DMPI &input) {

    int numLocalNumberEntries = tensor->getLocalNumEntries();
    if (numLocalNumberEntries != 0) {
        double *data = tensor->getLocalTensor()->data();
        const double *inputData = input.tensor->getLocalTensor()->data();

        for (int i = 0; i < numLocalNumberEntries; ++i) {
            data[i] /= inputData[i];
        }
    }
    return *this;
}

Tensor3DMPI &Tensor3DMPI::operator+(const double d) const {

    Tensor3DMPI result(dimx,
                       dimy,
                       dimz,
                       comm);

    int numLocalNumberEntries = tensor->getLocalNumEntries();
    if (numLocalNumberEntries != 0) {
        const double *data = tensor->getLocalTensor()->data();

        double *resultData = result.tensor->getLocalTensor()->data();

        for (int i = 0; i < numLocalNumberEntries; ++i) {
            resultData[i] = data[i] + d;
        }
    }
    return result;
}

Tensor3DMPI &Tensor3DMPI::operator-(const double d) const {

    Tensor3DMPI result(dimx,
                       dimy,
                       dimz,
                       comm);

    int numLocalNumberEntries = tensor->getLocalNumEntries();
    if (numLocalNumberEntries != 0) {

        const double *data = tensor->getLocalTensor()->data();

        double *resultData = result.tensor->getLocalTensor()->data();

        for (int i = 0; i < numLocalNumberEntries; ++i) {
            resultData[i] = data[i] - d;
        }
    }
    return result;
}

Tensor3DMPI &Tensor3DMPI::operator*(const double d) const {

    Tensor3DMPI result(dimx,
                       dimy,
                       dimz,
                       comm);

    int numLocalNumberEntries = tensor->getLocalNumEntries();

    if (numLocalNumberEntries != 0) {
        const double *data = tensor->getLocalTensor()->data();

        double *resultData = result.tensor->getLocalTensor()->data();

        for (int i = 0; i < numLocalNumberEntries; ++i) {
            resultData[i] = data[i] * d;
        }
    }
    return result;
}

Tensor3DMPI &Tensor3DMPI::operator/(const double d) const {

    Tensor3DMPI result(dimx,
                       dimy,
                       dimz,
                       comm);

    int numLocalNumberEntries = tensor->getLocalNumEntries();
    if (numLocalNumberEntries != 0) {

        const double *data = tensor->getLocalTensor()->data();

        double *resultData = result.tensor->getLocalTensor()->data();

        for (int i = 0; i < numLocalNumberEntries; ++i) {
            resultData[i] = data[i] / d;
        }
    }
    return result;
}

Tensor3DMPI &Tensor3DMPI::operator+=(const double d) {

    int numLocalNumberEntries = tensor->getLocalNumEntries();
    if (numLocalNumberEntries != 0) {

        double *data = tensor->getLocalTensor()->data();

        for (int i = 0; i < numLocalNumberEntries; ++i) {
            data[i] += d;
        }
    }

    return *this;
}

Tensor3DMPI &Tensor3DMPI::operator-=(const double d) {

    int numLocalNumberEntries = tensor->getLocalNumEntries();
    if (numLocalNumberEntries != 0) {

        double *data = tensor->getLocalTensor()->data();

        for (int i = 0; i < numLocalNumberEntries; ++i) {
            data[i] -= d;
        }
    }
    return *this;
}

Tensor3DMPI &Tensor3DMPI::operator*=(const double d) {

    int numLocalNumberEntries = tensor->getLocalNumEntries();
    if (numLocalNumberEntries != 0) {

        double *data = tensor->getLocalTensor()->data();

        for (int i = 0; i < numLocalNumberEntries; ++i) {
            data[i] *= d;
        }
    }
    return *this;
}

Tensor3DMPI &Tensor3DMPI::operator/=(const double d) {

    int numLocalNumberEntries = tensor->getLocalNumEntries();
    if (numLocalNumberEntries != 0) {

        double *data = tensor->getLocalTensor()->data();

        for (int i = 0; i < numLocalNumberEntries; ++i) {
            data[i] /= d;
        }
    }
    return *this;
}

int Tensor3DMPI::getIstartGlobal() const {
    return index[0];
}

int Tensor3DMPI::getIendGlobal() const {
    return index[1];
}

int Tensor3DMPI::getJstartGlobal() const {
    return index[2];
}

int Tensor3DMPI::getJendGlobal() const {
    return index[3];
}

int Tensor3DMPI::getKstartGlobal() const {
    return index[4];
}

int Tensor3DMPI::getKendGlobal() const {
    return index[5];
}

const std::array<int, 6> &Tensor3DMPI::getGlobalIndex() const {
    return index;
}

void Tensor3DMPI::set_proc(int x,
                           int y,
                           int z) {
    proc_x = x;
    proc_y = y;
    proc_z = z;
}