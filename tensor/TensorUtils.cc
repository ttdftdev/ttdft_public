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

#include <petsclog.h>
#include "TensorUtils.h"

void TensorUtils::allreduce_tensor(TuckerMPI::Tensor *mpitensor,
                                   Tucker::Tensor *seqtensor) {
    bool squeezed = false;

    // Get the communicator of the mpi tensor
    MPI_Comm comm = mpitensor->getDistribution()->getComm(squeezed);
    int mpirank;
    MPI_Comm_rank(comm,
                  &mpirank);

    const Tucker::SizeArray *offsetX = mpitensor->getDistribution()->getMap(0,
                                                                            squeezed)->getOffsets();
    const Tucker::SizeArray *offsetY = mpitensor->getDistribution()->getMap(1,
                                                                            squeezed)->getOffsets();
    const Tucker::SizeArray *offsetZ = mpitensor->getDistribution()->getMap(2,
                                                                            squeezed)->getOffsets();
    int procGrid[3];
    mpitensor->getDistribution()->getProcessorGrid()->getCoordinates(procGrid);
    int idx_x0 = (*offsetX)[procGrid[0]], idx_x1 = (*offsetX)[procGrid[0] + 1];
    int idx_y0 = (*offsetY)[procGrid[1]], idx_y1 = (*offsetY)[procGrid[1] + 1];
    int idx_z0 = (*offsetZ)[procGrid[2]], idx_z1 = (*offsetZ)[procGrid[2] + 1];

    int global_size_x = mpitensor->getGlobalSize(0);
    int global_size_y = mpitensor->getGlobalSize(1);
    int global_size_z = mpitensor->getGlobalSize(2);

    seqtensor->initialize();
    double *seqtensor_data = seqtensor->data();
    double *mpitensor_data = nullptr;
    if (mpitensor->getLocalNumEntries() != 0) mpitensor_data = mpitensor->getLocalTensor()->data();
    int cnt = 0;
    for (int k = idx_z0; k < idx_z1; ++k) {
        for (int j = idx_y0; j < idx_y1; ++j) {
            for (int i = idx_x0; i < idx_x1; ++i) {
                seqtensor_data[i + j * global_size_x + k * global_size_x * global_size_y] = mpitensor_data[cnt];
                cnt++;
            }
        }
    }

    int num_global_entries = mpitensor->getGlobalNumEntries();
    MPI_Allreduce(MPI_IN_PLACE,
                  seqtensor_data,
                  num_global_entries,
                  MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD);
}

void TensorUtils::allgatherTensor(TuckerMPI::Tensor *mpitensor,
                                  Tucker::Tensor *seqtensor) {
    bool squeezed = false;

    // Get the communicator of the mpi tensor
    MPI_Comm comm = mpitensor->getDistribution()->getComm(squeezed);
    int mpirank;
    MPI_Comm_rank(comm,
                  &mpirank);

    // Get the number of element on each processor in each direction
    const Tucker::SizeArray *elementsPerProcX = mpitensor->getDistribution()->getMap(0,
                                                                                     squeezed)->getNumElementsPerProc();
    const Tucker::SizeArray *elementsPerProcY = mpitensor->getDistribution()->getMap(1,
                                                                                     squeezed)->getNumElementsPerProc();
    const Tucker::SizeArray *elementsPerProcZ = mpitensor->getDistribution()->getMap(2,
                                                                                     squeezed)->getNumElementsPerProc();

    // Get the number of processor in each direction
    int nprocsgrid[3] = {mpitensor->getDistribution()->getProcessorGrid()->getNumProcs(0,
                                                                                       squeezed),
                         mpitensor->getDistribution()->getProcessorGrid()->getNumProcs(1,
                                                                                       squeezed),
                         mpitensor->getDistribution()->getProcessorGrid()->getNumProcs(2,
                                                                                       squeezed)};


    // Get the total number of processors in the communicator and allocate an array for displacement array used by MPI_Allgatherv
    int nprocs;
    MPI_Comm_size(mpitensor->getDistribution()->getComm(squeezed),
                  &nprocs);
    int *displ = new int[nprocs];
    int *recvcounts = new int[nprocs];

    for (int iproc = 0; iproc < nprocsgrid[0]; ++iproc) {
        for (int jproc = 0; jproc < nprocsgrid[1]; ++jproc) {
            for (int kproc = 0; kproc < nprocsgrid[2]; ++kproc) {
                int procCoord[3] = {iproc, jproc, kproc};
                int procRank = mpitensor->getDistribution()->getProcessorGrid()->getRank(procCoord);
                recvcounts[procRank] =
                        (*elementsPerProcX)[iproc] * (*elementsPerProcY)[jproc] * (*elementsPerProcZ)[kproc];
            }
        }
    }

    // compute the displacement array for use of allgatherv
    displ[0] = 0;
    for (int idx = 1; idx < nprocs; ++idx) {
        displ[idx] = displ[idx - 1] + recvcounts[idx - 1];
    }

    // if there is no data in the local tensor, it will intrigue an error, so if there is no element in the local tensor, just set the pointer to be a null pointer
    int localCount = mpitensor->getLocalNumEntries();
    double *sendbuff;
    if (localCount > 0)
        sendbuff = mpitensor->getLocalTensor()->data();
    else
        sendbuff = nullptr;

    // recvbuff is a buffer storing data column wise on each element and strings those in processors' order
    double *recvbuff = new double[mpitensor->getGlobalNumEntries()];

    MPI_Allgatherv(sendbuff,
                   localCount,
                   MPI_DOUBLE,
                   recvbuff,
                   recvcounts,
                   displ,
                   MPI_DOUBLE,
                   comm);


    // Start to redistribute data after allgatherv
    // get the index offset of entries on each direction
    // e.g. [0 1 2 2] for 2 entries distributed on 3 processors in x, y, or z direction
    const Tucker::SizeArray *eleOffsetX = mpitensor->getDistribution()->getMap(0,
                                                                               squeezed)->getOffsets();
    const Tucker::SizeArray *eleOffsetY = mpitensor->getDistribution()->getMap(1,
                                                                               squeezed)->getOffsets();
    const Tucker::SizeArray *eleOffsetZ = mpitensor->getDistribution()->getMap(2,
                                                                               squeezed)->getOffsets();

    // get global number of entries on each direction
    int globalSizeX = mpitensor->getGlobalSize(0);
    int globalSizeY = mpitensor->getGlobalSize(1);
    int globalSizeZ = mpitensor->getGlobalSize(2);

    for (int kproc = 0; kproc < nprocsgrid[2]; ++kproc) {
        for (int jproc = 0; jproc < nprocsgrid[1]; ++jproc) {
            for (int iproc = 0; iproc < nprocsgrid[0]; ++iproc) {
                // loop over cartesian coordinate of processor grid
                int procCoord[3] = {iproc, jproc, kproc};
                // get MPI rank of the procesor on grid (iproc, jproc, kproc)
                int procRank = mpitensor->getDistribution()->getProcessorGrid()->getRank(procCoord);
                int offset = displ[procRank];

                // copy elements from recvbuffer to the correct location in global sense
                for (int klocal = 0; klocal < (*elementsPerProcZ)[kproc]; ++klocal) {
                    for (int jlocal = 0; jlocal < (*elementsPerProcY)[jproc]; ++jlocal) {
                        for (int ilocal = 0; ilocal < (*elementsPerProcX)[iproc]; ++ilocal) {
                            int localOffset = offset + ilocal + jlocal * (*elementsPerProcX)[iproc] +
                                              klocal * (*elementsPerProcX)[iproc] * (*elementsPerProcY)[jproc];
                            int globalOffset = ((*eleOffsetX)[iproc] + ilocal) +
                                               ((*eleOffsetY)[jproc] + jlocal) * globalSizeX +
                                               ((*eleOffsetZ)[kproc] + klocal) * globalSizeX * globalSizeY;
                            seqtensor->data()[globalOffset] = recvbuff[localOffset];
                        }
                    }
                }
            }
        }
    }

    // release the memeory
    delete[] recvbuff;
    delete[] displ;
    delete[] recvcounts;

}

void TensorUtils::factorMatGrid2Quad(const FEM &femX,
                                     const FEM &femY,
                                     const FEM &femZ,
                                     const TuckerMPI::TuckerTensor *gridTTensor,
                                     TuckerMPI::TuckerTensor *quadTTensor) {

    const Tucker::Matrix *gridU[3];
    gridU[0] = gridTTensor->U[0];
    gridU[1] = gridTTensor->U[1];
    gridU[2] = gridTTensor->U[2];

    int gridRows[3] = {gridU[0]->nrows(), gridU[1]->nrows(), gridU[2]->nrows()};
    // gridCols is equivalent to the decomposing rank
    int gridCols[3] = {gridU[0]->ncols(), gridU[1]->ncols(), gridU[2]->ncols()};

    const FEM *fem[3] = {&femX, &femY, &femZ};

    quadTTensor->G = Tucker::MemoryManager::safe_new<TuckerMPI::Tensor>(gridTTensor->G->getDistribution());
    for (int idx = 0; idx < gridTTensor->G->getLocalNumEntries(); ++idx) {
        quadTTensor->G->getLocalTensor()->data()[idx] = gridTTensor->G->getLocalTensor()->data()[idx];
    }

    const int quadElectro[3] = {femX.getNumberElements() * femX.getNumberQuadPointsPerElement(),
                                femY.getNumberElements() * femY.getNumberQuadPointsPerElement(),
                                femZ.getNumberElements() * femZ.getNumberQuadPointsPerElement()};

    quadTTensor->U[0] = Tucker::MemoryManager::safe_new<Tucker::Matrix>(quadElectro[0],
                                                                        gridCols[0]);
    quadTTensor->U[1] = Tucker::MemoryManager::safe_new<Tucker::Matrix>(quadElectro[1],
                                                                        gridCols[1]);
    quadTTensor->U[2] = Tucker::MemoryManager::safe_new<Tucker::Matrix>(quadElectro[2],
                                                                        gridCols[2]);

    for (int N = 0; N < 3; ++N) {
        std::vector<double> nodalVal(gridRows[N]);
        for (int i = 0; i < gridCols[N]; ++i) {
            std::vector<double> quadVal;
            std::vector<double> quadDiffVal;
            std::copy(gridU[N]->data() + i * nodalVal.size(),
                      gridU[N]->data() + (i + 1) * nodalVal.size(),
                      nodalVal.begin());
            fem[N]->computeFieldAndDiffFieldAtAllQuadPoints(nodalVal,
                                                            quadVal,
                                                            quadDiffVal);
            std::copy(quadVal.begin(),
                      quadVal.end(),
                      quadTTensor->U[N]->data() + i * quadElectro[N]);
        }
    }
}

const TuckerMPI::TuckerTensor *TensorUtils::computeSTHOSVDonQuadMPI(const FEM &quadFemX,
                                                                    const FEM &quadFemY,
                                                                    const FEM &quadFemZ,
                                                                    const int decompRankX,
                                                                    const int decompRankY,
                                                                    const int decompRankZ,
                                                                    Tensor3DMPI &nodalField) {

    Tucker::SizeArray gridSize(3), rank(3);
    gridSize[0] = nodalField.getGlobalDimension(0);
    gridSize[1] = nodalField.getGlobalDimension(1);
    gridSize[2] = nodalField.getGlobalDimension(2);
    rank[0] = decompRankX;
    rank[1] = decompRankY;
    rank[2] = decompRankZ;

    const TuckerMPI::TuckerTensor *decomposedRhoNodal = TuckerMPI::STHOSVD(nodalField.getTensor(),
                                                                           &rank,
                                                                           false,
                                                                           false);

    int numQuadPointsX = quadFemX.getTotalNumberQuadPoints();
    int numQuadPointsY = quadFemY.getTotalNumberQuadPoints();
    int numQuadPointsZ = quadFemZ.getTotalNumberQuadPoints();
    int numNodesX = gridSize[0];
    int numNodesY = gridSize[1];
    int numNodesZ = gridSize[2];

    Tucker::Matrix *matX = Tucker::MemoryManager::safe_new<Tucker::Matrix>(numQuadPointsX,
                                                                           decompRankX);
    Tucker::Matrix *matY = Tucker::MemoryManager::safe_new<Tucker::Matrix>(numQuadPointsY,
                                                                           decompRankY);
    Tucker::Matrix *matZ = Tucker::MemoryManager::safe_new<Tucker::Matrix>(numQuadPointsZ,
                                                                           decompRankZ);

    double *matXData = matX->data();
    double *matYData = matY->data();
    double *matZData = matZ->data();
    double *matNodalX = decomposedRhoNodal->U[0]->data();
    double *matNodalY = decomposedRhoNodal->U[1]->data();
    double *matNodalZ = decomposedRhoNodal->U[2]->data();

    for (int i = 0; i < decompRankX; ++i) {
        std::vector<double> nodalVal(gridSize[0]);
        std::vector<double> gridVal(numQuadPointsX), gridDiffVal(numQuadPointsX);
        std::copy(matNodalX + i * numNodesX,
                  matNodalX + (i + 1) * numNodesX,
                  nodalVal.begin());
        quadFemX.computeFieldAndDiffFieldAtAllQuadPoints(nodalVal,
                                                         gridVal,
                                                         gridDiffVal);
        std::copy(gridVal.begin(),
                  gridVal.end(),
                  matXData + i * numQuadPointsX);
    }
    for (int i = 0; i < decompRankY; ++i) {
        std::vector<double> nodalVal(gridSize[1]);
        std::vector<double> gridVal(numQuadPointsY), gridDiffVal(numQuadPointsY);
        std::copy(matNodalY + i * numNodesY,
                  matNodalY + (i + 1) * numNodesY,
                  nodalVal.begin());
        quadFemY.computeFieldAndDiffFieldAtAllQuadPoints(nodalVal,
                                                         gridVal,
                                                         gridDiffVal);
        std::copy(gridVal.begin(),
                  gridVal.end(),
                  matYData + i * numQuadPointsY);
    }
    for (int i = 0; i < decompRankZ; ++i) {
        std::vector<double> nodalVal(gridSize[2]);
        std::vector<double> gridVal(numQuadPointsZ), gridDiffVal(numQuadPointsZ);
        std::copy(matNodalZ + i * numNodesZ,
                  matNodalZ + (i + 1) * numNodesZ,
                  nodalVal.begin());
        quadFemZ.computeFieldAndDiffFieldAtAllQuadPoints(nodalVal,
                                                         gridVal,
                                                         gridDiffVal);
        std::copy(gridVal.begin(),
                  gridVal.end(),
                  matZData + i * numQuadPointsZ);
    }

    Tucker::MemoryManager::safe_delete(decomposedRhoNodal->U[0]);
    Tucker::MemoryManager::safe_delete(decomposedRhoNodal->U[1]);
    Tucker::MemoryManager::safe_delete(decomposedRhoNodal->U[2]);

    decomposedRhoNodal->U[0] = matX;
    decomposedRhoNodal->U[1] = matY;
    decomposedRhoNodal->U[2] = matZ;

    return decomposedRhoNodal;
}

void TensorUtils::AllGatherData(int number_local_entries,
                                double *send_buffer,
                                double *receive_buffer) {
    int nproc;
    MPI_Comm_size(MPI_COMM_WORLD,
                  &nproc);
    int sendcnt = number_local_entries;
    int *recvcnt = new int[nproc];
    MPI_Allgather(&sendcnt,
                  1,
                  MPI_INT,
                  recvcnt,
                  1,
                  MPI_INT,
                  MPI_COMM_WORLD);
    int *displ = new int[nproc];
    displ[0] = 0;
    for (int i = 1; i < nproc; ++i) {
        displ[i] = displ[i - 1] + recvcnt[i - 1];
    }
    MPI_Allgatherv(send_buffer,
                   sendcnt,
                   MPI_DOUBLE,
                   receive_buffer,
                   recvcnt,
                   displ,
                   MPI_DOUBLE,
                   MPI_COMM_WORLD);
    delete[] recvcnt;
    delete[] displ;

}

void TensorUtils::AllGather3DMPITensor(Tensor3DMPI &tensor,
                                       std::vector<double> &seq_tensor_data) {
    seq_tensor_data = std::vector<double>(tensor.getGlobalNumberEntries());
    int sendcnt = tensor.getLocalNumberEntries();
    AllGatherData(sendcnt,
                  tensor.getLocalData(),
                  seq_tensor_data.data());
}

void TensorUtils::AllGather3DMPITensor(Tensor3DMPI &tensor,
                                       double *seq_tensor_data) {
    int sendcnt = tensor.getLocalNumberEntries();
    AllGatherData(sendcnt,
                  tensor.getLocalData(),
                  seq_tensor_data);
}

// the subroutine somehow deletes the reconstructed tensor after the subroutine ended, don't use this until the bug is fixed
void TensorUtils::ReconstructSeqTuckerTensor(Tucker::Tensor *core,
                                             Tucker::Matrix *Ux,
                                             Tucker::Matrix *Uy,
                                             Tucker::Matrix *Uz,
                                             Tucker::Tensor *reconstructed_tensor) {
    Tucker::Tensor *temp;
    temp = core;
    reconstructed_tensor = Tucker::ttm(temp,
                                       0,
                                       Ux);
    temp = reconstructed_tensor;
    reconstructed_tensor = Tucker::ttm(temp,
                                       1,
                                       Uy);
    Tucker::MemoryManager::safe_delete(temp);
    temp = reconstructed_tensor;
    reconstructed_tensor = Tucker::ttm(temp,
                                       2,
                                       Uz);
    Tucker::MemoryManager::safe_delete(temp);
}
