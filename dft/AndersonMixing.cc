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

#include <numeric>
#include <cmath>
#include <petscsys.h>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <boost/filesystem.hpp>
#include "AndersonMixing.h"
#include "../blas_lapack/clinalg.h"

extern "C" {
int dgesv_(int *n,
           int *nrhs,
           double *a,
           int *lda,
           int *ipiv,
           double *b,
           int *ldb,
           int *info);
int dsysv_(char *uplo,
           int *n,
           int *nrhs,
           double *a,
           int *lda,
           int *ipiv,
           double *b,
           int *ldb,
           double *work,
           int *lwork,
           int *info);
}

AndersonMixing::AndersonMixing(const FEM &femX,
                               const FEM &femY,
                               const FEM &femZ,
                               const Tensor3DMPI &jacob3DMat,
                               const Tensor3DMPI &weight3DMat,
                               const std::string &rho_path,
                               const double alpha,
                               const int maxIterHistory,
                               bool restart,
                               bool keep_older_history) : femX(femX),
                                                          femY(femY),
                                                          femZ(femZ),
                                                          jacob3DMat(jacob3DMat),
                                                          weight3DMat(weight3DMat),
                                                          rho_path(rho_path),
                                                          alpha(alpha),
                                                          maxIterHistory(maxIterHistory),
                                                          history_offset(0),
                                                          keep_older_history(keep_older_history) {
    int taskId;
    MPI_Comm_rank(MPI_COMM_WORLD,
                  &taskId);
    rho_nodal_in_folder = rho_path + "/" + rho_nodal_in_prefix;
    rho_nodal_out_folder = rho_path + "/" + rho_nodal_out_prefix;
    rho_quad_in_folder = rho_path + "/" + rho_quad_in_prefix;
    rho_quad_out_folder = rho_path + "/" + rho_quad_out_prefix;
    boost::filesystem::create_directories(rho_nodal_in_folder);
    boost::filesystem::create_directories(rho_nodal_out_folder);
    boost::filesystem::create_directories(rho_quad_in_folder);
    boost::filesystem::create_directories(rho_quad_out_folder);

    if (restart == true) {
        // read in iters to be loaded
        std::ifstream fin(rho_path + "/history_scf.txt");
        int temp_scf;
        std::vector<int> scfs;
        while (fin >> temp_scf) scfs.emplace_back(temp_scf);
        if (taskId == 0) {
            std::cout << "loaded scfs: ";
            for (int i = 0; i < scfs.size(); ++i) std::cout << scfs[i] << ", ";
            std::cout << std::endl;
        }
        int number_nodes_x = femX.getTotalNumberNodes();
        int number_nodes_y = femY.getTotalNumberNodes();
        int number_nodes_z = femZ.getTotalNumberNodes();
        int number_quads_x = femX.getTotalNumberQuadPoints();
        int number_quads_y = femY.getTotalNumberQuadPoints();
        int number_quads_z = femZ.getTotalNumberQuadPoints();
        // read in rho's
        for (int scfIter = 0; scfIter < scfs.size(); ++scfIter) {
            int current_scf = scfs[scfIter];
            std::string rho_nodal_in =
                    rho_nodal_in_prefix + "_scf" + std::to_string(current_scf) + "_proc" + std::to_string(taskId);
            std::string rho_nodal_out =
                    rho_nodal_out_prefix + "_scf" + std::to_string(current_scf) + "_proc" + std::to_string(taskId);
            std::string rho_quad_in =
                    rho_quad_in_prefix + "_scf" + std::to_string(current_scf) + "_proc" + std::to_string(taskId);
            std::string rho_quad_out =
                    rho_quad_out_prefix + "_scf" + std::to_string(current_scf) + "_proc" + std::to_string(taskId);
            Tensor3DMPI rho_node_in_temp(number_nodes_x,
                                         number_nodes_y,
                                         number_nodes_z,
                                         MPI_COMM_WORLD);
            Tensor3DMPI rho_node_out_temp(number_nodes_x,
                                          number_nodes_y,
                                          number_nodes_z,
                                          MPI_COMM_WORLD);
            Tensor3DMPI rho_quad_in_temp(number_quads_x,
                                         number_quads_y,
                                         number_quads_z,
                                         MPI_COMM_WORLD);
            Tensor3DMPI rho_quad_out_temp(number_quads_x,
                                          number_quads_y,
                                          number_quads_z,
                                          MPI_COMM_WORLD);
            read_rho(rho_nodal_in_folder,
                     rho_nodal_in,
                     rho_node_in_temp);
            read_rho(rho_nodal_out_folder,
                     rho_nodal_out,
                     rho_node_out_temp);
            read_rho(rho_quad_in_folder,
                     rho_quad_in,
                     rho_quad_in_temp);
            read_rho(rho_quad_out_folder,
                     rho_quad_out,
                     rho_quad_out_temp);
            vectorNodalRhoIn.emplace_back(rho_node_in_temp);
            vectorGridRhoIn.emplace_back(rho_quad_in_temp);
            vectorNodalRhoOut.emplace_back(rho_node_out_temp);
            vectorGridRhoOut.emplace_back(rho_quad_out_temp);

        }
    }
}

void AndersonMixing::computeMixingConstants(const std::deque<Tensor3DMPI> &vectorRhoIn,
                                            const std::deque<Tensor3DMPI> &vectorRhoOut,
                                            std::vector<double> &mixingConstants) {

    const double *jacob3DMatLocal = jacob3DMat.getLocalData();
    const double *weight3DMatLocal = weight3DMat.getLocalData();

    // compute mixing constants
    int sizeHistory = vectorRhoIn.size();
    mixingConstants.resize(sizeHistory - 1);
    std::vector<double> coeffMatrix((sizeHistory - 1) * (sizeHistory - 1),
                                    0.0);
    std::vector<double> rhsvector(sizeHistory - 1,
                                  0.0);

    const double *rhoInn = vectorRhoIn[sizeHistory - 1].getLocalData();
    const double *rhoOutn = vectorRhoOut[sizeHistory - 1].getLocalData();

    int rhoLocalNumEle = vectorRhoIn[sizeHistory - 1].getLocalNumberEntries();

    std::vector<double> Fn(rhoLocalNumEle,
                           0.0);

    for (int i = 0; i < rhoLocalNumEle; ++i) {
        Fn[i] = rhoOutn[i] - rhoInn[i];
    }

    int taskId;
    MPI_Comm_rank(MPI_COMM_WORLD,
                  &taskId);

    for (int m = 0; m < sizeHistory - 1; ++m) {
        std::vector<double> Fnm(rhoLocalNumEle,
                                0.0);
        const double *localRhoOutm2 = vectorRhoOut[sizeHistory - m - 2].getLocalData();
        const double *localRhoInm2 = vectorRhoIn[sizeHistory - m - 2].getLocalData();
        for (int i = 0; i < rhoLocalNumEle; ++i) {
            Fnm[i] = localRhoOutm2[i] - localRhoInm2[i];
        }
        std::vector<double> diffnm = Fn;
        for (int i = 0; i < rhoLocalNumEle; ++i) {
            diffnm[i] -= Fnm[i];
        }

        for (int k = 0; k < sizeHistory - 1; ++k) {
            std::vector<double> Fnk(rhoLocalNumEle,
                                    0.0);
            const double *localRhoOutk2 = vectorRhoOut[sizeHistory - k - 2].getLocalData();
            const double *localRhoInk2 = vectorRhoIn[sizeHistory - k - 2].getLocalData();
            for (int i = 0; i < rhoLocalNumEle; ++i) {
                Fnk[i] = localRhoOutk2[i] - localRhoInk2[i];
            }
            std::vector<double> diffnk = Fn;
            for (int i = 0; i < rhoLocalNumEle; ++i) {
                diffnk[i] -= Fnk[i];
            }
            std::vector<double> coeffmk(rhoLocalNumEle,
                                        0.0);
            for (int i = 0; i < rhoLocalNumEle; ++i) {
                coeffmk[i] = diffnk[i] * diffnm[i] * jacob3DMatLocal[i] * weight3DMatLocal[i];
            }
            coeffMatrix[m + k * (sizeHistory - 1)] = std::accumulate(coeffmk.begin(),
                                                                     coeffmk.end(),
                                                                     0.0);
        }
        std::vector<double> rhsm(rhoLocalNumEle,
                                 0.0);
        for (int i = 0; i < rhoLocalNumEle; ++i) {
            rhsm[i] = diffnm[i] * Fn[i] * jacob3DMatLocal[i] * weight3DMatLocal[i];
        }
        rhsvector[m] = std::accumulate(rhsm.begin(),
                                       rhsm.end(),
                                       0.0);
    }

    std::vector<double> reducedCoeffMat(coeffMatrix.size());
    std::vector<double> reducedRHS(rhsvector.size());
    int rootId = 0;
    MPI_Reduce(&coeffMatrix[0],
               &reducedCoeffMat[0],
               coeffMatrix.size(),
               MPI_DOUBLE,
               MPI_SUM,
               rootId,
               MPI_COMM_WORLD);
    MPI_Reduce(&rhsvector[0],
               &reducedRHS[0],
               rhsvector.size(),
               MPI_DOUBLE,
               MPI_SUM,
               rootId,
               MPI_COMM_WORLD);

    if (taskId == rootId) {
        int n = sizeHistory - 1;
        int nrhs = 1;
        std::vector<int> ipiv(n);
        int info;
        clinalg::dgesv_(n,
                        1,
                        reducedCoeffMat.data(),
                        n,
                        ipiv.data(),
                        reducedRHS.data(),
                        n,
                        &info);
        std::copy(reducedRHS.begin(),
                  reducedRHS.end(),
                  mixingConstants.begin());
    }
    MPI_Bcast(&mixingConstants[0],
              sizeHistory - 1,
              MPI_DOUBLE,
              rootId,
              MPI_COMM_WORLD);
}

void AndersonMixing::computeRhoIn(const std::deque<Tensor3DMPI> &vectorRhoIn,
                                  const std::deque<Tensor3DMPI> &vectorRhoOut,
                                  const std::vector<double> &mixingConstants,
                                  Tensor3DMPI &rhoIn) {
    int sizeHistory = vectorRhoIn.size();

    const double *rhoInnLocal = vectorRhoIn[sizeHistory - 1].getLocalData();
    const double *rhoOutnLocal = vectorRhoOut[sizeHistory - 1].getLocalData();

    int rhoDimX = vectorRhoIn[sizeHistory - 1].getGlobalDimension(0);
    int rhoDimY = vectorRhoIn[sizeHistory - 1].getGlobalDimension(1);
    int rhoDimZ = vectorRhoIn[sizeHistory - 1].getGlobalDimension(2);

    Tensor3DMPI inputChargeValues(rhoDimX,
                                  rhoDimY,
                                  rhoDimZ,
                                  MPI_COMM_WORLD);
    double *inputChargeValuesLocal = inputChargeValues.getLocalData();

    Tensor3DMPI outputChargeValues(rhoDimX,
                                   rhoDimY,
                                   rhoDimZ,
                                   MPI_COMM_WORLD);
    double *outputChargeValuesLocal = outputChargeValues.getLocalData();

    int localNumElements = inputChargeValues.getLocalNumberEntries();

    for (int k = 0; k < sizeHistory - 1; ++k) {
        const double *rhoInkLocal = vectorRhoIn[sizeHistory - k - 2].getLocalData();
        const double *rhoOutkLocal = vectorRhoOut[sizeHistory - k - 2].getLocalData();
        for (int i = 0; i < localNumElements; ++i) {
            inputChargeValuesLocal[i] += mixingConstants[k] * (rhoInkLocal[i] - rhoInnLocal[i]);
        }
        for (int i = 0; i < localNumElements; ++i) {
            outputChargeValuesLocal[i] += mixingConstants[k] * (rhoOutkLocal[i] - rhoOutnLocal[i]);
        }
    }

    for (int i = 0; i < localNumElements; ++i) {
        inputChargeValuesLocal[i] += rhoInnLocal[i];
    }
    for (int i = 0; i < localNumElements; ++i) {
        outputChargeValuesLocal[i] += rhoOutnLocal[i];
    }

    double *rhoInLocal = rhoIn.getLocalData();
    for (int i = 0; i < localNumElements; ++i) {
        rhoInLocal[i] = std::abs(alpha * outputChargeValuesLocal[i] + (1.0 - alpha) * inputChargeValuesLocal[i]);
    }
}

void AndersonMixing::computeRhoIn(const int scfIter,
                                  Tensor3DMPI &rhoNodalIn,
                                  Tensor3DMPI &rhoGridIn) {
    int numberRhoNodalInLocalEntries = rhoNodalIn.getLocalNumberEntries();
    int numberRhoGridInLocalEntries = rhoGridIn.getLocalNumberEntries();
    double *rhoNodalInLocalData = rhoNodalIn.getLocalData();
    double *rhoGridInLocalData = rhoGridIn.getLocalData();
    double *rhoNodalOutLocalData;
    double *rhoGridOutLocalData;

    int offset_scfIter = scfIter - history_offset;

    if (offset_scfIter == 0) {
        std::for_each(rhoNodalInLocalData,
                      rhoNodalInLocalData + numberRhoNodalInLocalEntries,
                      [](double &d) { d = std::abs(d); });
        std::for_each(rhoGridInLocalData,
                      rhoGridInLocalData + numberRhoGridInLocalEntries,
                      [](double &d) { d = std::abs(d); });
    } else if (offset_scfIter == 1) {
        rhoNodalOutLocalData = vectorNodalRhoOut.back().getLocalData();
        rhoGridOutLocalData = vectorGridRhoOut.back().getLocalData();
        double *rhoVectorNodalInLocalData = vectorNodalRhoIn.back().getLocalData();;
        double *rhoVectorGridInLocalData = vectorGridRhoIn.back().getLocalData();
        for (int i = 0; i < numberRhoNodalInLocalEntries; ++i) {
            rhoNodalInLocalData[i] = alpha * rhoNodalOutLocalData[i] + (1 - alpha) * rhoVectorNodalInLocalData[i];
        }
        for (int i = 0; i < numberRhoGridInLocalEntries; ++i) {
            rhoGridInLocalData[i] = alpha * rhoGridOutLocalData[i] + (1 - alpha) * rhoVectorGridInLocalData[i];
        }
    } else {
        std::vector<double> mixingConstants;
        computeMixingConstants(vectorGridRhoIn,
                               vectorGridRhoOut,
                               mixingConstants);
#ifndef NDEBUG
        int taskId;
        MPI_Comm_rank(PETSC_COMM_WORLD,
                      &taskId);
        if (taskId == 0) {
            for (auto i: mixingConstants) {
                std::cout << i << ", ";
            }
            std::cout << std::endl;
        }
#endif
        computeRhoIn(vectorGridRhoIn,
                     vectorGridRhoOut,
                     mixingConstants,
                     rhoGridIn);
        computeRhoIn(vectorNodalRhoIn,
                     vectorNodalRhoOut,
                     mixingConstants,
                     rhoNodalIn);
    }

}

void AndersonMixing::updateRho(Tensor3DMPI &rhoNodalIn,
                               Tensor3DMPI &rhoGridIn,
                               Tensor3DMPI &rhoNodalOut,
                               Tensor3DMPI &rhoGridOut,
                               int scfIter) {
    vectorNodalRhoIn.emplace_back(rhoNodalIn);
    vectorGridRhoIn.emplace_back(rhoGridIn);
    vectorNodalRhoOut.emplace_back(rhoNodalOut);
    vectorGridRhoOut.emplace_back(rhoGridOut);

    unsigned dequeSize = vectorNodalRhoIn.size();

    int taskId;
    MPI_Comm_rank(MPI_COMM_WORLD,
                  &taskId);
    std::string rho_nodal_in =
            rho_nodal_in_prefix + "_scf" + std::to_string(scfIter) + "_proc" + std::to_string(taskId);
    std::string rho_nodal_out =
            rho_nodal_out_prefix + "_scf" + std::to_string(scfIter) + "_proc" + std::to_string(taskId);
    std::string rho_quad_in = rho_quad_in_prefix + "_scf" + std::to_string(scfIter) + "_proc" + std::to_string(taskId);
    std::string rho_quad_out =
            rho_quad_out_prefix + "_scf" + std::to_string(scfIter) + "_proc" + std::to_string(taskId);
    print_rho(rho_nodal_in_folder,
              rho_nodal_in,
              rhoNodalIn);
    print_rho(rho_nodal_out_folder,
              rho_nodal_out,
              rhoNodalOut);
    print_rho(rho_quad_in_folder,
              rho_quad_in,
              rhoGridIn);
    print_rho(rho_quad_out_folder,
              rho_quad_out,
              rhoGridOut);

    if (dequeSize > maxIterHistory) {
        vectorNodalRhoIn.pop_front();
        vectorGridRhoIn.pop_front();
        vectorNodalRhoOut.pop_front();
        vectorGridRhoOut.pop_front();

        vectorNodalRhoIn.shrink_to_fit();
        vectorGridRhoIn.shrink_to_fit();
        vectorNodalRhoOut.shrink_to_fit();
        vectorGridRhoOut.shrink_to_fit();

        int oldest_scf = scfIter - vectorNodalRhoIn.size();
        std::string rho_nodal_in_old =
                rho_nodal_in_folder + "/" + rho_nodal_in_prefix + "_scf" + std::to_string(oldest_scf) + "_proc" +
                std::to_string(taskId);
        std::string rho_nodal_out_old =
                rho_nodal_out_folder + "/" + rho_nodal_out_prefix + "_scf" + std::to_string(oldest_scf) + "_proc" +
                std::to_string(taskId);
        std::string rho_quad_in_old =
                rho_quad_in_folder + "/" + rho_quad_in_prefix + "_scf" + std::to_string(oldest_scf) + "_proc" +
                std::to_string(taskId);
        std::string rho_quad_out_old =
                rho_quad_out_folder + "/" + rho_quad_out_prefix + "_scf" + std::to_string(oldest_scf) + "_proc" +
                std::to_string(taskId);
        boost::filesystem::remove(rho_nodal_in_old);
        boost::filesystem::remove(rho_nodal_out_old);
        boost::filesystem::remove(rho_quad_in_old);
        boost::filesystem::remove(rho_quad_out_old);
    }
    std::ofstream fout(rho_path + "/history_scf.txt");
    for (int i = scfIter - vectorNodalRhoIn.size() + 1; i <= scfIter; ++i) fout << i << std::endl;
    fout.close();
}

void AndersonMixing::clearHistory(int scf_iter) {
    vectorNodalRhoIn.clear();
    vectorGridRhoIn.clear();
    vectorNodalRhoOut.clear();
    vectorGridRhoOut.clear();

    vectorNodalRhoIn.shrink_to_fit();
    vectorGridRhoIn.shrink_to_fit();
    vectorNodalRhoOut.shrink_to_fit();
    vectorGridRhoOut.shrink_to_fit();

    history_offset = scf_iter + 1;
    int task_id;
    MPI_Comm_rank(MPI_COMM_WORLD,
                  &task_id);
    if (task_id == 0) {
        std::cout << "anderson mixing history is reset. " << std::endl;
    }
}

void AndersonMixing::print_rho(const std::string &path,
                               const std::string &file_name,
                               Tensor3DMPI &tensor) {
    std::string filename = path + "/" + file_name;
    int fd;
    PetscBinaryOpen(filename.c_str(),
                    FILE_MODE_WRITE,
                    &fd);
    PetscBinaryWrite(fd,
                     tensor.getLocalData(),
                     tensor.getLocalNumberEntries(),
                     PETSC_DOUBLE,
                     PETSC_FALSE);
    PetscBinaryClose(fd);
}

void AndersonMixing::read_rho(const std::string &path,
                              const std::string &file_name,
                              Tensor3DMPI &tensor) {
    std::string filename = path + "/" + file_name;
    int fd;
    PetscBinaryOpen(filename.c_str(),
                    FILE_MODE_READ,
                    &fd);
    PetscInt num_entries = tensor.getLocalNumberEntries();
#if PETSC_VERSION_GE(3, 12, 0)
    PetscBinaryRead(fd,
                    tensor.getLocalData(),
                    num_entries,
                    PETSC_NULL,
                    PETSC_REAL);
#else
    PetscBinaryRead(fd, tensor.getLocalData(), num_entries, PETSC_DOUBLE);
#endif
    PetscBinaryClose(fd);
}
