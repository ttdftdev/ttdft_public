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

#include <fstream>
#include <mpi.h>
#include <boost/math/special_functions/spherical_harmonic.hpp>
#include <boost/math/distributions/normal.hpp>
#include <petscmat.h>
#include "PsiInitialGuessGnerator.h"
#include "../utils/FileReader.h"
#include "../tensor/TuckerTensor.h"

PsiInitialGuessGnerator::PsiInitialGuessGnerator(const FEM &fem_x,
                                                 const FEM &fem_y,
                                                 const FEM &fem_z,
                                                 const int field_decomposition_rank_x,
                                                 const int field_decomposition_rank_y,
                                                 const int field_decomposition_rank_z,
                                                 const std::vector<std::vector<double>> &atom_information,
                                                 unsigned int number_eigen_values)
        : fem_x(fem_x),
          fem_y(fem_y),
          fem_z(fem_z),
          field_decomposition_rank(3),
          atom_information(atom_information),
          numeber_eigen_values(number_eigen_values) {
    field_decomposition_rank[0] = field_decomposition_rank_x;
    field_decomposition_rank[1] = field_decomposition_rank_y;
    field_decomposition_rank[2] = field_decomposition_rank_z;
    DetermineOrbitalFilling();
    ComputeOwnedEigenvectors();
}

PsiInitialGuessGnerator::~PsiInitialGuessGnerator() {
    for (auto iteri = radial_values.begin(); iteri != radial_values.end(); ++iteri) {
        for (auto iterj = iteri->second.begin(); iterj != iteri->second.end(); ++iterj) {
            for (auto iterk = iterj->second.begin(); iterk != iterj->second.end(); ++iterk) {
                delete iterk->second;
            }
        }
    }
}

/**
 * @brief used to compute which eigenvectors are to be caluclated by this processor
 */
void PsiInitialGuessGnerator::ComputeOwnedEigenvectors() {
    int taskId, taskSize;
    MPI_Comm_rank(MPI_COMM_WORLD,
                  &taskId);
    MPI_Comm_size(MPI_COMM_WORLD,
                  &taskSize);
    int number_eigenvectors_per_proc = std::ceil(double(numeber_eigen_values) / taskSize);
    owned_eigenvectors_start = taskId * number_eigenvectors_per_proc;
    owned_eigenvectors_end = (taskId + 1) * number_eigenvectors_per_proc;
    if (owned_eigenvectors_start > numeber_eigen_values) {
        owned_eigenvectors_start = numeber_eigen_values;
    }
    if (owned_eigenvectors_end > numeber_eigen_values) {
        owned_eigenvectors_end = numeber_eigen_values;
    }
}

void PsiInitialGuessGnerator::LoadPSIFIles(unsigned int Z,
                                           unsigned int n,
                                           unsigned int l,
                                           unsigned int &file_read_flag) {
    if (radial_values[Z][n].count(l) > 0) {
        file_read_flag = 1;
        return;
    }

    //
    //set the paths for the Single-Atom wavefunction data
    //
    char psiFile[256];
    sprintf(psiFile,
            "z%upsi%u%u.inp",
            Z,
            n,
            l);

    std::vector<std::vector<double> > values;

    file_read_flag = Utils::ReadFile(2,
                                     psiFile,
                                     values);


    //
    //spline fitting for single-atom wavefunctions
    //
    int taskId;
    MPI_Comm_rank(MPI_COMM_WORLD,
                  &taskId);
    if (file_read_flag > 0) {
        if (taskId == 0) { std::cout << "read in file: " << psiFile << std::endl; }
        int numRows = values.size() - 1;
        std::vector<double> xData(numRows), yData(numRows);

        for (int irow = 0; irow < numRows; ++irow) {
            xData[irow] = values[irow][0];
        }
        outerValues[Z][n][l] = xData[numRows - 1];
        alglib::real_1d_array x;
        x.setcontent(numRows,
                     &xData[0]);

        for (int irow = 0; irow < numRows; ++irow) {
            yData[irow] = values[irow][1];
        }
        alglib::real_1d_array y;
        y.setcontent(numRows,
                     &yData[0]);

        alglib::ae_int_t natural_bound_type = 0;
        alglib::spline1dinterpolant *spline = new alglib::spline1dinterpolant;
        alglib::spline1dbuildcubic(x,
                                   y,
                                   numRows,
                                   natural_bound_type,
                                   0.0,
                                   natural_bound_type,
                                   0.0,
                                   *spline);

        radial_values[Z][n][l] = spline;
    }
}

//
//determine Orbital ordering
//
void PsiInitialGuessGnerator::DetermineOrbitalFilling() {

    int taskId;
    MPI_Comm_rank(MPI_COMM_WORLD,
                  &taskId);

    //
    //create a stencil following Orbital filling order
    //

    typedef std::array<unsigned, 2> level;
    std::vector<level> stencil;

    //1s
    stencil.emplace_back(level({1, 0}));
    //2s
    stencil.emplace_back(level({2, 0}));
    //2p
    stencil.emplace_back(level({2, 1}));
    //3s
    stencil.emplace_back(level({3, 0}));
    //3p
    stencil.emplace_back(level({3, 1}));
    //4s
    stencil.emplace_back(level({4, 0}));
    //3d
    stencil.emplace_back(level({3, 2}));
    //4p
    stencil.emplace_back(level({4, 1}));
    //5s
    stencil.emplace_back(level({5, 0}));
    //4d
    stencil.emplace_back(level({4, 2}));
    //5p
    stencil.emplace_back(level({5, 1}));
    //6s
    stencil.emplace_back(level({6, 0}));
    //4f
    stencil.emplace_back(level({4, 3}));
    //5d
    stencil.emplace_back(level({5, 2}));
    //6p
    stencil.emplace_back(level({6, 1}));
    //7s
    stencil.emplace_back(level({7, 1}));
    //5f
    stencil.emplace_back(level({5, 3}));
    //6d
    stencil.emplace_back(level({6, 2}));
    //7p
    stencil.emplace_back(level({7, 1}));
    //8s
    stencil.emplace_back(level({8, 0}));

    unsigned int fileReadFlag = 0;
    unsigned int waveFunctionCount = 0;
    unsigned int numberGlobalAtoms = atom_information.size();
    unsigned int errorReadFile = 0;

    int orbita_count = 0;
    for (std::vector<level>::iterator it = stencil.begin(); it < stencil.end(); it++) {
        unsigned int n = (*it)[0], l = (*it)[1];

        for (int m = -l; m <= (int) l; m++) {
            for (unsigned int iAtom = 0; iAtom < numberGlobalAtoms; iAtom++) {
                unsigned int Z = atom_information[iAtom][5];

                //
                //load PSI files
                //
                LoadPSIFIles(Z,
                             n,
                             l,
                             fileReadFlag);

                if (fileReadFlag > 0) {
                    Orbital temp;
                    temp.atom_id = iAtom;
                    temp.Z = Z;
                    temp.n = n;
                    temp.l = l;
                    temp.m = m;
                    temp.psi = radial_values[Z][n][l];
                    wave_functions_vector.emplace_back(temp);
                    waveFunctionCount++;
                    if (waveFunctionCount >= numeber_eigen_values && waveFunctionCount >= numberGlobalAtoms) break;
                }

            }
            if (waveFunctionCount >= numeber_eigen_values && waveFunctionCount >= numberGlobalAtoms) break;
        }

        if (waveFunctionCount >= numeber_eigen_values && waveFunctionCount >= numberGlobalAtoms) break;

        if (fileReadFlag == 0)
            errorReadFile += 1;
    }

    if (errorReadFile == stencil.size()) {
        std::cerr << "Error: Require single-atom wavefunctions as initial guess for starting the SCF." << std::endl;
        std::cerr << "Error: Could not find single-atom wavefunctions for any atom: " << std::endl;
        exit(-1);
    }

    if (wave_functions_vector.size() > numeber_eigen_values) {
        numeber_eigen_values = wave_functions_vector.size();
    }

    if (taskId == 0) {
        std::cout
                << "============================================================================================================================="
                << std::endl;
        std::cout
                << "number of wavefunctions computed using single atom data to be used as initial guess for starting the SCF: "
                << waveFunctionCount << std::endl;
        std::cout
                << "============================================================================================================================="
                << std::endl;
    }
}

void PsiInitialGuessGnerator::ComputePSI(Orbital &orbital,
                                         Tucker::Tensor *psi) {
    double *psi_data = psi->data();
    const std::vector<double> &nodes_x = fem_x.getGlobalNodalCoord();
    const std::vector<double> &nodes_y = fem_y.getGlobalNodalCoord();
    const std::vector<double> &nodes_z = fem_z.getGlobalNodalCoord();

    const std::vector<double> &atom_coord = atom_information[orbital.atom_id];

    for (int k = 0; k < nodes_z.size(); ++k) {
        for (int j = 0; j < nodes_y.size(); ++j) {
            for (int i = 0; i < nodes_x.size(); ++i) {
                double x = nodes_x[i] - atom_coord[1];
                double y = nodes_y[j] - atom_coord[2];
                double z = nodes_z[k] - atom_coord[3];

                double r = sqrt(x * x + y * y + z * z);
                double theta = acos(z / r);
                double phi = atan2(y,
                                   x);

                if (r == 0) {
                    theta = 0;
                    phi = 0;
                }
                //radial part
                double R = 0.0;
                if (r <= outerValues[orbital.Z][orbital.n][orbital.l])
                    R = alglib::spline1dcalc(*(orbital.psi),
                                             r);
                //spherical part
                if (orbital.m > 0) {
                    psi_data[i] +=
                            R * std::sqrt(2) * boost::math::spherical_harmonic_r(orbital.l,
                                                                                 orbital.m,
                                                                                 theta,
                                                                                 phi);
                } else if (orbital.m == 0) {
                    psi_data[i] += R * boost::math::spherical_harmonic_r(orbital.l,
                                                                         orbital.m,
                                                                         theta,
                                                                         phi);
                } else {
                    psi_data[i] +=
                            R * std::sqrt(2) * boost::math::spherical_harmonic_i(orbital.l,
                                                                                 -orbital.m,
                                                                                 theta,
                                                                                 phi);
                }
            }
        }
    }
}

void PsiInitialGuessGnerator::ComputeFieldTuckerDecompositionOnQuad(Tucker::Tensor *nodal_field,
                                                                    Tucker::Tensor *seq_core_tensor,
                                                                    std::vector<std::vector<double>> &Ux_quad,
                                                                    std::vector<std::vector<double>> &Uy_quad,
                                                                    std::vector<std::vector<double>> &Uz_quad) {
    const Tucker::TuckerTensor *decomposed_nodal_field = Tucker::STHOSVD(nodal_field,
                                                                         &field_decomposition_rank,
                                                                         false);

    Tucker::Tensor *core_quad_field = decomposed_nodal_field->G;
//  seq_core_tensor = Tucker::MemoryManager::safe_new<Tucker::Tensor>(core_quad_field->size());
    std::copy(core_quad_field->data(),
              core_quad_field->data() + core_quad_field->getNumElements(),
              seq_core_tensor->data());
    Tucker::Matrix *Ux = decomposed_nodal_field->U[0];
    Tucker::Matrix *Uy = decomposed_nodal_field->U[1];
    Tucker::Matrix *Uz = decomposed_nodal_field->U[2];
    int Ux_ncols = Ux->ncols(), Ux_nrows = Ux->nrows();
    int Uy_ncols = Uy->ncols(), Uy_nrows = Uy->nrows();
    int Uz_ncols = Uz->ncols(), Uz_nrows = Uz->nrows();
    Ux_quad = std::vector<std::vector<double>>(Ux_ncols);
    Uy_quad = std::vector<std::vector<double>>(Uy_ncols);
    Uz_quad = std::vector<std::vector<double>>(Uz_ncols);
    for (int i = 0; i < Ux_ncols; ++i) {
        std::vector<double> nodal_values(Ux_nrows,
                                         0.0);
        std::copy(Ux->data() + i * Ux_nrows,
                  Ux->data() + (i + 1) * Ux_nrows,
                  nodal_values.begin());
        fem_x.computeQuadValuesFromNodalValues(nodal_values,
                                               Ux_quad[i]);
    }
    for (int i = 0; i < Uy_ncols; ++i) {
        std::vector<double> nodal_values(Uy_nrows,
                                         0.0);
        std::copy(Uy->data() + i * Uy_nrows,
                  Uy->data() + (i + 1) * Uy_nrows,
                  nodal_values.begin());
        fem_y.computeQuadValuesFromNodalValues(nodal_values,
                                               Uy_quad[i]);
    }
    for (int i = 0; i < Uz_ncols; ++i) {
        std::vector<double> nodal_values(Uz_nrows,
                                         0.0);
        std::copy(Uz->data() + i * Uz_nrows,
                  Uz->data() + (i + 1) * Uz_nrows,
                  nodal_values.begin());
        fem_z.computeQuadValuesFromNodalValues(nodal_values,
                                               Uz_quad[i]);
    }
    Tucker::MemoryManager::safe_delete(decomposed_nodal_field);
}

void PsiInitialGuessGnerator::ComputeTuckerBasisOnQuad(const std::vector<std::vector<double>> &basis_x,
                                                       const std::vector<std::vector<double>> &basis_y,
                                                       const std::vector<std::vector<double>> &basis_z,
                                                       std::vector<std::vector<double>> &basis_quad_x,
                                                       std::vector<std::vector<double>> &basis_quad_y,
                                                       std::vector<std::vector<double>> &basis_quad_z) {
    basis_quad_x = std::vector<std::vector<double>>(basis_x.size());
    basis_quad_y = std::vector<std::vector<double>>(basis_y.size());
    basis_quad_z = std::vector<std::vector<double>>(basis_z.size());
    for (int i = 0; i < basis_x.size(); ++i) {
        fem_x.computeQuadValuesFromNodalValues(basis_x[i],
                                               basis_quad_x[i]);
    }
    for (int i = 0; i < basis_y.size(); ++i) {
        fem_y.computeQuadValuesFromNodalValues(basis_y[i],
                                               basis_quad_y[i]);
    }
    for (int i = 0; i < basis_z.size(); ++i) {
        fem_z.computeQuadValuesFromNodalValues(basis_z[i],
                                               basis_quad_z[i]);
    }
}

void PsiInitialGuessGnerator::ProjectFieldOntoTuckerBasis(Tucker::Tensor *seq_core_tensor,
                                                          const std::vector<std::vector<double>> &decomposed_field_quad_ux,
                                                          const std::vector<std::vector<double>> &decomposed_field_quad_uy,
                                                          const std::vector<std::vector<double>> &decomposed_field_quad_uz,
                                                          const std::vector<std::vector<double>> &basis_quad_x,
                                                          const std::vector<std::vector<double>> &basis_quad_y,
                                                          const std::vector<std::vector<double>> &basis_quad_z,
                                                          std::vector<double> &projection_coefficients) {
    int field_rank_x = decomposed_field_quad_ux.size();
    int field_rank_y = decomposed_field_quad_uy.size();
    int field_rank_z = decomposed_field_quad_uz.size();
    int tucker_rank_x = basis_quad_x.size();
    int tucker_rank_y = basis_quad_y.size();
    int tucker_rank_z = basis_quad_z.size();
    Tucker::Matrix *Ux = Tucker::MemoryManager::safe_new<Tucker::Matrix>(tucker_rank_x,
                                                                         field_rank_x);
    Tucker::Matrix *Uy = Tucker::MemoryManager::safe_new<Tucker::Matrix>(tucker_rank_y,
                                                                         field_rank_y);
    Tucker::Matrix *Uz = Tucker::MemoryManager::safe_new<Tucker::Matrix>(tucker_rank_z,
                                                                         field_rank_z);

    int number_quad_points_x = fem_x.getTotalNumberQuadPoints();
    int number_quad_points_y = fem_y.getTotalNumberQuadPoints();
    int number_quad_points_z = fem_z.getTotalNumberQuadPoints();
    const std::vector<double> &jacob_quad_x = fem_x.getJacobQuadPointValues();
    const std::vector<double> &jacob_quad_y = fem_y.getJacobQuadPointValues();
    const std::vector<double> &jacob_quad_z = fem_z.getJacobQuadPointValues();
    const std::vector<double> &weight_quad_x = fem_x.getWeightQuadPointValues();
    const std::vector<double> &weight_quad_y = fem_y.getWeightQuadPointValues();
    const std::vector<double> &weight_quad_z = fem_z.getWeightQuadPointValues();

    double *Ux_data = Ux->data();
    int Ux_cnt = 0;
    for (int j = 0; j < field_rank_x; ++j) {
        for (int i = 0; i < tucker_rank_x; ++i) {
            double result = 0.0;
            for (int q = 0; q != number_quad_points_x; ++q) {
                result += basis_quad_x[i][q] * decomposed_field_quad_ux[j][q] * jacob_quad_x[q] * weight_quad_x[q];
            }
            Ux_data[Ux_cnt++] = result;
        }
    }

    double *Uy_data = Uy->data();
    int Uy_cnt = 0;
    for (int j = 0; j < field_rank_y; ++j) {
        for (int i = 0; i < tucker_rank_y; ++i) {
            double result = 0.0;
            for (int q = 0; q != number_quad_points_y; ++q) {
                result += basis_quad_y[i][q] * decomposed_field_quad_uy[j][q] * jacob_quad_y[q] * weight_quad_y[q];
            }
            Uy_data[Uy_cnt++] = result;
        }
    }

    double *Uz_data = Uz->data();
    int Uz_cnt = 0;
    for (int j = 0; j < field_rank_z; ++j) {
        for (int i = 0; i < tucker_rank_z; ++i) {
            double result = 0.0;
            for (int q = 0; q != number_quad_points_z; ++q) {
                result += basis_quad_z[i][q] * decomposed_field_quad_uz[j][q] * jacob_quad_z[q] * weight_quad_z[q];
            }
            Uz_data[Uz_cnt++] = result;
        }
    }

    Tucker::Tensor *reconstructed_tensor;
    Tucker::Tensor *temp;
    temp = seq_core_tensor;
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

    projection_coefficients = std::vector<double>(reconstructed_tensor->data(),
                                                  reconstructed_tensor->data() +
                                                  reconstructed_tensor->getNumElements());

    Tucker::MemoryManager::safe_delete(reconstructed_tensor);
    Tucker::MemoryManager::safe_delete(Ux);
    Tucker::MemoryManager::safe_delete(Uy);
    Tucker::MemoryManager::safe_delete(Uz);
}

/*
 * @param mat_to_tucker is the container having the mapping between Mat noding to the Tucker coordinates
 */
void PsiInitialGuessGnerator::ComputeInitialGuessPSI(const std::vector<std::vector<double>> &basis_x,
                                                     const std::vector<std::vector<double>> &basis_y,
                                                     const std::vector<std::vector<double>> &basis_z,
                                                     Mat &eig_vecs) {

    std::vector<std::vector<double>> basis_quad_x, basis_quad_y, basis_quad_z;
    ComputeTuckerBasisOnQuad(basis_x,
                             basis_y,
                             basis_z,
                             basis_quad_x,
                             basis_quad_y,
                             basis_quad_z);

    MatZeroEntries(eig_vecs);
    double *eig_vecs_data;
    MatDenseGetArray(eig_vecs,
                     &eig_vecs_data);
    Tucker::SizeArray psi_size(3);
    psi_size[0] = fem_x.getTotalNumberNodes();
    psi_size[1] = fem_y.getTotalNumberNodes();
    psi_size[2] = fem_z.getTotalNumberNodes();

    int owned_atom_eigenvectors_start = owned_eigenvectors_start;
    int owned_atom_eigenvectors_end = owned_eigenvectors_end;
    int owned_nonatom_eigenvectors_start = wave_functions_vector.size();
    int owned_nonatom_eigenvectors_end = wave_functions_vector.size();
    if (owned_eigenvectors_start >= wave_functions_vector.size()) {
        owned_atom_eigenvectors_start = wave_functions_vector.size();
        owned_atom_eigenvectors_end = wave_functions_vector.size();
        owned_nonatom_eigenvectors_start = owned_eigenvectors_start;
        owned_nonatom_eigenvectors_end = owned_eigenvectors_end;
    } else if (owned_eigenvectors_end > wave_functions_vector.size()) {
        owned_atom_eigenvectors_start = owned_eigenvectors_start;
        owned_atom_eigenvectors_end = wave_functions_vector.size();
        owned_nonatom_eigenvectors_start = wave_functions_vector.size();
        owned_nonatom_eigenvectors_end = owned_eigenvectors_end;
    }
    for (int i = owned_atom_eigenvectors_start; i < owned_atom_eigenvectors_end; ++i) {
        Tucker::Tensor *psi = Tucker::MemoryManager::safe_new<Tucker::Tensor>(psi_size);
        ComputePSI(wave_functions_vector[i],
                   psi);
        Tucker::Tensor *core = Tucker::MemoryManager::safe_new<Tucker::Tensor>(field_decomposition_rank);
        std::vector<std::vector<double>> Ux_quad, Uy_quad, Uz_quad;
        ComputeFieldTuckerDecompositionOnQuad(psi,
                                              core,
                                              Ux_quad,
                                              Uy_quad,
                                              Uz_quad);

        std::vector<double> projection_coefficients;
        ProjectFieldOntoTuckerBasis(core,
                                    Ux_quad,
                                    Uy_quad,
                                    Uz_quad,
                                    basis_quad_x,
                                    basis_quad_y,
                                    basis_quad_z,
                                    projection_coefficients);
        std::copy(projection_coefficients.begin(),
                  projection_coefficients.end(),
                  eig_vecs_data + i * projection_coefficients.size());
        Tucker::MemoryManager::safe_delete(core);
        Tucker::MemoryManager::safe_delete(psi);
    }

    if (wave_functions_vector.size() < numeber_eigen_values) {
        unsigned non_atomic_wave_functions = numeber_eigen_values - wave_functions_vector.size();
        PetscPrintf(MPI_COMM_WORLD,
                    "                                                                                             \n");
        PetscPrintf(MPI_COMM_WORLD,
                    "number of wavefunctions generated randomly to be used as initial guess for starting the SCF : %d\n",
                    non_atomic_wave_functions);

        // assign the rest of the wavefunctions using a standard normal distribution
        boost::math::normal normDist;

        for (unsigned wave_i = owned_nonatom_eigenvectors_start; wave_i < owned_nonatom_eigenvectors_end; ++wave_i) {
            Tucker::Tensor *random_function = Tucker::MemoryManager::safe_new<Tucker::Tensor>(psi_size);
            double *random_function_data = random_function->data();
            int random_function_size = random_function->getNumElements();
            for (int i = 0; i < random_function_size; ++i) {
                double z = (-0.5 + (rand() + 0.0) / (RAND_MAX)) * 3.0;
                double value = boost::math::pdf(normDist,
                                                z);
                if (rand() % 2 == 0)
                    value = -1.0 * value;
                random_function_data[i] = value;
            }

            Tucker::Tensor *core = Tucker::MemoryManager::safe_new<Tucker::Tensor>(field_decomposition_rank);
            std::vector<std::vector<double>> Ux_quad, Uy_quad, Uz_quad;
            ComputeFieldTuckerDecompositionOnQuad(random_function,
                                                  core,
                                                  Ux_quad,
                                                  Uy_quad,
                                                  Uz_quad);
            std::vector<double> projection_coefficients;
            ProjectFieldOntoTuckerBasis(core,
                                        Ux_quad,
                                        Uy_quad,
                                        Uz_quad,
                                        basis_quad_x,
                                        basis_quad_y,
                                        basis_quad_z,
                                        projection_coefficients);
            std::copy(projection_coefficients.begin(),
                      projection_coefficients.end(),
                      eig_vecs_data + wave_i * projection_coefficients.size());
            Tucker::MemoryManager::safe_delete(core);
            Tucker::MemoryManager::safe_delete(random_function);
        }
    }
    PetscInt eig_vecs_size, eig_vecs_m, eig_vecs_n;
    MatGetSize(eig_vecs,
               &eig_vecs_m,
               &eig_vecs_n);
    eig_vecs_size = eig_vecs_m * eig_vecs_n;
    MPI_Allreduce(MPI_IN_PLACE,
                  eig_vecs_data,
                  eig_vecs_size,
                  MPI_DOUBLE,
                  MPI_SUM,
                  PETSC_COMM_WORLD);
    MatDenseRestoreArray(eig_vecs,
                         &eig_vecs_data);
}