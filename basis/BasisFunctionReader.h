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

#ifndef TUCKER_TENSOR_KSDFT_BASISFUNCTIONREADER_H
#define TUCKER_TENSOR_KSDFT_BASISFUNCTIONREADER_H

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <mpi.h>
#include "../alglib/src/ap.h"
#include "../alglib/src/interpolation.h"

namespace BasisFunctionReader {

    void copy_basis_function_from_file(const std::string &filename,
                                       std::vector<std::vector<double>> &basis) {
        std::ifstream fin(filename);
        std::string temp;
        while (std::getline(fin,
                            temp)) {
            std::vector<double> vtemp;
            std::istringstream ssin(temp);
            double dtemp;
            vtemp.push_back(0.0);
            while (ssin >> dtemp) {
                vtemp.push_back(dtemp);
            }
            vtemp.push_back(0.0);
            basis.push_back(vtemp);
        }
        fin.close();
    }

    void interpolate_basis_function_from_file(const std::string &filename,
                                              int num_ranks,
                                              const std::vector<double> &nodal_coordinates,
                                              std::vector<std::vector<double>> &basis_functions) {
        int taskId;
        MPI_Comm_rank(MPI_COMM_WORLD,
                      &taskId);

        basis_functions = std::vector<std::vector<double>>(num_ranks,
                                                           std::vector<double>(nodal_coordinates.size()));
        if (taskId == 0) {
            std::cout << "read in basis functions from file: " << filename << std::endl;
            // read in first line to determine the rows and columns
            std::ifstream fin(filename);
            std::string input_temp;
            std::getline(fin,
                         input_temp);
            std::istringstream ssin(input_temp);
            double digit_temp;
            std::vector<double> temp_vec;
            while (ssin >> digit_temp) {
                temp_vec.emplace_back(digit_temp);
            }
            ssin.clear();
            int ncols = temp_vec.size();
            int num_file_rank = temp_vec.size() - 1;

            try {
                if (num_file_rank < num_ranks) {
                    throw std::string("ranks of the basis function input is less than requested by the code");
                }
            } catch (const std::string &e) {
                std::cout << e << std::endl;
                std::terminate();
            }


            // read in nodal values of basis functions
            std::vector<std::vector<double> > eig(ncols);
            for (int i = 0; i < ncols; ++i) {
                eig[i].emplace_back(temp_vec[i]);
            }
            while (std::getline(fin,
                                input_temp)) {
                ssin.str(input_temp);
                for (int i = 0; i < ncols; ++i) {
                    ssin >> digit_temp;
                    eig[i].emplace_back(digit_temp);
                }
                ssin.clear();
            }

            // construct spline object
            int n = eig[0].size();
            alglib::real_1d_array coordinates;
            coordinates.setlength(n);
            std::copy(eig[0].begin(),
                      eig[0].end(),
                      coordinates.getcontent());

            std::vector<alglib::spline1dinterpolant> basis_functions_interpolant(num_file_rank);
            for (int i = 0; i < num_file_rank; ++i) {
                alglib::real_1d_array nodal_basis_function;
                nodal_basis_function.setlength(n);
                std::copy(eig[i + 1].begin(),
                          eig[i + 1].end(),
                          nodal_basis_function.getcontent());
                alglib::spline1dbuildcubic(coordinates,
                                           nodal_basis_function,
                                           n,
                                           1,
                                           0.0,
                                           1,
                                           0.0,
                                           basis_functions_interpolant[i]);
            }

            for (int i = 0; i < num_ranks; ++i) {
                for (int j = 0; j < basis_functions[i].size(); ++j) {
                    if (nodal_coordinates[j] > eig[0].back() || nodal_coordinates[j] < eig[0].front()) {
                        basis_functions[i][j] = 0.0;
                    } else {
                        basis_functions[i][j] = alglib::spline1dcalc(basis_functions_interpolant[i],
                                                                     nodal_coordinates[j]);
                    }
                }
            }
        }

        for (int irank = 0; irank < num_ranks; ++irank) {
            MPI_Bcast(basis_functions[irank].data(),
                      basis_functions[irank].size(),
                      MPI_DOUBLE,
                      0,
                      MPI_COMM_WORLD);
        }

    }
}

#endif //TUCKER_TENSOR_KSDFT_BASISFUNCTIONREADER_H
