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

#include <mpi.h>
#include <boost/filesystem.hpp>
#include <iostream>
#include <iomanip>
#include <petscmat.h>
#include <fstream>
#include "WavefunctionWriter.h"

WavefunctionWriter::WavefunctionWriter(const int tucker_rank) {
    MPI_Comm_rank(MPI_COMM_WORLD,
                  &task_id);
    // create folders for storing rho and wavefunctions
    wfn_path = "restart/rank" + std::to_string(tucker_rank);
    if (task_id == 0) std::cout << "restart is on, the information is kept in the folder" << wfn_path << std::endl;
    boost::filesystem::path boost_wfn_path(wfn_path);
    boost::filesystem::create_directories(boost_wfn_path);
}

void WavefunctionWriter::reset_path(const int tucker_rank) {
    wfn_path = "restart/rank" + std::to_string(tucker_rank);
    if (task_id == 0) std::cout << "restart is on, the information is kept in the folder" << wfn_path << std::endl;
    boost::filesystem::path boost_wfn_path(wfn_path);
    boost::filesystem::create_directories(boost_wfn_path);
}

void WavefunctionWriter::write_wfn(double ub_unwanted,
                                   double lb_unwanted,
                                   double lb_wanted,
                                   double err_in_ub,
                                   double occ_orbital_energy,
                                   double old_gs_energy,
                                   Mat &wfn) {
    if (task_id == 0) {
        std::cout << "wrting out wavefunctions ......" << std::endl;
        std::ofstream fout(wfn_path + "/cheby.info");
        fout << ub_unwanted << lb_unwanted << lb_wanted << err_in_ub << occ_orbital_energy << old_gs_energy;
        fout.close();
    }

    PetscViewer viewer;
    std::string wfn_filename = wfn_path + "/wavefunctions.wfn";
    PetscViewerBinaryOpen(PETSC_COMM_WORLD,
                          wfn_filename.c_str(),
                          FILE_MODE_WRITE,
                          &viewer);
    PetscViewerPushFormat(viewer,
                          PETSC_VIEWER_NATIVE);
    MatView(wfn,
            viewer);
    PetscViewerDestroy(&viewer);
}

void WavefunctionWriter::read_wfn(double &ub_unwanted,
                                  double &lb_unwanted,
                                  double &lb_wanted,
                                  double &err_in_ub,
                                  double &occ_orbital_energy,
                                  double &old_gs_energy,
                                  Mat &wfn) {
    if (task_id == 0) {
        std::ifstream fin("initial_guess/cheby.info");
        fin >> ub_unwanted >> lb_unwanted >> lb_wanted >> err_in_ub >> occ_orbital_energy >> old_gs_energy;
        fin.close();
        std::cout << "reading in chebyshev values for restart ......." << std::endl;
        printf("upper bound unwanted spectrum: %.8e\n",
               ub_unwanted);
        printf("lower bound unwanted spectrum: %.8e\n",
               lb_unwanted);
        printf("lower bound wanted spectrum: %.8e\n",
               lb_wanted);
        printf("error in upper bound: %.8e\n",
               err_in_ub);
        printf("occupied orbital energy: %.8e\n",
               occ_orbital_energy);
        printf("old ground state energy: %.8e\n",
               old_gs_energy);
    }

    MPI_Bcast(&ub_unwanted,
              1,
              MPI_DOUBLE,
              0,
              MPI_COMM_WORLD);
    MPI_Bcast(&lb_unwanted,
              1,
              MPI_DOUBLE,
              0,
              MPI_COMM_WORLD);
    MPI_Bcast(&lb_wanted,
              1,
              MPI_DOUBLE,
              0,
              MPI_COMM_WORLD);
    MPI_Bcast(&err_in_ub,
              1,
              MPI_DOUBLE,
              0,
              MPI_COMM_WORLD);
    MPI_Bcast(&occ_orbital_energy,
              1,
              MPI_DOUBLE,
              0,
              MPI_COMM_WORLD);
    MPI_Bcast(&old_gs_energy,
              1,
              MPI_DOUBLE,
              0,
              MPI_COMM_WORLD);


    if (task_id == 0) std::cout << "reading in wavefunctions for restart ......" << std::endl;
    PetscViewer viewer;
    PetscViewerBinaryOpen(PETSC_COMM_WORLD,
                          "initial_guess/wavefunctions.wfn",
                          FILE_MODE_READ,
                          &viewer);
    MatCreate(PETSC_COMM_WORLD,
              &wfn);
    MatSetType(wfn,
               MATDENSE);
    MatLoad(wfn,
            viewer);
    PetscViewerDestroy(&viewer);
}

