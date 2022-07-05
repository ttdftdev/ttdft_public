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
#include <petsc.h>
#include <iostream>
#include <algorithm>
#include "AtomInformation.h"

AtomInformation::AtomInformation(const std::string &systemName) : systemName(systemName) {
    std::fstream fin(systemName,
                     std::fstream::in);
    std::string temp;
    std::getline(fin,
                 temp);
    std::stringstream ssin(temp);
    ssin >> numAtomType >> numAtoms >> numElectrons;
    ssin.clear();
    std::getline(fin,
                 temp);
    ssin.str(temp);
    lMax = std::vector<int>(numAtomType);
    for (int i = 0; i < numAtomType; ++i) {
        ssin >> lMax[i];
    }
    ssin.clear();
    nuclei = std::vector<std::vector<std::vector<double>>>(numAtomType);
    int atom_cnt = 0;
    while (std::getline(fin,
                        temp)) {
        std::vector<double> atom(7,
                                 0.0);
        ssin.str(temp);
        char ctemp;
        ssin >> atom[0] >> ctemp >> atom[1] >> ctemp >> atom[2] >> ctemp >> atom[3] >> ctemp >> atom[4] >> ctemp
             >> atom[5]
             >> ctemp >> atom[6];
        nuclei[int(atom[6])].emplace_back(atom);
        ssin.clear();
        atom_cnt++;
    }
    if (atom_cnt != numAtoms) {
        std::cerr << "number of atoms does not match.";
        std::terminate();
    }
    PetscPrintf(MPI_COMM_WORLD,
                "Atom Information: \n");
    PetscPrintf(MPI_COMM_WORLD,
                "num atom type: %d, num atoms: %d, num electrons: %d\n",
                numAtomType,
                numAtoms,
                numElectrons);
    PetscPrintf(MPI_COMM_WORLD,
                "lmax: ");
    for (int i = 0; i < numAtomType; ++i)
        PetscPrintf(MPI_COMM_WORLD,
                    "%d ",
                    lMax[i]);
    PetscPrintf(MPI_COMM_WORLD,
                "\n");
    for (int i = 0; i < numAtomType; ++i) {
        PetscPrintf(MPI_COMM_WORLD,
                    "atom type %d\n",
                    i);
        int lmax_square = lMax[i] * lMax[i];
        for (int j = 0; j < nuclei[i].size(); ++j) {
            all_nuclei.emplace_back(nuclei[i][j]);
        }
    }
}