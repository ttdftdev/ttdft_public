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
#include <iostream>
#include <sstream>
#include <mpi.h>
#include "FileReader.h"

int Utils::ReadFile(unsigned number_columns,
                    const std::string &filename,
                    std::vector<std::vector<double>> &data) {

    int taskId;
    MPI_Comm_rank(MPI_COMM_WORLD,
                  &taskId);

    int return_value = 1;
    if (taskId == 0) {
        std::vector<double> rowData(number_columns,
                                    0.0);
        std::ifstream readFile(filename.c_str());

        if (readFile.fail()) {
            return_value = 0;
        } else {

            // String to store line and word
            std::string readLine;
            std::string word;

            // column index
            int column_count;

            if (readFile.is_open()) {
                while (std::getline(readFile,
                                    readLine)) {
                    std::istringstream iss(readLine);

                    column_count = 0;

                    while (iss >> word && column_count < number_columns)
                        rowData[column_count++] = atof(word.c_str());

                    data.push_back(rowData);
                }
            }
            readFile.close();
        }
    }
    MPI_Bcast(&return_value,
              1,
              MPI_INT,
              0,
              MPI_COMM_WORLD);
    if (return_value == 0) {
        return return_value;
    }

    int data_size = data.size();
    MPI_Bcast(&data_size,
              1,
              MPI_INT,
              0,
              MPI_COMM_WORLD);
    if (taskId != 0) {
        data = std::vector<std::vector<double>>(data_size);
    }
    for (int i = 0; i < data_size; ++i) {
        int data_entry_size = data[i].size();
        MPI_Bcast(&data_entry_size,
                  1,
                  MPI_INT,
                  0,
                  MPI_COMM_WORLD);
        if (taskId != 0) {
            data[i] = std::vector<double>(data_entry_size);
        }
        MPI_Bcast(data[i].data(),
                  data_entry_size,
                  MPI_DOUBLE,
                  0,
                  MPI_COMM_WORLD);
    }

    return return_value;
}

void Utils::ReadSingleColumnFile(const std::string &filename,
                                 std::vector<double> &data) {

    int taskId;
    MPI_Comm_rank(MPI_COMM_WORLD,
                  &taskId);
    if (taskId == 0) {
        std::fstream fin;
        fin.open(filename.c_str());
        double readintemp;
        data.clear();
        while (fin >> readintemp) {
            data.push_back(readintemp);
        }
        data.shrink_to_fit();
        fin.close();
    }
    int data_size = data.size();
    MPI_Bcast(&data_size,
              1,
              MPI_INT,
              0,
              MPI_COMM_WORLD);
    if (taskId != 0) {
        data = std::vector<double>(data_size);
    }
    MPI_Bcast(data.data(),
              data_size,
              MPI_DOUBLE,
              0,
              MPI_COMM_WORLD);

}