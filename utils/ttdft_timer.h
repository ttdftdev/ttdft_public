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

#ifndef TTDFT_GPU_TTDFT_TIMER_H
#define TTDFT_GPU_TTDFT_TIMER_H

class timer {
public:
    void start() {
#ifdef TIME_THE_FUNCTION
        time = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);
#endif
    }

    void end(std::string s) {
#ifdef TIME_THE_FUNCTION
        MPI_Barrier(MPI_COMM_WORLD);
        time = MPI_Wtime() - time;
        int irank;
        MPI_Comm_rank(MPI_COMM_WORLD, &irank);
        if (irank == 0) {
            std::cout << "time for " << s << ": " << time << " sec." << std::endl;
        }
#endif
    }

private:
    double time;
};

#endif //TTDFT_GPU_TTDFT_TIMER_H
