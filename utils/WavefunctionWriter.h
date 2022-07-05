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

#ifndef TUCKERDFTSPARSE_UTILS_WAVEFUNCTIONWRITER_H_
#define TUCKERDFTSPARSE_UTILS_WAVEFUNCTIONWRITER_H_

/**
 * To restart, copy the file under restart/rankN as initial_guess, i.e. cp restart/rankN initial_guess
 */

#include <string>
#include "../tensor/Tensor3DMPI.h"

class WavefunctionWriter {
public:
    WavefunctionWriter(const int tucker_rank);

    void reset_path(const int tucker_rank);

    void write_wfn(double ub_unwanted,
                   double lb_unwanted,
                   double lb_wanted,
                   double err_in_ub,
                   double occ_orbital_energy,
                   double old_gs_energy,
                   Mat &wfn);

    void read_wfn(double &ub_unwanted,
                  double &lb_unwanted,
                  double &lb_wanted,
                  double &err_in_ub,
                  double &occ_orbital_energy,
                  double &old_gs_energy,
                  Mat &wfn);

    void write_tensor(std::string &tensor_name,
                      Tensor3DMPI &tensor);

    void read_tensor(std::string &tensor_name,
                     Tensor3DMPI &tensor);

private:
    int task_id;
    std::string wfn_path;

    WavefunctionWriter() {}
};

#endif //TUCKERDFTSPARSE_UTILS_WAVEFUNCTIONWRITER_H_
