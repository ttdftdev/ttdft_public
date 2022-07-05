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

#ifndef TUCKER_TENSOR_KSDFT_SEPARABLEHAMILTONIAN_H
#define TUCKER_TENSOR_KSDFT_SEPARABLEHAMILTONIAN_H

#include <vector>
#include <petscsnes.h>
#include "../dft/solver/FunctionalRayleighQuotientSeperable.h"

enum SeparableSCFType {
    NONE, SIMPLE, ANDERSON, PERIODIC_ANDERSON
};

class SeparableHamiltonian {
public:

    SeparableHamiltonian(FunctionalRayleighQuotientSeperable *functional);


    FunctionalRayleighQuotientSeperable *getFunctional() const;

    bool solveSCF(const std::vector<double> &initialGuessX,
                  const std::vector<double> &initialGuessY,
                  const std::vector<double> &initialGuessZ,
                  SeparableSCFType scf_type,
                  double tolerance,
                  int maxIter,
                  double alpha = 0.5,
                  int number_history = 1,
                  int period = 5);

    const std::vector<double> &getNodalFieldX() const;

    const std::vector<double> &getNodalFieldY() const;

    const std::vector<double> &getNodalFieldZ() const;

    const double getLm() const;

private:
    FunctionalRayleighQuotientSeperable *functional;
    const FEM &femX;
    const FEM &femY;
    const FEM &femZ;
    std::vector<double> nodalFieldX;
    std::vector<double> nodalFieldY;
    std::vector<double> nodalFieldZ;
    double lm; // lagrange multiplier
};

#endif //TUCKER_TENSOR_KSDFT_SEPARABLEHAMILTONIAN_H
