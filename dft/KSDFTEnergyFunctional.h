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

#ifndef TUCKER_TENSOR_KSDFT_KSDFTENERGYFUNCTIONAL_H
#define TUCKER_TENSOR_KSDFT_KSDFTENERGYFUNCTIONAL_H

#include "../tensor/Tensor3DMPI.h"
#include "../fem/FEM.h"
#include <TuckerMPI.hpp>

class KSDFTEnergyFunctional {
public:
    static void computeLDAExchangeCorrelationEnergy(const Tensor3DMPI &rhoGrid,
                                                    Tensor3DMPI &exchGrid,
                                                    TensorOperation op);

    static void computeHartreeEnergy(const Tensor3DMPI &rhoGrid,
                                     const Tensor3DMPI &potHartGrid,
                                     Tensor3DMPI &potHartEnergyGrid,
                                     TensorOperation op);

    static void computeEffMinusPSP(const Tensor3DMPI &rhoGrid,
                                   const Tensor3DMPI &potEffIn,
                                   const Tensor3DMPI &potLocIn,
                                   Tensor3DMPI &potHartEnergyGrid,
                                   TensorOperation op);

    static double computeFermiEnergy(const double *eigenValues,
                                     int numberEigenValues,
                                     int numberElectrons,
                                     double kb,
                                     double T,
                                     int maxFermiIter = 20);

    static double computeRepulsiveEnergy(const std::vector<std::vector<double> > &nuclei);

    static double compute3DIntegral(const Tensor3DMPI &fieldGridValue,
                                    const Tensor3DMPI &jacob3DMat,
                                    const Tensor3DMPI &weight3DMat);

    static double compute3DIntegralTuckerCuboid(const Tensor3DMPI &fieldGridValue,
                                                const int rankX,
                                                const int rankY,
                                                const int rankZ,
                                                const FEM &femX,
                                                const FEM &femY,
                                                const FEM &femZ);

    static double compute3DIntegralTuckerCuboidNodal(const Tensor3DMPI &fieldNodalValue,
                                                     const int rankX,
                                                     const int rankY,
                                                     const int rankZ,
                                                     const FEM &femX,
                                                     const FEM &femY,
                                                     const FEM &femZ);

private:
    static constexpr double beta1 = 1.0529;
    static constexpr double beta2 = 0.3334;
    static constexpr double A = 0.0311;
    static constexpr double B = -0.048;
    static constexpr double C = 0.002;
    static constexpr double D = -0.0116;
    static constexpr double gamma = -0.1423;

};

#endif //TUCKER_TENSOR_KSDFT_KSDFTENERGYFUNCTIONAL_H
