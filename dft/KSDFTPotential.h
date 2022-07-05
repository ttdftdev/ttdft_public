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

#ifndef KSDFTPotential_H
#define KSDFTPotential_H

#include "../fem/FEM.h"
#include "TuckerMPI.hpp"
#include "../tensor/Tensor3DMPI.h"
#include "../alglib/src/interpolation.h"
#include <petscsys.h>

class KSDFTPotential {
public:

    KSDFTPotential(const FEM &femX,
                   const FEM &femY,
                   const FEM &femZ,
                   const FEM &femElectroX,
                   const FEM &femElectroY,
                   const FEM &femElectroZ,
                   const std::string &alphafile,
                   const std::string &omegafile,
                   const double Asquare);

    void computeLDAExchangePot(const TuckerMPI::Tensor *rhoGrid,
                               TuckerMPI::Tensor *effPot,
                               InsertMode addv);

    void computeLDACorrelationPot(const TuckerMPI::Tensor *rhoGrid,
                                  TuckerMPI::Tensor *effPot,
                                  InsertMode addv);

    void computeHartreePotNew(const TuckerMPI::TuckerTensor *rhoGridTT,
                              TuckerMPI::Tensor *effPot,
                              InsertMode addv);

    void computeHartreePotNew(const TuckerMPI::TuckerTensor *rhoGridTT,
                              Tensor3DMPI &effPot,
                              TensorOperation addv);

    void computeLDAExchangePot(const Tensor3DMPI &rhoGrid,
                               Tensor3DMPI &effPot,
                               TensorOperation addv);

    void computeLDACorrelationPot(const Tensor3DMPI &rhoGrid,
                                  Tensor3DMPI &effPot,
                                  TensorOperation addv);

    void computeEvanescentPSPOnGrid(const std::vector<std::vector<double> > &atomInformation,
                                    const double R,
                                    const double alpha,
                                    Tensor3DMPI &effPot,
                                    TensorOperation op);

protected:
    const FEM &femX;
    const FEM &femY;
    const FEM &femZ;
    const FEM &femElectroX;
    const FEM &femElectroY;
    const FEM &femElectroZ;
    std::string alphafile;
    std::string omegafile;
    double Asquare;
};

#endif
