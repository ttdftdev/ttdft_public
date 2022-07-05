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

#ifndef TUCKER_TENSOR_KSDFT_INITIALGUESS_H
#define TUCKER_TENSOR_KSDFT_INITIALGUESS_H

#include "../fem/FEM.h"
#include "../tensor/Tensor3DMPI.h"
#include "../utils/InputParameter.h"
#include "../atoms/AtomInformation.h"

class InitialGuess {
public:

    void initialize_rho(const FEM &femX,
                        const FEM &femY,
                        const FEM &femZ,
                        InputParameter &inputParameter,
                        AtomInformation &atomInformation,
                        Tensor3DMPI &rhoNodal,
                        Tensor3DMPI &rhoGrid);

    void initialize_rho_tucker(const FEM &femX,
                               const FEM &femY,
                               const FEM &femZ,
                               Tensor3DMPI &rhoNodal,
                               Tensor3DMPI &rhoGrid);

    void initializeRhoNodal(const FEM &femX,
                            const FEM &femY,
                            const FEM &femZ,
                            Tensor3DMPI &rhoNodal);

    void initializeRhoGrid(const FEM &femX,
                           const FEM &femY,
                           const FEM &femZ,
                           Tensor3DMPI &rhoGrid);

    void initialize_rho_by_hydrogen(const FEM &femX,
                                    const FEM &femY,
                                    const FEM &femZ,
                                    const std::vector<std::vector<double> > &nuclei,
                                    Tensor3DMPI &rhoNodal);

    void initialize_rho_grid_by_hydrogen(const FEM &femX,
                                         const FEM &femY,
                                         const FEM &femZ,
                                         const std::vector<std::vector<double> > &nuclei,
                                         Tensor3DMPI &rhoGrid);

    void initilize_rho_from_single_atom_fem(const FEM &femX,
                                            const FEM &femY,
                                            const FEM &femZ,
                                            const std::vector<std::vector<double> > &nuclei,
                                            std::string nodalFileX,
                                            std::string nodalFileY,
                                            std::string nodalFileZ,
                                            std::string fieldFile,
                                            Tensor3DMPI &rhoNodal,
                                            Tensor3DMPI &rhoGrid);

    void initilize_rho_from_previous_calculation_fem(const FEM &femX,
                                                     const FEM &femY,
                                                     const FEM &femZ,
                                                     std::string nodalFileX,
                                                     std::string nodalFileY,
                                                     std::string nodalFileZ,
                                                     std::string fieldFile,
                                                     Tensor3DMPI &rhoNodal,
                                                     Tensor3DMPI &rhoGrid);

    void initilize_rho_from_dftfe(const FEM &femX,
                                  const FEM &femY,
                                  const FEM &femZ,
                                  Tensor3DMPI &rhoNodal,
                                  Tensor3DMPI &rhoGrid);

    void initializeRhoFromRadiusFile(const FEM &femX,
                                     const FEM &femY,
                                     const FEM &femZ,
                                     const std::vector<std::vector<double> > &nuclei,
                                     std::string filename,
                                     Tensor3DMPI &rhoNodal,
                                     Tensor3DMPI &rhoGrid);

    void initializeRhoFromFile(const FEM &femX,
                               const FEM &femY,
                               const FEM &femZ,
                               const std::vector<std::vector<double> > &nuclei,
                               const int numberAtomType,
                               const std::vector<std::string> &nodalFileX,
                               const std::vector<std::string> &nodalFileY,
                               const std::vector<std::string> &nodalFileZ,
                               const std::vector<std::string> &fieldFile,
                               Tensor3DMPI &rhoNodal,
                               Tensor3DMPI &rhoGrid);

    void initializePsi(const FEM &femX,
                       const FEM &femY,
                       const FEM &femZ,
                       const std::vector<std::vector<double> > &nuclei,
                       Tensor3DMPI &psiNodal);

    void initializePsiFromFile(const FEM &femX,
                               const FEM &femY,
                               const FEM &femZ,
                               const std::vector<std::vector<double> > &nuclei,
                               std::string nodalFileX,
                               std::string nodalFileY,
                               std::string nodalFileZ,
                               std::string fieldFile,
                               int option,
                               Tensor3DMPI &psiNodal);

    void initializePsiFromFile(const FEM &femX,
                               const FEM &femY,
                               const FEM &femZ,
                               const std::vector<std::vector<double> > &nuclei,
                               const int numberAtomType,
                               const std::vector<std::string> &nodalFileX,
                               const std::vector<std::string> &nodalFileY,
                               const std::vector<std::string> &nodalFileZ,
                               const std::vector<std::string> &fieldFile,
                               Tensor3DMPI &psiNodal);

    void initializeLocPSPGrid(const FEM &femX,
                              const FEM &femY,
                              const FEM &femZ,
                              const std::vector<std::vector<double> > &nuclei,
                              double cutoffRadius,
                              Tensor3DMPI &localPSP);

};

#endif //TUCKER_TENSOR_KSDFT_INITIALGUESS_H
