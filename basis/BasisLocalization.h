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

//
// Created by iancclin on 2/20/18.
//

#ifndef TUCKER_TENSOR_KSDFT_BASISLOCALIZATION_H
#define TUCKER_TENSOR_KSDFT_BASISLOCALIZATION_H

#include <vector>
#include "../fem/FEM.h"

class BasisLocalization {
public:
    BasisLocalization(const std::vector<std::vector<double>> &nuclei,
                      const FEM &femX,
                      const FEM &femY,
                      const FEM &femZ);

    void Localize1D(std::vector<std::vector<double>> &basis_x,
                    std::vector<std::vector<double>> &basis_y,
                    std::vector<std::vector<double>> &basis_z,
                    std::vector<std::vector<double>> &localized_basis_x,
                    std::vector<std::vector<double>> &localized_basis_y,
                    std::vector<std::vector<double>> &localized_basis_z);

    void TruncateWithTolerance(std::vector<std::vector<double>> &basis_x,
                               std::vector<std::vector<double>> &basis_y,
                               std::vector<std::vector<double>> &basis_z,
                               double tolerance);

    static void ComputeCompactSupportNodeId(const std::vector<std::vector<double>> &basis_x,
                                            const std::vector<std::vector<double>> &basis_y,
                                            const std::vector<std::vector<double>> &basis_z,
                                            std::vector<std::pair<unsigned, unsigned>> &compact_support_nodeid_x,
                                            std::vector<std::pair<unsigned, unsigned>> &compact_support_nodeid_y,
                                            std::vector<std::pair<unsigned, unsigned>> &compact_support_nodeid_z);

    static void ComputeInteractingNodes(const std::vector<std::pair<unsigned, unsigned>> &compact_support_nodeid_x,
                                        const std::vector<std::pair<unsigned, unsigned>> &compact_support_nodeid_y,
                                        const std::vector<std::pair<unsigned, unsigned>> &compact_support_nodeid_z,
                                        std::vector<std::vector<unsigned>> &interacting_list_x,
                                        std::vector<std::vector<unsigned>> &interacting_list_y,
                                        std::vector<std::vector<unsigned>> &interacting_list_z);

private:
    const std::vector<std::vector<double> > &nuclei;
    const FEM &femX;
    const FEM &femY;
    const FEM &femZ;

    /**
     * @brief compute how many localized states will be computed corresponding to each atom
     */
    void GenerateNumberStatesPerCoordinate(int num_states_wanted,
                                           std::vector<int> &num_states_per_coordinate);

    /**
     * @brief convert the basis vectors stored in STL vector of STL vector into a Petsc matrix
     */
    void MatricizeBases(const std::vector<std::vector<double>> &basis,
                        Mat &L);

    /**
     * @brief construct matrix Kij = int{|x-R|^2*N_i(x)*N_j(x)}dx
     */
    void ConstructPenaltyKernel(const FEM &fem,
                                const double atom_coordination,
                                Mat &K,
                                double kernel_power = 2.0);

    void SolveEigenVectors(Mat &A,
                           const int num_eigenvectors,
                           std::vector<double> &eigenvectors);

    /**
     * @brief compute the coefficients for localization
     * @param basis orthogonal Tucker-tensor bases on FEM nodes
     * @param nuclei nuclei information including the coordinations
     * @param nuclei_coord used to indicate which direction will be comupted, 1:x, 2:y, 3:z
     * @param coefficients_vector output of the function, the coefficients for the bases rotation
     */
    void ComputeLocalizationCoefficients(const FEM &fem,
                                         const std::vector<std::vector<double>> &basis,
                                         const std::vector<std::vector<double>> &nuclei,
                                         int nuclei_coord,
                                         std::vector<double> &coefficients_vector);

    void ComputeLocalizationCoefficientByRank(const FEM &fem,
                                              const std::vector<std::vector<double>> &basis,
                                              int number_centers,
                                              double kernel_power,
                                              std::vector<double> &coefficients_vector);

    void number_of_states_creator(const FEM &fem,
                                  int rank,
                                  int number_localization_centers,
                                  std::vector<int> &number_of_states_per_center,
                                  std::vector<double> &center_coordinates);

    void BasisRotation(std::vector<std::vector<double>> &basis,
                       std::vector<double> &coefficients_vector,
                       std::vector<std::vector<double>> &localized_basis);

};

#endif //TUCKER_TENSOR_KSDFT_BASISLOCALIZATION_H
