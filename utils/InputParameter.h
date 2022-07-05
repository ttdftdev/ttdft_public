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

#ifndef TUCKER_TENSOR_KSDFT_INPUTPARAMETER_H
#define TUCKER_TENSOR_KSDFT_INPUTPARAMETER_H

#include <string>
#include <vector>
#include "../dft/SeparableHamiltonian.h"

enum RhoIGType {
    HYDROGEN, FEM_SINGLEATOM_DATA, FEM_READ_IN, RADIALDATA, TUCKER_READ_IN, DFTFE
};

class InputParameter {
public:

    InputParameter(char *filename);

    void PrintParameter();

    // default value is set to be FEMDATA for compatability with old inputs
    RhoIGType using_initial_guess_electron_density = RhoIGType::FEM_SINGLEATOM_DATA;
    std::string ig_fem_x_electron_density_filename = "femRhoNodalR20X.txt";
    std::string ig_fem_y_electron_density_filename = "femRhoNodalR20Y.txt";
    std::string ig_fem_z_electron_density_filename = "femRhoNodalR20Z.txt";
    std::string ig_electron_density_3d_filename = "rhoNodalR20.txt";
    std::string ig_electron_density_radius_filename = "single_rho.txt";

    int using_initial_guess_wavefunction = 0;
    std::string ig_fem_x_wavefunction_filename = "femRhoNodalR20X.txt";
    std::string ig_fem_y_wavefunction_filename = "femRhoNodalR20Y.txt";
    std::string ig_fem_z_wavefunction_filename = "femRhoNodalR20Z.txt";
    std::string ig_wavefunction_3d_filename = "psiNodalR20.txt";
    std::string ig_wavefunction_radius_filename = "";

    bool is_output_electron_density = false;
    std::vector<int> output_electron_rank_idx;

    char *filename;
    bool chebFlag;
    int numberElementsX, numberElementsY, numberElementsZ;
    int numberNonLocalElementsX, numberNonLocalElementsY, numberNonLocalElementsZ;
    std::string quadRule, quadRuleElectro;
//  std::string elementType;
    int number_nodes_per_element;
    std::string meshType;
    double domainStart, domainEnd, domainElectroStart, domainElectroEnd;
    int numberInnerElements;
    double innerDomainSize;
    double coarsingFactor;
    double smearingTemerature;
    double alpha;
    int polynomialDegree;
    int chebyshev_restart_times_first = 15;
    int chebyshev_restart_times_other = 1;
    bool is_calculation_restart = false;
    int start_from_rank_index = 0;
    int start_from_scf_iter = 0;
    int rhoRankX, rhoRankY, rhoRankZ;
    int veffRankX, veffRankY, veffRankZ;
    std::vector<int> tuckerRankX, tuckerRankY, tuckerRankZ;
    int rankNloc;
    int rankEnergy;
    bool is_using_monopole_boundary;
    bool is_using_larger_domain;
    std::string hartree_mesh_type = "adaptive";
    double hartree_domain_start_x, hartree_domain_end_x;
    double hartree_domain_start_y, hartree_domain_end_y;
    double hartree_domain_start_z, hartree_domain_end_z;
    int hartree_domain_num_additional_elements;
    double hartree_domain_coarsing_factor;
    //bool usingKernelExpansion;
    std::vector<int> which_using_kernel_expansion;
    std::string omegafile, alphafile;
    double Asquare;
    double hartreeTolerance = 1.0e-9;
    int maxHartreeIter = 5000;
    std::string poisson_omegafile, poisson_alphafile;
    double poisson_Asquare;
    int numLanczosIter;
    int maxScfIter;
    int numberHistory;
    double scfTol;
    double nonloca_radius_delta_x = 2.7937501, nonloca_radius_delta_y = 2.7937501, nonloca_radius_delta_z = 2.7937501;

    bool is_using_localization_tucker;
    bool is_using_localization_cheby;

    bool is_break_apart_energies = true;

    std::string system_name;
    int numEigenValues;

    // temp variables
    bool using_fixed_basis = false;
    std::vector<int> fixing_basis_checking_size;// = 20;
    std::vector<double> fixing_basis_norm_tolerance;// = 1.0e-3;
    SeparableSCFType scf_separable_type = SeparableSCFType::ANDERSON;
    double scf_tol_1d;
    int scf_max_iter_1d;
    double scf_alpha_1d = 0.5;
    int scf_history_1d = 10;
    bool is_print_out_basis_function;
//  bool using_input_fixed_basis = false;
    std::string filename_fixed_basis_x, filename_fixed_basis_y, filename_fixed_basis_z;
    std::vector<int> which_using_input_fixed_basis;
    bool is_using_block_technique;
    int number_blocks;
    bool is_wavefn_ig_fixed = false;

    bool read_in_mpi_data = false;

};

#endif //TUCKER_TENSOR_KSDFT_INPUTPARAMETER_H
