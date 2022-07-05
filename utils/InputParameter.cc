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
#include <sstream>
#include <iostream>
#include "InputParameter.h"

InputParameter::InputParameter(char *filename) : filename(filename) {
    std::string test;
    std::ifstream fin(filename);

    while (std::getline(fin,
                        test)) {
        if (test[0] != '#' && (!test.empty())) {
            std::stringstream ssin(test);
            std::string s1, s2;
            std::getline(ssin,
                         s1,
                         '=');
            if (s1 == "numEleX") {
                ssin >> numberElementsX;
            } else if (s1 == "numEleY") {
                ssin >> numberElementsY;
            } else if (s1 == "numEleZ") {
                ssin >> numberElementsZ;
            } else if (s1 == "quadrtureRule") {
                ssin >> quadRule;
            } else if (s1 == "quadRuleElectro") {
                ssin >> quadRuleElectro;
            } else if (s1 == "eleType") {
                std::string elementType;
                ssin >> elementType;
                if (elementType.compare("linear") == 0)
                    number_nodes_per_element = 2;
                else if (elementType.compare("quadratic") == 0)
                    number_nodes_per_element = 3;
                else if (elementType.compare("cubic") == 0)
                    number_nodes_per_element = 4;
                else if (elementType.compare("quartic") == 0)
                    number_nodes_per_element = 5;
                else if (elementType.compare("quintic") == 0)
                    number_nodes_per_element = 6;
                else {
                    std::stringstream input_element_type(elementType);
                    input_element_type >> number_nodes_per_element;
                }
            } else if (s1 == "meshType") {
                ssin >> meshType;
            } else if (s1 == "innerElements") {
                ssin >> numberInnerElements;
            } else if (s1 == "innerDomainSize") {
                ssin >> innerDomainSize;
            } else if (s1 == "coarsing ratio") {
                ssin >> coarsingFactor;
            } else if (s1 == "domainStart") {
                ssin >> domainStart;
            } else if (s1 == "domainEnd") {
                ssin >> domainEnd;
            } else if (s1 == "domainElectroStart") {
                ssin >> domainElectroStart;
            } else if (s1 == "domainElectroEnd") {
                ssin >> domainElectroEnd;
            } else if (s1 == "smearing temperature") {
                ssin >> smearingTemerature;
            } else if (s1 == "decomposition rank rho x") {
                ssin >> rhoRankX;
            } else if (s1 == "decomposition rank rho y") {
                ssin >> rhoRankY;
            } else if (s1 == "decomposition rank rho z") {
                ssin >> rhoRankZ;
            } else if (s1 == "decomposition rank veff x") {
                ssin >> veffRankX;
            } else if (s1 == "decomposition rank veff y") {
                ssin >> veffRankY;
            } else if (s1 == "decomposition rank veff z") {
                ssin >> veffRankZ;
            } else if (s1 == "rankNloc") {
                ssin >> rankNloc;
            } else if (s1 == "rankEnergy") {
                ssin >> rankEnergy;
            } else if (s1 == "tucker rank X") {
                char temp;
                ssin >> temp;
                int rank;
                while (ssin >> rank >> temp) {
                    tuckerRankX.push_back(rank);
                }
            } else if (s1 == "tucker rank Y") {
                char temp;
                ssin >> temp;
                int rank;
                while (ssin >> rank >> temp) {
                    tuckerRankY.push_back(rank);
                }
            } else if (s1 == "tucker rank Z") {
                char temp;
                ssin >> temp;
                int rank;
                while (ssin >> rank >> temp) {
                    tuckerRankZ.push_back(rank);
                }
            } /*else if (s1 == "using kernel expansion") {
        ssin >> usingKernelExpansion;
      } */else if (s1 == "which using kernel expansion") {
                char temp;
                ssin >> temp;
                int rank_num;
                while (ssin >> rank_num >> temp) {
                    which_using_kernel_expansion.push_back(rank_num);
                }
            } else if (s1 == "omega file") {
                ssin >> omegafile;
            } else if (s1 == "alpha file") {
                ssin >> alphafile;
            } else if (s1 == "Asquare") {
                ssin >> Asquare;
            } else if (s1 == "Poisson solver tolerance") {
                ssin >> hartreeTolerance;
            } else if (s1 == "max Poisson solver iteration") {
                ssin >> maxHartreeIter;
            } else if (s1 == "poisson omega file") {
                ssin >> poisson_omegafile;
            } else if (s1 == "poisson alpha file") {
                ssin >> poisson_alphafile;
            } else if (s1 == "poisson Asquare") {
                ssin >> poisson_Asquare;
            } else if (s1 == "using Chebyshev filter") {
                ssin >> chebFlag;
            } else if (s1 == "max iteration steps for Lanczos iteration") {
                ssin >> numLanczosIter;
            } else if (s1 == "polynomial degree for Chebyshev filter") {
                ssin >> polynomialDegree;
            } else if (s1 == "max iteration steps for total SCF iteration") {
                ssin >> maxScfIter;
            } else if (s1 == "alpha") {
                ssin >> alpha;
            } else if (s1 == "history") {
                ssin >> numberHistory;
            } else if (s1 == "SCF tolerance") {
                ssin >> scfTol;
            } else if (s1 == "number elements of nonlocal X") {
                ssin >> numberNonLocalElementsX;
            } else if (s1 == "number elements of nonlocal Y") {
                ssin >> numberNonLocalElementsY;
            } else if (s1 == "number elements of nonlocal Z") {
                ssin >> numberNonLocalElementsZ;
            } else if (s1 == "system") {
                ssin >> system_name;
            } else if (s1 == "number eigenvalues") {
                ssin >> numEigenValues;
            } else if (s1 == "tucker basis localization") {
                ssin >> is_using_localization_tucker;
            } else if (s1 == "chebyshev filtered wavefunction localization") {
                ssin >> is_using_localization_cheby;
            } else if (s1 == "nonlocal radius delta x") {
                ssin >> nonloca_radius_delta_x;
            } else if (s1 == "nonlocal radius delta y") {
                ssin >> nonloca_radius_delta_y;
            } else if (s1 == "nonlocal radius delta z") {
                ssin >> nonloca_radius_delta_z;
            } else if (s1 == "is break apart energies") {
                ssin >> is_break_apart_energies;
            } else if (s1 == "number of additional elements") {
                ssin >> hartree_domain_num_additional_elements;
            } else if (s1 == "hartree domain coarsing factor") {
                ssin >> hartree_domain_coarsing_factor;
            } else if (s1 == "hartree domain start x") {
                ssin >> hartree_domain_start_x;
            } else if (s1 == "hartree domain end x") {
                ssin >> hartree_domain_end_x;
            } else if (s1 == "hartree domain start y") {
                ssin >> hartree_domain_start_y;
            } else if (s1 == "hartree domain end y") {
                ssin >> hartree_domain_end_y;
            } else if (s1 == "hartree domain start z") {
                ssin >> hartree_domain_start_z;
            } else if (s1 == "hartree domain end z") {
                ssin >> hartree_domain_end_z;
            } else if (s1 == "using initial guess files for electron density") {
                std::string rho_ig_type;
                ssin >> rho_ig_type;
                if (rho_ig_type == "fem_single_atom") {
                    using_initial_guess_electron_density = RhoIGType::FEM_SINGLEATOM_DATA;
                } else if (rho_ig_type == "fem_previous_calculation") {
                    using_initial_guess_electron_density = RhoIGType::FEM_READ_IN;
                } else if (rho_ig_type == "radial") {
                    using_initial_guess_electron_density = RhoIGType::RADIALDATA;
                } else if (rho_ig_type == "tucker_previous") {
                    using_initial_guess_electron_density = RhoIGType::TUCKER_READ_IN;
                } else if (rho_ig_type == "dftfe") {
                    using_initial_guess_electron_density = RhoIGType::DFTFE;
                } else {
                    using_initial_guess_electron_density = RhoIGType::HYDROGEN;
                }
            } else if (s1 == "initial guess electron density fem x filename") {
                ssin >> ig_fem_x_electron_density_filename;
            } else if (s1 == "initial guess electron density fem y filename") {
                ssin >> ig_fem_y_electron_density_filename;
            } else if (s1 == "initial guess electron density fem z filename") {
                ssin >> ig_fem_z_electron_density_filename;
            } else if (s1 == "initial guess 3d electron density filename") {
                ssin >> ig_electron_density_3d_filename;
            } else if (s1 == "initial guess electron density radius filename") {
                ssin >> ig_electron_density_radius_filename;
            } else if (s1 == "using initial guess files for wavefunction") {
                ssin >> using_initial_guess_wavefunction;
            } else if (s1 == "initial guess wavefunction fem x filename") {
                ssin >> ig_fem_x_wavefunction_filename;
            } else if (s1 == "initial guess wavefunction fem y filename") {
                ssin >> ig_fem_y_wavefunction_filename;
            } else if (s1 == "initial guess wavefunction fem z filename") {
                ssin >> ig_fem_z_wavefunction_filename;
            } else if (s1 == "initial guess 3d wavefunction filename") {
                ssin >> ig_wavefunction_3d_filename;
            } else if (s1 == "initial guess wavefunction radius filename") {
                ssin >> ig_wavefunction_radius_filename;
            } else if (s1 == "output final electron density") {
                ssin >> is_output_electron_density;
            } else if (s1 == "output which rank") {
                char temp;
                ssin >> temp;
                int rank_idx;
                while (ssin >> rank_idx >> temp) {
                    output_electron_rank_idx.push_back(rank_idx);
                }
                // TEMP VARIABLES START FROM HERE
            } else if (s1 == "using fixed 1d basis") {
                ssin >> using_fixed_basis;
            } else if (s1 == "largest fixed basis checking size") {
                char temp;
                ssin >> temp;
                int rank_idx;
                while (ssin >> rank_idx >> temp) {
                    fixing_basis_checking_size.push_back(rank_idx);
                }
            } else if (s1 == "norm of electron density difference tolerance for fixed basis") {
                char temp;
                ssin >> temp;
                double rank_idx;
                while (ssin >> rank_idx >> temp) {
                    fixing_basis_norm_tolerance.push_back(rank_idx);
                }
            } else if (s1 == "mixing scheme for 1d") {
                std::string mix_scheme;
                ssin >> mix_scheme;
                if (mix_scheme == "none") {
                    scf_separable_type = SeparableSCFType::NONE;
                } else if (mix_scheme == "simple") {
                    scf_separable_type = SeparableSCFType::SIMPLE;
                } else if (mix_scheme == "periodic_anderson") {
                    scf_separable_type = SeparableSCFType::PERIODIC_ANDERSON;
                } else {
                    scf_separable_type = SeparableSCFType::ANDERSON;
                }
            } else if (s1 == "scf tol for 1d") {
                ssin >> scf_tol_1d;
            } else if (s1 == "scf max iter for 1d") {
                ssin >> scf_max_iter_1d;
            } else if (s1 == "scf alpha for 1d") {
                ssin >> scf_alpha_1d;
            } else if (s1 == "scf history size for 1d") {
                ssin >> scf_history_1d;
            } else if (s1 == "printing out basis functions") {
                ssin >> is_print_out_basis_function;
            } else if (s1 == "which rank number using input fixed basis") {
                char temp;
                ssin >> temp;
                int rank_num;
                while (ssin >> rank_num >> temp) {
                    which_using_input_fixed_basis.push_back(rank_num);
                }
            } else if (s1 == "input fixed basis x") {
                ssin >> filename_fixed_basis_x;
            } else if (s1 == "input fixed basis y") {
                ssin >> filename_fixed_basis_y;
            } else if (s1 == "input fixed basis z") {
                ssin >> filename_fixed_basis_z;
            } else if (s1 == "using block technique") {
                ssin >> is_using_block_technique;
            } else if (s1 == "number blocks") {
                ssin >> number_blocks;
            } else if (s1 == "is guess fixed") {
                ssin >> is_wavefn_ig_fixed;
            } else if (s1 == "chebyshev restart times first") {
                ssin >> chebyshev_restart_times_first;
            } else if (s1 == "chebyshev restart times others") {
                ssin >> chebyshev_restart_times_other;
            } else if (s1 == "is calculation restart") {
                ssin >> is_calculation_restart;
            } else if (s1 == "start from rank index") {
                ssin >> start_from_rank_index;
            } else if (s1 == "start from scf iter") {
                ssin >> start_from_scf_iter;
            }
        }
    }
    fin.close();
}

void InputParameter::PrintParameter() {
    std::cout << "====================printing out the parameters====================" << std::endl;
    std::cout << "number elements X: " << numberElementsX << std::endl;
    std::cout << "number elements Y: " << numberElementsY << std::endl;
    std::cout << "number elements Z: " << numberElementsZ << std::endl;
    std::cout << "number nonlocal elements X: " << numberNonLocalElementsX << std::endl;
    std::cout << "number nonlocal elements Y: " << numberNonLocalElementsY << std::endl;
    std::cout << "number nonlocal elements Z: " << numberNonLocalElementsZ << std::endl;
    std::cout << "nonlocal radius delta x: " << nonloca_radius_delta_x << std::endl;
    std::cout << "nonlocal radius delta y: " << nonloca_radius_delta_y << std::endl;
    std::cout << "nonlocal radius delta z: " << nonloca_radius_delta_z << std::endl;
    std::cout << "initial guess used: " << using_initial_guess_electron_density << std::endl;
    std::cout << "quadrature rule: " << quadRule << std::endl;
    std::cout << "electro quadrature rule: " << quadRuleElectro << std::endl;
    std::cout << "number nodes per element (element type): " << number_nodes_per_element << std::endl;
    std::cout << "Kohn-Sham domain: (" << domainStart << ", " << domainEnd << ")" << std::endl;
    if (which_using_kernel_expansion.size() != 0) {
        std::cout << "using kernel expansion at: ";
        for (int i = 0; i < which_using_kernel_expansion.size(); ++i) {
            std::cout << which_using_kernel_expansion[i] << " ";
        };
        std::cout << std::endl;
        std::cout << "omega file: " << omegafile << std::endl;
        std::cout << "alpha file: " << alphafile << std::endl;
        std::cout << "Asquare: " << Asquare << std::endl;
        std::cout << "Kernel expansion integration domain: (" << domainElectroStart << ", " << domainElectroEnd << ")"
                  << std::endl;
    } else {
        std::cout << "omega file for poisson domain: " << poisson_omegafile << std::endl;
        std::cout << "alpha file for poisson domain: " << poisson_alphafile << std::endl;
        std::cout << "Asquare for poisson domain: " << poisson_Asquare << std::endl;
        std::cout << "tolerance for Poisson's solver for Hartree potential: " << hartreeTolerance << std::endl;
        std::cout << "max iterations for Poisson's solver for Hartree potential: " << maxHartreeIter << std::endl;
        std::cout << "number of addtional elements for larger domain: " << hartree_domain_num_additional_elements
                  << std::endl;
        std::cout << "coarsing factor for hartree domain: " << hartree_domain_coarsing_factor << std::endl;
        std::cout << "domain start for hartree x: (" << hartree_domain_start_x << ", " << hartree_domain_end_x << ")"
                  << std::endl;
        std::cout << "domain start for hartree y: (" << hartree_domain_start_y << ", " << hartree_domain_end_y << ")"
                  << std::endl;
        std::cout << "domain start for hartree z: (" << hartree_domain_start_z << ", " << hartree_domain_end_z << ")"
                  << std::endl;
    }
    std::cout << "mesh type: " << meshType << std::endl;
    if (meshType != "uniform") {
        std::cout << "number of inner elements: " << numberInnerElements << std::endl;
        std::cout << "inner domain size: " << innerDomainSize << std::endl;
        std::cout << "coarsing factor: " << coarsingFactor << std::endl;
    }
    std::cout << "smearing temperature: " << smearingTemerature << std::endl;
    std::cout << "Anderson mixnig coefficient alpha: " << alpha << std::endl;
    std::cout << "using Chebyshev filter: " << chebFlag << std::endl;
    if (chebFlag == true) {
        std::cout << "Chebyshev polynomial degree: " << polynomialDegree << std::endl;
        std::cout << "max Lanczos iteration: " << numLanczosIter << std::endl;
    }
    std::cout << "decomposition rank for SE rho: (" << rhoRankX << ", " << rhoRankY << ", " << rhoRankZ
              << ")" << std::endl;
    std::cout << "decomposition rank for effective potential Veff: (" << veffRankX << ", " << veffRankY << ", "
              << veffRankZ << ")" << std::endl;
    std::cout << "Tucker ranks X: { ";
    for (auto &i: tuckerRankX) { std::cout << i << " "; }
    std::cout << "}" << std::endl;
    std::cout << "Tucker ranks Y: { ";
    for (auto &i: tuckerRankY) { std::cout << i << " "; }
    std::cout << "}" << std::endl;
    std::cout << "Tucker ranks Z: { ";
    for (auto &i: tuckerRankZ) { std::cout << i << " "; }
    std::cout << "}" << std::endl;
    std::cout << "decomposition rank for nonlocal fem: " << rankNloc << std::endl;
    std::cout << "max number of iterations for SCF: " << maxScfIter << std::endl;
    std::cout << "tolerance for SCF: " << scfTol << std::endl;
    std::cout << "mixing coefficient for Anderson: " << alpha << std::endl;
    std::cout << "mixing history steps for Anderson: " << numberHistory << std::endl;
    std::cout << "system: " << system_name << std::endl;
    std::cout << "number of eigenvalues to be computed: " << numEigenValues << std::endl;
    std::cout << "=================end of printing out the parameters=================" << std::endl;
}

