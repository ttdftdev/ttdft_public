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

#include <petscmat.h>
#include <slepceps.h>
#include <algorithm>
#include "BasisLocalization.h"
#include "../blas_lapack/clinalg.h"

extern "C" {
int dgemm_(char *transa,
           char *transb,
           int *m,
           int *n,
           int *k,
           double *alpha,
           double *a,
           int *lda,
           double *b,
           int *ldb,
           double *beta,
           double *c,
           int *ldc);
}

BasisLocalization::BasisLocalization(const std::vector<std::vector<double>> &nuclei,
                                     const FEM &femX,
                                     const FEM &femY,
                                     const FEM &femZ) :
        nuclei(nuclei),
        femX(femX),
        femY(femY),
        femZ(femZ) {}

void BasisLocalization::Localize1D(std::vector<std::vector<double>> &basis_x,
                                   std::vector<std::vector<double>> &basis_y,
                                   std::vector<std::vector<double>> &basis_z,
                                   std::vector<std::vector<double>> &localized_basis_x,
                                   std::vector<std::vector<double>> &localized_basis_y,
                                   std::vector<std::vector<double>> &localized_basis_z) {

    std::vector<double> localize_coefficients_x, localize_coefficients_y, localize_coefficients_z;
    ComputeLocalizationCoefficients(femX,
                                    basis_x,
                                    nuclei,
                                    1,
                                    localize_coefficients_x);
    ComputeLocalizationCoefficients(femY,
                                    basis_y,
                                    nuclei,
                                    2,
                                    localize_coefficients_y);
    ComputeLocalizationCoefficients(femZ,
                                    basis_z,
                                    nuclei,
                                    3,
                                    localize_coefficients_z);

    BasisRotation(basis_x,
                  localize_coefficients_x,
                  localized_basis_x);
    BasisRotation(basis_y,
                  localize_coefficients_y,
                  localized_basis_y);
    BasisRotation(basis_z,
                  localize_coefficients_z,
                  localized_basis_z);
}

void BasisLocalization::GenerateNumberStatesPerCoordinate(int num_states_wanted,
                                                          std::vector<int> &num_states_per_coordinate) {
    int num_coordinate = num_states_per_coordinate.size();
    if (num_states_wanted <= num_coordinate) {
        num_states_per_coordinate = std::vector<int>(num_states_wanted,
                                                     0);
        for (int i = 0; i < num_states_wanted; ++i) {
            num_states_per_coordinate[i] = 1;
        }
    } else {
        int num_even_states = num_states_wanted / num_coordinate;
        int num_remaining_states = num_states_wanted % num_coordinate;

        num_states_per_coordinate = std::vector<int>(num_coordinate,
                                                     num_even_states);

        for (int i = 0; i < num_remaining_states; ++i) {
            num_states_per_coordinate[i] += 1;
        }
    }
}

void BasisLocalization::ConstructPenaltyKernel(const FEM &fem,
                                               const double atom_coordination,
                                               Mat &K,
                                               double kernel_power) {

    // compute weighting function |x - R|^2 at quad points
    std::vector<double> penalty_function_quad = fem.getPositionQuadPointValues();
    for (auto &p: penalty_function_quad) {
        double relative_distance = p - atom_coordination;
        p = std::pow(relative_distance,
                     kernel_power);
    }

    int num_nodes = fem.getTotalNumberNodes();
    int num_elements = fem.getNumberElements();
    int num_nodes_per_element = fem.getNumberNodesPerElement();
    int num_quad_per_elements = fem.getNumberQuadPointsPerElement();

    const std::vector<double> &weight_quad = fem.getWeightQuadPointValues();
    const std::vector<double> &jacobian_quad = fem.getJacobQuadPointValues();

    const std::vector<std::vector<double>> &shape_function = fem.getShapeFunctionAtQuadPoints();

    MatCreateSeqDense(PETSC_COMM_SELF,
                      num_nodes,
                      num_nodes,
                      PETSC_NULL,
                      &K);
    MatZeroEntries(K);

    double *K_data;
    MatDenseGetArray(K,
                     &K_data);
    for (int ele = 0; ele < num_elements; ++ele) {

        const double *penalty_function_quad_x_elementwise = penalty_function_quad.data() + ele * num_quad_per_elements;
        const double *weight_quad_x_elementwise = weight_quad.data() + ele * num_quad_per_elements;
        const double *jacobian_quad_x_elementwise = jacobian_quad.data() + ele * num_quad_per_elements;

        for (int node_i = 0; node_i < num_nodes_per_element; ++node_i) {
            int node_I = node_i + ele * (num_nodes_per_element - 1);
            for (int node_j = 0; node_j < num_nodes_per_element; ++node_j) {
                int node_J = node_j + ele * (num_nodes_per_element - 1);

                double integration_Ni_p_Nj = 0.0;
                for (int q = 0; q < num_quad_per_elements; ++q) {
                    integration_Ni_p_Nj += shape_function[node_i][q] * shape_function[node_j][q] *
                                           penalty_function_quad_x_elementwise[q] *
                                           weight_quad_x_elementwise[q] * jacobian_quad_x_elementwise[q];
                }

                K_data[node_I + node_J * num_nodes] += integration_Ni_p_Nj;
            }
        }
    }
    MatDenseRestoreArray(K,
                         &K_data);

}

void BasisLocalization::SolveEigenVectors(Mat &A,
                                          const int num_eigenvectors,
                                          std::vector<double> &eigenvectors) {
    EPS eps;
    EPSCreate(PETSC_COMM_SELF,
              &eps);
    EPSSetOperators(eps,
                    A,
                    NULL);
    EPSSetDimensions(eps,
                     num_eigenvectors,
                     PETSC_DEFAULT,
                     PETSC_DEFAULT);
    EPSSetProblemType(eps,
                      EPS_HEP);
    EPSSetType(eps,
               EPSLAPACK);
    EPSSetWhichEigenpairs(eps,
                          EPS_SMALLEST_REAL);
    EPSSetFromOptions(eps);
    EPSSetUp(eps);

    EPSSolve(eps);

    PetscInt size_vector;
    MatGetSize(A,
               PETSC_NULL,
               &size_vector);
    eigenvectors = std::vector<double>(num_eigenvectors * size_vector,
                                       0.0);

    Vec
            eigenvector;
    MatCreateVecs(A,
                  &eigenvector,
                  PETSC_NULL);
    std::vector<double>::iterator eigenvectors_iter = eigenvectors.begin();

    for (int i = 0; i < num_eigenvectors; ++i) {
        EPSGetEigenvector(eps,
                          i,
                          eigenvector,
                          PETSC_NULL);
        double *eigenvector_data;
        VecGetArray(eigenvector,
                    &eigenvector_data);
        eigenvectors_iter = std::copy(eigenvector_data,
                                      eigenvector_data + size_vector,
                                      eigenvectors_iter);
        VecRestoreArray(eigenvector,
                        &eigenvector_data);
    }
    VecDestroy(&eigenvector);

    EPSDestroy(&eps);
}

void BasisLocalization::MatricizeBases(const std::vector<std::vector<double>> &basis,
                                       Mat &L) {
    int mx = basis[0].size(), nx = basis.size();
    MatCreateSeqDense(PETSC_COMM_SELF,
                      mx,
                      nx,
                      PETSC_NULL,
                      &L);
    double *L_data;
    MatDenseGetArray(L,
                     &L_data);
    for (auto &i: basis) {
        L_data = std::copy(i.begin(),
                           i.end(),
                           L_data);
    }
    MatDenseRestoreArray(L,
                         &L_data);
}

void BasisLocalization::ComputeLocalizationCoefficients(const FEM &fem,
                                                        const std::vector<std::vector<double>> &basis,
                                                        const std::vector<std::vector<double>> &nuclei,
                                                        int nuclei_coord,
                                                        std::vector<double> &coefficients_vector) {
    int tucker_rank = basis.size();
    std::vector<double> non_coincide_coordinates;
    for (int i = 0; i < nuclei.size(); ++i) {
        double coordinate = nuclei[i][nuclei_coord];
        auto iter = std::find_if(non_coincide_coordinates.begin(),
                                 non_coincide_coordinates.end(),
                                 [&coordinate](const double &v) { return std::abs(coordinate - v) < 0.01; });
        if (iter == non_coincide_coordinates.end()) {
            non_coincide_coordinates.emplace_back(coordinate);
        }
    }

    std::vector<int> num_states_per_coordinate(non_coincide_coordinates.size());
    GenerateNumberStatesPerCoordinate(tucker_rank,
                                      num_states_per_coordinate);

    coefficients_vector = std::vector<double>(tucker_rank * tucker_rank);

    Mat L;
    MatricizeBases(basis,
                   L);
    std::vector<double>::iterator coefficients_vector_iter = coefficients_vector.begin();
    for (int i = 0; i < non_coincide_coordinates.size(); ++i) {
        Mat K;
        ConstructPenaltyKernel(fem,
                               non_coincide_coordinates[i],
                               K);
        Mat LtKL, KL;
        MatMatMult(K,
                   L,
                   MAT_INITIAL_MATRIX,
                   PETSC_DEFAULT,
                   &KL);
        MatTransposeMatMult(L,
                            KL,
                            MAT_INITIAL_MATRIX,
                            PETSC_DEFAULT,
                            &LtKL);

        std::vector<double> eigenvector;
        SolveEigenVectors(LtKL,
                          num_states_per_coordinate[i],
                          eigenvector);

        MatDestroy(&K);
        MatDestroy(&KL);
        MatDestroy(&LtKL);

        coefficients_vector_iter = std::copy(eigenvector.begin(),
                                             eigenvector.end(),
                                             coefficients_vector_iter);
    }
    MatDestroy(&L);
}

void BasisLocalization::number_of_states_creator(const FEM &fem,
                                                 int rank,
                                                 int number_localization_centers,
                                                 std::vector<int> &number_of_states_per_center,
                                                 std::vector<double> &center_coordinates) {
    number_of_states_per_center = std::vector<int>(number_localization_centers,
                                                   int(rank / number_localization_centers));
    center_coordinates = std::vector<double>(number_localization_centers,
                                             0.0);
    double domain_size = fem.get_domainEnd() - fem.get_domainStart();
    double interval = domain_size / (number_localization_centers + 1.0);
    double first_center_coordinate = fem.get_domainStart() + interval;
    for (int i = 0; i < rank % number_localization_centers; ++i) number_of_states_per_center[i] += 1;
    for (int i = 0; i < number_localization_centers; ++i) {
        center_coordinates[i] = first_center_coordinate + i * interval;
    }
}

void BasisLocalization::ComputeLocalizationCoefficientByRank(const FEM &fem,
                                                             const std::vector<std::vector<double>> &basis,
                                                             int number_centers,
                                                             double kernel_power,
                                                             std::vector<double> &coefficients_vector) {
    int tucker_rank = basis.size();

    std::vector<double> localized_center;
    std::vector<int> num_states_per_center;
    number_of_states_creator(fem,
                             tucker_rank,
                             number_centers,
                             num_states_per_center,
                             localized_center);

    coefficients_vector = std::vector<double>(tucker_rank * tucker_rank);

    Mat
            L;
    MatricizeBases(basis,
                   L);
    std::vector<double>::iterator coefficients_vector_iter = coefficients_vector.begin();
    for (int i = 0; i < localized_center.size(); ++i) {
        Mat K;
        ConstructPenaltyKernel(fem,
                               localized_center[i],
                               K,
                               kernel_power);
        Mat LtKL, KL;
        MatMatMult(K,
                   L,
                   MAT_INITIAL_MATRIX,
                   PETSC_DEFAULT,
                   &KL);
        MatTransposeMatMult(L,
                            KL,
                            MAT_INITIAL_MATRIX,
                            PETSC_DEFAULT,
                            &LtKL);

        std::vector<double> eigenvector;
        SolveEigenVectors(LtKL,
                          num_states_per_center[i],
                          eigenvector);

        MatDestroy(&K);
        MatDestroy(&KL);
        MatDestroy(&LtKL);

        coefficients_vector_iter = std::copy(eigenvector.begin(),
                                             eigenvector.end(),
                                             coefficients_vector_iter);
    }
    MatDestroy(&L);
}

void BasisLocalization::BasisRotation(std::vector<std::vector<double>> &basis,
                                      std::vector<double> &coefficients_vector,
                                      std::vector<std::vector<double>> &localized_basis) {
    std::vector<double> flatten_basis(basis.size() * basis[0].size());
    std::vector<double> flatten_localized_basis(basis.size() * basis[0].size());
    std::vector<double>::iterator flatten_basis_iter = flatten_basis.begin();
    for (int i = 0; i < basis.size(); ++i) {
        flatten_basis_iter = std::copy(basis[i].begin(),
                                       basis[i].end(),
                                       flatten_basis_iter);
    }

    char transa = 'N';
    char transb = 'N';
    int m = basis[0].size();
    int k = basis.size();
    int n = basis.size();
    double alpha = 1.0;
    int lda = m;
    int ldb = k;
    double beta = 0.0;
    int ldc = m;
    dgemm_(&transa,
           &transb,
           &m,
           &n,
           &k,
           &alpha,
           flatten_basis.data(),
           &lda,
           coefficients_vector.data(),
           &ldb,
           &beta,
           flatten_localized_basis.data(),
           &ldc);

    localized_basis = std::vector<std::vector<double>>(basis.size(),
                                                       std::vector<double>(basis[0].size()));
    for (int i = 0; i < basis.size(); ++i) {
        std::copy(flatten_localized_basis.begin() + i * m,
                  flatten_localized_basis.begin() + (i + 1) * m,
                  localized_basis[i].begin());
        double max_sign = 1.0;
        double max = std::abs(localized_basis[i][0]);
        for (int j = 1; j < localized_basis[i].size(); ++j) {
            if (std::abs(localized_basis[i][j]) > max) {
                max = std::abs(localized_basis[i][j]);
                max_sign = localized_basis[i][j] / max;
            }
        }
        if (max_sign < 0) {
            for (int j = 0; j < localized_basis[i].size(); ++j) {
                localized_basis[i][j] *= max_sign;
            }
        }
    }

}

void BasisLocalization::TruncateWithTolerance(std::vector<std::vector<double>> &basis_x,
                                              std::vector<std::vector<double>> &basis_y,
                                              std::vector<std::vector<double>> &basis_z,
                                              double tolerance) {
    PetscPrintf(PETSC_COMM_WORLD,
                "truncate tolerance: %.2e\n",
                tolerance);
    std::vector<int> num_zeros_x(basis_x.size(),
                                 0), num_zeros_y(basis_y.size(),
                                                 0), num_zeros_z(basis_z.size(),
                                                                 0);
    for (int i = 0; i < basis_x.size(); ++i) {
        for (int j = 0; j < basis_x[i].size(); ++j) {
            if (std::abs(basis_x[i][j]) <= tolerance) {
                basis_x[i][j] = 0;
                num_zeros_x[i] += 1;
            }
        }
    }

    for (int i = 0; i < basis_y.size(); ++i) {
        for (int j = 0; j < basis_y[i].size(); ++j) {
            if (std::abs(basis_y[i][j]) <= tolerance) {
                basis_y[i][j] = 0;
                num_zeros_y[i] += 1;
            }
        }
    }

    for (int i = 0; i < basis_z.size(); ++i) {
        for (int j = 0; j < basis_z[i].size(); ++j) {
            if (std::abs(basis_z[i][j]) <= tolerance) {
                basis_z[i][j] = 0;
                num_zeros_z[i] += 1;
            }
        }
    }
    PetscPrintf(PETSC_COMM_WORLD,
                "num zeros of truncated basis functions x: ");
    for (int i = 0; i < num_zeros_x.size(); ++i) {
        PetscPrintf(PETSC_COMM_WORLD,
                    "%d, ",
                    num_zeros_x[i]);
    }
    PetscPrintf(PETSC_COMM_WORLD,
                "\n");
    PetscPrintf(PETSC_COMM_WORLD,
                "num zeros of truncated basis functions y: ");
    for (int i = 0; i < num_zeros_y.size(); ++i) {
        PetscPrintf(PETSC_COMM_WORLD,
                    "%d, ",
                    num_zeros_y[i]);
    }
    PetscPrintf(PETSC_COMM_WORLD,
                "\n");
    PetscPrintf(PETSC_COMM_WORLD,
                "num zeros of truncated basis functions z: ");
    for (int i = 0; i < num_zeros_z.size(); ++i) {
        PetscPrintf(PETSC_COMM_WORLD,
                    "%d, ",
                    num_zeros_z[i]);
    }
    PetscPrintf(PETSC_COMM_WORLD,
                "\n");
}

void BasisLocalization::ComputeCompactSupportNodeId(const std::vector<std::vector<double>> &basis_x,
                                                    const std::vector<std::vector<double>> &basis_y,
                                                    const std::vector<std::vector<double>> &basis_z,
                                                    std::vector<std::pair<unsigned,
                                                            unsigned>> &compact_support_nodeid_x,
                                                    std::vector<std::pair<unsigned,
                                                            unsigned>> &compact_support_nodeid_y,
                                                    std::vector<std::pair<unsigned,
                                                            unsigned>> &compact_support_nodeid_z) {
    compact_support_nodeid_x.resize(basis_x.size());
    compact_support_nodeid_y.resize(basis_y.size());
    compact_support_nodeid_z.resize(basis_z.size());

    // end points are compact supported, no boundary check needed for p-1 and p+1
    auto check_front_back = [](const std::vector<std::vector<double>> &basis,
                               std::vector<std::pair<unsigned, unsigned>> &compact_support_nodeid) {
        for (int i = 0; i < basis.size(); ++i) {
            for (int p = 0; p < basis[i].size(); ++p) {
                if (std::abs(basis[i][p]) > 1.0e-16) {
                    compact_support_nodeid[i].first = p - 1;
                    break;
                }
            }
            for (int p = basis[i].size() - 1; p >= 0; --p) {
                if (std::abs(basis[i][p]) > 1.0e-16) {
                    compact_support_nodeid[i].second = p + 1;
                    break;
                }
            }
        }
    };

    check_front_back(basis_x,
                     compact_support_nodeid_x);
    check_front_back(basis_y,
                     compact_support_nodeid_y);
    check_front_back(basis_z,
                     compact_support_nodeid_z);

}

void BasisLocalization::ComputeInteractingNodes(const std::vector<std::pair<unsigned,
        unsigned>> &compact_support_nodeid_x,
                                                const std::vector<std::pair<unsigned,
                                                        unsigned>> &compact_support_nodeid_y,
                                                const std::vector<std::pair<unsigned,
                                                        unsigned>> &compact_support_nodeid_z,
                                                std::vector<std::vector<unsigned>> &interacting_list_x,
                                                std::vector<std::vector<unsigned>> &interacting_list_y,
                                                std::vector<std::vector<unsigned>> &interacting_list_z) {

    interacting_list_x = std::vector<std::vector<unsigned>>(compact_support_nodeid_x.size());
    interacting_list_y = std::vector<std::vector<unsigned>>(compact_support_nodeid_y.size());
    interacting_list_z = std::vector<std::vector<unsigned>>(compact_support_nodeid_z.size());

    // check if a and b are interacting
    auto check_interact = [](const std::pair<unsigned, unsigned> &a,
                             const std::pair<unsigned, unsigned> &b) -> bool {
        return !(a.second < b.first || a.first > b.second);
    };

    // lambda expression to loop over each basis
    auto create_interacting_list =
            [&check_interact](const std::vector<std::pair<unsigned, unsigned>> &compact_support_nodeid,
                              std::vector<std::vector<unsigned>> &interacting_list) {
                int rank = compact_support_nodeid.size();
                for (int i = 0; i < rank; ++i) {
                    for (int j = 0; j < rank; ++j) {
                        bool check = check_interact(compact_support_nodeid[i],
                                                    compact_support_nodeid[j]);
                        if (check_interact(compact_support_nodeid[i],
                                           compact_support_nodeid[j])) {
                            interacting_list[i].emplace_back(j);
                        }
                    }
                }
            };

    create_interacting_list(compact_support_nodeid_x,
                            interacting_list_x);
    create_interacting_list(compact_support_nodeid_y,
                            interacting_list_y);
    create_interacting_list(compact_support_nodeid_z,
                            interacting_list_z);
}
