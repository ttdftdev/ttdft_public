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

#include "SeparableHamiltonian.h"
#include "../blas_lapack/clinalg.h"
#include <algorithm>
#include <iomanip>
#include <deque>

extern PetscErrorCode formFunction(SNES snes,
                                   Vec x,
                                   Vec f,
                                   void *ctx);

extern PetscErrorCode formJacobian(SNES snes,
                                   Vec x,
                                   Mat jac,
                                   Mat B,
                                   void *ctx);

SeparableHamiltonian::SeparableHamiltonian(FunctionalRayleighQuotientSeperable *functional)
        : functional(functional),
          femX(functional->get_femX()),
          femY(functional->get_femY()),
          femZ(functional->get_femZ()) {
    nodalFieldX = std::vector<double>(functional->get_femX().getTotalNumberNodes(),
                                      0.0);
    nodalFieldY = std::vector<double>(functional->get_femY().getTotalNumberNodes(),
                                      0.0);
    nodalFieldZ = std::vector<double>(functional->get_femZ().getTotalNumberNodes(),
                                      0.0);
}

const std::vector<double> &SeparableHamiltonian::getNodalFieldX() const {
    return nodalFieldX;
}

const std::vector<double> &SeparableHamiltonian::getNodalFieldY() const {
    return nodalFieldY;
}

const std::vector<double> &SeparableHamiltonian::getNodalFieldZ() const {
    return nodalFieldZ;
}

FunctionalRayleighQuotientSeperable *SeparableHamiltonian::getFunctional() const {
    return functional;
}

const double SeparableHamiltonian::getLm() const {
    return lm;
}

void ComputeMixingConstants(const FEM &fem,
                            const std::deque<std::vector<double>> &rho_in,
                            const std::deque<std::vector<double>> &rho_out,
                            std::vector<double> &mixing_constants) {
    int history_size = rho_in.size();
    mixing_constants = std::vector<double>(history_size - 1,
                                           0.0);

    std::vector<double> coeff_matrix((history_size - 1) * (history_size - 1),
                                     0.0);
    std::vector<double> rhs_vector(history_size - 1,
                                   0.0);

    int vector_size = rho_out.back().size();

    std::vector<double> Fn(vector_size);
    for (int i = 0; i < Fn.size(); ++i) {
        Fn[i] = rho_out.back()[i] - rho_in.back()[i];
    }

    for (int m = 0; m < history_size - 1; ++m) {
        std::vector<double> Fnm(vector_size);
        for (int i = 0; i < Fnm.size(); ++i) {
            Fnm[i] = rho_out[history_size - m - 2][i] - rho_in[history_size - m - 2][i];
        }
        std::vector<double> diffnm(vector_size);
        for (int i = 0; i < diffnm.size(); ++i) {
            diffnm[i] = Fn[i] - Fnm[i];
        }
        for (int k = 0; k < history_size - 1; ++k) {
            std::vector<double> Fnk(vector_size);
            for (int i = 0; i < Fnk.size(); ++i) {
                Fnk[i] = rho_out[history_size - k - 2][i] - rho_in[history_size - k - 2][i];
            }
            std::vector<double> diffnk(vector_size);
            for (int i = 0; i < diffnk.size(); ++i) {
                diffnk[i] = Fn[i] - Fnk[i];
            }

            std::vector<double> temp(vector_size);
            for (int i = 0; i < diffnk.size(); ++i) {
                temp[i] = diffnk[i] * diffnm[i];
            }
            coeff_matrix[m + k * (history_size - 1)] = fem.integrate_by_nodal_values(temp);
        }
        std::vector<double> temp(vector_size);
        for (int i = 0; i < temp.size(); ++i) {
            temp[i] = diffnm[i] * Fn[i];
        }
        rhs_vector[m] = fem.integrate_by_nodal_values(temp);
    }
    int n = history_size - 1;
    std::vector<int> ipiv(n);
    int info;
    clinalg::dgesv_(n,
                    1,
                    coeff_matrix.data(),
                    n,
                    ipiv.data(),
                    rhs_vector.data(),
                    n,
                    &info);
    std::copy(rhs_vector.begin(),
              rhs_vector.end(),
              mixing_constants.begin());

}

void ComputeMixingConstants(const FEM &femX,
                            const FEM &femY,
                            const FEM &femZ,
                            const std::deque<std::vector<double>> &rho_in_x,
                            const std::deque<std::vector<double>> &rho_in_y,
                            const std::deque<std::vector<double>> &rho_in_z,
                            const std::deque<std::vector<double>> &rho_out_x,
                            const std::deque<std::vector<double>> &rho_out_y,
                            const std::deque<std::vector<double>> &rho_out_z,
                            std::vector<double> &mixing_constants) {
    int history_size = rho_in_x.size();
    mixing_constants = std::vector<double>(history_size - 1,
                                           0.0);

    std::vector<double> coeff_matrix((history_size - 1) * (history_size - 1),
                                     0.0);
    std::vector<double> rhs_vector(history_size - 1,
                                   0.0);

    int vector_size_x = rho_in_x.back().size();
    int vector_size_y = rho_in_y.back().size();
    int vector_size_z = rho_in_z.back().size();

    auto ComputeDifference = [](const std::vector<double> &x,
                                const std::vector<double> &y,
                                std::vector<double> &x_minus_y) {
        x_minus_y = std::vector<double>(x.size());
        std::transform(x.begin(),
                       x.end(),
                       y.begin(),
                       x_minus_y.begin(),
                       std::minus<double>());
    };

    auto ComputeDotIntegration = [](const FEM &fem,
                                    const std::vector<double> &x,
                                    const std::vector<double> &y) {
        std::vector<double> x_times_y(x.size());
        std::transform(x.begin(),
                       x.end(),
                       y.begin(),
                       x_times_y.begin(),
                       std::multiplies<double>());
        return fem.integrate_by_nodal_values(x_times_y);
    };

    std::vector<double> Fn_x, Fn_y, Fn_z;
    ComputeDifference(rho_out_x.back(),
                      rho_in_x.back(),
                      Fn_x);
    ComputeDifference(rho_out_y.back(),
                      rho_in_y.back(),
                      Fn_y);
    ComputeDifference(rho_out_z.back(),
                      rho_in_z.back(),
                      Fn_z);

    for (int m = 0; m < history_size - 1; ++m) {
        std::vector<double> Fnm_x, Fnm_y, Fnm_z;
        ComputeDifference(rho_out_x[history_size - m - 2],
                          rho_in_x[history_size - m - 2],
                          Fnm_x);
        ComputeDifference(rho_out_y[history_size - m - 2],
                          rho_in_y[history_size - m - 2],
                          Fnm_y);
        ComputeDifference(rho_out_z[history_size - m - 2],
                          rho_in_z[history_size - m - 2],
                          Fnm_z);

        std::vector<double> diffnm_x, diffnm_y, diffnm_z;
        ComputeDifference(Fn_x,
                          Fnm_x,
                          diffnm_x);
        ComputeDifference(Fn_y,
                          Fnm_y,
                          diffnm_y);
        ComputeDifference(Fn_z,
                          Fnm_z,
                          diffnm_z);

        for (int k = 0; k < history_size - 1; ++k) {
            std::vector<double> Fnk_x, Fnk_y, Fnk_z;
            ComputeDifference(rho_out_x[history_size - k - 2],
                              rho_in_x[history_size - k - 2],
                              Fnk_x);
            ComputeDifference(rho_out_y[history_size - k - 2],
                              rho_in_y[history_size - k - 2],
                              Fnk_y);
            ComputeDifference(rho_out_z[history_size - k - 2],
                              rho_in_z[history_size - k - 2],
                              Fnk_z);

            std::vector<double> diffnk_x, diffnk_y, diffnk_z;
            ComputeDifference(Fn_x,
                              Fnk_x,
                              diffnk_x);
            ComputeDifference(Fn_y,
                              Fnk_y,
                              diffnk_y);
            ComputeDifference(Fn_z,
                              Fnk_z,
                              diffnk_z);

            double diff_int_x = ComputeDotIntegration(femX,
                                                      diffnk_x,
                                                      diffnm_x);
            double diff_int_y = ComputeDotIntegration(femY,
                                                      diffnk_y,
                                                      diffnm_y);
            double diff_int_z = ComputeDotIntegration(femZ,
                                                      diffnk_z,
                                                      diffnm_z);

            coeff_matrix[m + k * (history_size - 1)] = diff_int_x * diff_int_y * diff_int_z;
        }

        double diff_Fn_int_x = ComputeDotIntegration(femX,
                                                     diffnm_x,
                                                     Fn_x);
        double diff_Fn_int_y = ComputeDotIntegration(femY,
                                                     diffnm_y,
                                                     Fn_y);
        double diff_Fn_int_z = ComputeDotIntegration(femZ,
                                                     diffnm_z,
                                                     Fn_z);
        rhs_vector[m] = diff_Fn_int_x * diff_Fn_int_y * diff_Fn_int_z;
    }
    int n = history_size - 1;
    int nrhs = 1;
    std::vector<int> ipiv(n);
    int info;
    clinalg::dgesv_(n,
                    1,
                    coeff_matrix.data(),
                    n,
                    ipiv.data(),
                    rhs_vector.data(),
                    n,
                    &info);
    std::copy(rhs_vector.begin(),
              rhs_vector.end(),
              mixing_constants.begin());

}

void ComputeAndersonMixedField(const std::deque<std::vector<double>> &rho_in,
                               const std::deque<std::vector<double>> &rho_out,
                               const std::vector<double> &mixing_constants,
                               const double alpha,
                               std::vector<double> &rho) {
    int vector_size = rho_in.back().size();
    int history_size = rho_in.size();

    const std::vector<double> &rho_in_n = rho_in.back();
    const std::vector<double> &rho_out_n = rho_out.back();

    std::vector<double> input = rho_in_n;
    std::vector<double> output = rho_out_n;
    int inc = 1;

    for (int k = 0; k < mixing_constants.size(); ++k) {
        const std::vector<double> &rho_in_k = rho_in[history_size - k - 2];
        const std::vector<double> &rho_out_k = rho_out[history_size - k - 2];
        std::vector<double> input_k_minus_n(vector_size);
        std::transform(rho_in_k.begin(),
                       rho_in_k.end(),
                       rho_in_n.begin(),
                       input_k_minus_n.begin(),
                       std::minus<double>());
        clinalg::daxpy_(vector_size,
                        mixing_constants[k],
                        input_k_minus_n.data(),
                        inc,
                        input.data(),
                        inc);
        std::vector<double> output_k_minus_n(vector_size);
        std::transform(rho_out_k.begin(),
                       rho_out_k.end(),
                       rho_out_n.begin(),
                       output_k_minus_n.begin(),
                       std::minus<double>());
        clinalg::daxpy_(vector_size,
                        mixing_constants[k],
                        output_k_minus_n.data(),
                        inc,
                        output.data(),
                        inc);

    }

    rho = output;
    double one_minus_alpha = 1 - alpha;
    clinalg::dscal_(vector_size,
                    alpha,
                    rho.data(),
                    inc);
    clinalg::daxpy_(vector_size,
                    one_minus_alpha,
                    input.data(),
                    inc,
                    rho.data(),
                    inc);
}

bool SeparableHamiltonian::solveSCF(const std::vector<double> &initialGuessX,
                                    const std::vector<double> &initialGuessY,
                                    const std::vector<double> &initialGuessZ,
                                    SeparableSCFType scf_type,
                                    double tolerance,
                                    int maxIter,
                                    double alpha,
                                    int number_history,
                                    int period) {
    bool is_converged = true;

    assert(nodalFieldX.size() == functional->get_femX().getTotalNumberNodes());
    assert(nodalFieldY.size() == functional->get_femY().getTotalNumberNodes());
    assert(nodalFieldZ.size() == functional->get_femZ().getTotalNumberNodes());

    std::copy(initialGuessX.begin(),
              initialGuessX.end(),
              nodalFieldX.begin());
    std::copy(initialGuessY.begin(),
              initialGuessY.end(),
              nodalFieldY.begin());
    std::copy(initialGuessZ.begin(),
              initialGuessZ.end(),
              nodalFieldZ.begin());

    const double boundary = 0.0;
    nodalFieldX.front() = boundary;
    nodalFieldX.back() = boundary;
    nodalFieldY.front() = boundary;
    nodalFieldY.back() = boundary;
    nodalFieldZ.front() = boundary;
    nodalFieldZ.back() = boundary;

    std::vector<std::vector<double> > tempGuessFieldX(1,
                                                      nodalFieldX);
    std::vector<std::vector<double> > tempGuessFieldY(1,
                                                      nodalFieldY);
    std::vector<std::vector<double> > tempGuessFieldZ(1,
                                                      nodalFieldZ);
    int iter = 0;
    Mat
            Hx, Hy, Hz, Mx, My, Mz;
    PetscInt
            mx = femX.getTotalNumberNodes() - 2, nx = femX.getTotalNumberNodes() - 2;
    PetscInt
            my = femY.getTotalNumberNodes() - 2, ny = femY.getTotalNumberNodes() - 2;
    PetscInt
            mz = femZ.getTotalNumberNodes() - 2, nz = femZ.getTotalNumberNodes() - 2;
    PetscInt
            nzx = 3 * mx, nzy = 3 * my, nzz = 3 * mz;
    MatCreateSeqDense(PETSC_COMM_SELF,
                      mx,
                      nx,
                      NULL,
                      &Hx);
    MatCreateSeqDense(PETSC_COMM_SELF,
                      mx,
                      nx,
                      NULL,
                      &Mx);
    MatCreateSeqDense(PETSC_COMM_SELF,
                      my,
                      ny,
                      NULL,
                      &Hy);
    MatCreateSeqDense(PETSC_COMM_SELF,
                      my,
                      ny,
                      NULL,
                      &My);
    MatCreateSeqDense(PETSC_COMM_SELF,
                      mz,
                      nz,
                      NULL,
                      &Hz);
    MatCreateSeqDense(PETSC_COMM_SELF,
                      mz,
                      nz,
                      NULL,
                      &Mz);

    std::deque<std::vector<double>> history_x_in;
    std::deque<std::vector<double>> history_y_in;
    std::deque<std::vector<double>> history_z_in;

    std::deque<std::vector<double>> history_x_out;
    std::deque<std::vector<double>> history_y_out;
    std::deque<std::vector<double>> history_z_out;

    std::cout << "maxIter: " << maxIter << std::endl;
    std::cout << "alpha: " << alpha << std::endl;
    std::cout << "number history: " << number_history << std::endl;
    std::cout << "mixing scheme: ";
    if (scf_type == SeparableSCFType::NONE) {
        std::cout << "no mixing";
    } else if (scf_type == SeparableSCFType::SIMPLE) {
        std::cout << "simple mixing";
    } else {
        std::cout << "anderson mixing scheme";
    }
    std::cout << std::endl;

    if (scf_type == SeparableSCFType::SIMPLE) {
        number_history = 1;
    }
    for (; iter < maxIter; ++iter) {
        // mixing
        if (iter == 0 || scf_type == SeparableSCFType::NONE) {
            // doing nothing
        } else if (iter == 1 || scf_type == SeparableSCFType::SIMPLE
                   || (scf_type == SeparableSCFType::PERIODIC_ANDERSON && (iter) % period != 0)) {
            // simple mixing
            for (int i = 0; i < nodalFieldX.size(); ++i) {
                nodalFieldX[i] = (1 - alpha) * history_x_in.back()[i] + alpha * history_x_out.back()[i];
            }
            for (int i = 0; i < nodalFieldY.size(); ++i) {
                nodalFieldY[i] = (1 - alpha) * history_y_in.back()[i] + alpha * history_y_out.back()[i];
            }
            for (int i = 0; i < nodalFieldZ.size(); ++i) {
                nodalFieldZ[i] = (1 - alpha) * history_z_in.back()[i] + alpha * history_z_out.back()[i];
            }
        } else {
            // anderson mixing
            std::vector<double> mixing_coefficients;
            ComputeMixingConstants(femX,
                                   femY,
                                   femZ,
                                   history_x_in,
                                   history_y_in,
                                   history_z_in,
                                   history_x_out,
                                   history_y_out,
                                   history_z_out,
                                   mixing_coefficients);
            ComputeAndersonMixedField(history_x_in,
                                      history_x_out,
                                      mixing_coefficients,
                                      alpha,
                                      nodalFieldX);
            ComputeAndersonMixedField(history_y_in,
                                      history_y_out,
                                      mixing_coefficients,
                                      alpha,
                                      nodalFieldY);
            ComputeAndersonMixedField(history_z_in,
                                      history_z_out,
                                      mixing_coefficients,
                                      alpha,
                                      nodalFieldZ);
        }

        functional->generateHamiltonianGenericPotential(nodalFieldX,
                                                        nodalFieldY,
                                                        nodalFieldZ,
                                                        Hx,
                                                        Hy,
                                                        Hz,
                                                        Mx,
                                                        My,
                                                        Mz);
        functional->solveHamiltonianGenericPotential(Hx,
                                                     Hy,
                                                     Hz,
                                                     Mx,
                                                     My,
                                                     Mz,
                                                     1,
                                                     1,
                                                     1,
                                                     tempGuessFieldX,
                                                     tempGuessFieldY,
                                                     tempGuessFieldZ);


        double err = 0;
        double norm = 0;
        double errx = 0, erry = 0, errz = 0, normx = 0, normy = 0, normz = 0;
        for (int i = 1; i < nodalFieldX.size() - 1; ++i) {
            err = err + (nodalFieldX[i] - tempGuessFieldX[0][i]) * (nodalFieldX[i] - tempGuessFieldX[0][i]);
            norm = norm + tempGuessFieldX[0][i] * tempGuessFieldX[0][i];

            errx = errx + (nodalFieldX[i] - tempGuessFieldX[0][i]) * (nodalFieldX[i] - tempGuessFieldX[0][i]);
            normx = normx + tempGuessFieldX[0][i] * tempGuessFieldX[0][i];
        }
        for (int i = 1; i < nodalFieldY.size() - 1; ++i) {
            err = err + (nodalFieldY[i] - tempGuessFieldY[0][i]) * (nodalFieldY[i] - tempGuessFieldY[0][i]);
            norm = norm + tempGuessFieldY[0][i] * tempGuessFieldY[0][i];

            erry = erry + (nodalFieldY[i] - tempGuessFieldY[0][i]) * (nodalFieldY[i] - tempGuessFieldY[0][i]);
            normy = normy + tempGuessFieldY[0][i] * tempGuessFieldY[0][i];
        }
        for (int i = 1; i < nodalFieldZ.size() - 1; ++i) {
            err = err + (nodalFieldZ[i] - tempGuessFieldZ[0][i]) * (nodalFieldZ[i] - tempGuessFieldZ[0][i]);
            norm = norm + tempGuessFieldZ[0][i] * tempGuessFieldZ[0][i];

            errz = errz + (nodalFieldZ[i] - tempGuessFieldZ[0][i]) * (nodalFieldZ[i] - tempGuessFieldZ[0][i]);
            normz = normz + tempGuessFieldZ[0][i] * tempGuessFieldZ[0][i];
        }
        err = std::sqrt(err) / std::sqrt(norm);

        errx = std::sqrt(errx) / std::sqrt(normx);
        erry = std::sqrt(erry) / std::sqrt(normy);
        errz = std::sqrt(errz) / std::sqrt(normz);

        // update history
        history_x_in.emplace_back(nodalFieldX);
        history_y_in.emplace_back(nodalFieldY);
        history_z_in.emplace_back(nodalFieldZ);

        if (history_x_in.size() > number_history) {
            history_x_in.pop_front();
        }
        if (history_y_in.size() > number_history) {
            history_y_in.pop_front();
        }
        if (history_z_in.size() > number_history) {
            history_z_in.pop_front();
        }

        history_x_out.emplace_back(tempGuessFieldX[0]);
        history_y_out.emplace_back(tempGuessFieldY[0]);
        history_z_out.emplace_back(tempGuessFieldZ[0]);

        if (history_x_out.size() > number_history) {
            history_x_out.pop_front();
        }
        if (history_y_out.size() > number_history) {
            history_y_out.pop_front();
        }
        if (history_z_out.size() > number_history) {
            history_z_out.pop_front();
        }

        nodalFieldX = tempGuessFieldX[0];
        nodalFieldY = tempGuessFieldY[0];
        nodalFieldZ = tempGuessFieldZ[0];

        std::cout << "iter " << iter << ":  " << err << ", " << errx << ", " << erry << ", " << errz << std::endl;
        if (err <= tolerance) {
            std::cout << "iter " << iter << ":  " << err << std::endl;
            break;
        }
        if (iter == (maxIter - 1)) {
            std::cout << "iter not converged at " << iter << std::endl;
            is_converged = false;
        }
    }
    MatDestroy(&Hx);
    MatDestroy(&Hy);
    MatDestroy(&Hz);
    MatDestroy(&Mx);
    MatDestroy(&My);
    MatDestroy(&Mz);

    return is_converged;

}





