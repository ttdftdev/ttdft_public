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

#include <slepceps.h>
#include <iostream>
#include <iomanip>
#include <assert.h>
#include "EigenSolver.h"
#include "../blas_lapack/clinalg.h"
#include "../dft/ProjectHamiltonianSparse.h"

void EigenSolver::computeEigenPairsSeq(Mat &A,
                                       int numberEigenPairs,
                                       int numGlobalRows,
                                       std::vector<double> &eigVals,
                                       Mat &eigMat,
                                       bool lapack) {

    int taskId;
    MPI_Comm_rank(PETSC_COMM_WORLD,
                  &taskId);

    if (taskId == 0) {
        EPS
                eps;
        EPSCreate(PETSC_COMM_SELF,
                  &eps);
        EPSSetOperators(eps,
                        A,
                        NULL);
        EPSSetDimensions(eps,
                         numberEigenPairs,
                         PETSC_DEFAULT,
                         PETSC_DEFAULT);
        EPSSetTolerances(eps,
                         1e-10,
                         10000);
        EPSSetProblemType(eps,
                          EPS_HEP);

        if (lapack == 0) {
            EPSSetType(eps,
                       EPSKRYLOVSCHUR);
        } else {
            EPSSetType(eps,
                       EPSLAPACK);
            DS ds;
            EPSGetDS(eps,
                     &ds);
            DSSetMethod(ds,
                        2);
        }

        EPSSetWhichEigenpairs(eps,
                              EPS_SMALLEST_REAL);
        EPSSetFromOptions(eps);
        EPSSetUp(eps);
        EPSSolve(eps);

//    EPSView(eps, PETSC_VIEWER_STDOUT_SELF);

        PetscInt itrNum;
        EPSGetIterationNumber(eps,
                              &itrNum);
        EPSType soltype;
        EPSGetType(eps,
                   &soltype);
        PetscInt nconv;
        EPSGetConverged(eps,
                        &nconv);


        PetscScalar
                eigr, eigi, *eigMatData, *EiData;
        Mat
                Ei;
        MatDuplicate(eigMat,
                     MAT_DO_NOT_COPY_VALUES,
                     &Ei);
        MatDenseGetArray(eigMat,
                         &eigMatData);
        MatDenseGetArray(Ei,
                         &EiData);
        Vec
                vr, vi;
        MatCreateVecs(A,
                      &vr,
                      PETSC_NULL);
        MatCreateVecs(A,
                      &vi,
                      PETSC_NULL);
        if (nconv > 0) {
            if (taskId == 0) {
                std::cout << "        k          ||Ax-kx||/||kx|| " << std::endl;
                std::cout << "----------------- ------------------" << std::endl;
            }
            for (int i = 0; i < numberEigenPairs; ++i) {
                EPSGetEigenpair(eps,
                                i,
                                &eigr,
                                &eigi,
                                vr,
                                vi);
                eigVals[i] = eigr;
                if (taskId == 0) {
                    if (eigi != 0.0)
                        std::cout << "Imaginary part does not equal to zero" << std::endl;
                    else
                        printf("%dth e-vals:\t%.18e\n", i, eigr);
//                        std::cout << i << "th e-vals:\t" << std::setprecision(16) << eigr << std::endl;
                }
                PetscScalar *vrData, *viData;
                VecGetArray(vr,
                            &vrData);
                VecGetArray(vi,
                            &viData);
                PetscMemcpy(eigMatData + i * numGlobalRows,
                            vrData,
                            sizeof(PetscScalar) * numGlobalRows);
                PetscMemcpy(EiData + i * numGlobalRows,
                            viData,
                            sizeof(PetscScalar) * numGlobalRows);
                VecRestoreArray(vr,
                                &vrData);
                VecRestoreArray(vi,
                                &viData);
            }

        }
        MatDenseRestoreArray(eigMat,
                             &eigMatData);
        MatAssemblyBegin(eigMat,
                         MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(eigMat,
                       MAT_FINAL_ASSEMBLY);
        MatDenseRestoreArray(Ei,
                             &EiData);
        MatAssemblyBegin(Ei,
                         MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(Ei,
                       MAT_FINAL_ASSEMBLY);
        MatDestroy(&Ei);
        VecDestroy(&vi);
        VecDestroy(&vr);
        EPSDestroy(&eps);
    }
    double *eigMatData;
    MatDenseGetArray(eigMat,
                     &eigMatData);
    MPI_Bcast(eigMatData,
              numGlobalRows * numberEigenPairs,
              MPI_DOUBLE,
              0,
              PETSC_COMM_WORLD);
    MPI_Bcast(eigVals.data(),
              eigVals.size(),
              MPI_DOUBLE,
              0,
              PETSC_COMM_WORLD);
    MatDenseRestoreArray(eigMat,
                         &eigMatData);
    MatAssemblyBegin(eigMat,
                     MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(eigMat,
                   MAT_FINAL_ASSEMBLY);
}

void EigenSolver::computeUpperBoundWithLanczos(ProjectHamiltonianSparse &H,
                                               int maxLanczosIteration,
                                               double &upperBound,
                                               double &lowerBound) {

    int taskId;
    MPI_Comm_rank(MPI_COMM_WORLD,
                  &taskId);

    PetscInt dim;
    MatGetSize(H.H_loc,
               PETSC_NULL,
               &dim);
    Vec v, f, v0, vloc, vcnl, vnl;
    MatCreateVecs(H.H_loc,
                  &v,
                  &vloc);
    MatCreateVecs(H.C_nloc,
                  PETSC_NULL,
                  &vcnl);
    VecDuplicate(vloc,
                 &vnl);
    VecDuplicate(vloc,
                 &f);
    VecDuplicate(vloc,
                 &v0);
    VecSetRandom(v,
                 NULL);


    double alpha, beta;
    double normv;
    VecNorm(v,
            NORM_2,
            &normv);
    VecScale(v,
             1.0 / normv);

    MatMult(H.H_loc,
            v,
            vloc);
    MatMult(H.C_nloc,
            v,
            vcnl);
    MatMult(H.C_nloc_trans,
            vcnl,
            vnl);
    VecAXPBYPCZ(f,
                1.0,
                1.0,
                0.0,
                vloc,
                vnl);

    VecDot(f,
           v,
           &alpha);

    // compute fseq = fseq - alpha*v
    VecAXPY(f,
            -alpha,
            v);

    Mat TMat;
    MatCreateSeqAIJ(MPI_COMM_SELF,
                    maxLanczosIteration,
                    maxLanczosIteration,
                    3,
                    NULL,
                    &TMat);
    MatSetValue(TMat,
                0,
                0,
                alpha,
                INSERT_VALUES);

    for (int iterCount = 1; iterCount < maxLanczosIteration; ++iterCount) {
        VecNorm(f,
                NORM_2,
                &beta);
        VecScale(f,
                 1.0 / beta);

        VecCopy(v,
                v0);
        VecCopy(f,
                v);

        MatMult(H.H_loc,
                v,
                vloc);
        MatMult(H.C_nloc,
                v,
                vcnl);
        MatMult(H.C_nloc_trans,
                vcnl,
                vnl);
        VecAXPBYPCZ(f,
                    1.0,
                    1.0,
                    0.0,
                    vloc,
                    vnl);

        VecAXPY(f,
                -beta,
                v0);
        VecDot(v,
               f,
               &alpha);
        VecAXPY(f,
                -alpha,
                v);

        MatSetValue(TMat,
                    iterCount,
                    iterCount - 1,
                    beta,
                    INSERT_VALUES);
        MatSetValue(TMat,
                    iterCount - 1,
                    iterCount,
                    beta,
                    INSERT_VALUES);
        MatSetValue(TMat,
                    iterCount,
                    iterCount,
                    alpha,
                    INSERT_VALUES);
    }

    VecNorm(f,
            NORM_2,
            &beta);

    MatAssemblyBegin(TMat,
                     MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(TMat,
                   MAT_FINAL_ASSEMBLY);

    // Compute max and min e-value of the TMat
    double norm2TMat;
    EPS eps;
    EPSCreate(PETSC_COMM_SELF,
              &eps);
    EPSSetOperators(eps,
                    TMat,
                    NULL);
    EPSSetProblemType(eps,
                      EPS_HEP);

    // compute largest e-value
    EPSSetWhichEigenpairs(eps,
                          EPS_LARGEST_REAL);
    EPSSetDimensions(eps,
                     1,
                     PETSC_DEFAULT,
                     PETSC_DEFAULT);
    EPSSolve(eps);
    EPSGetEigenvalue(eps,
                     0,
                     &norm2TMat,
                     NULL);

    // compute smallest e-value
    EPSSetWhichEigenpairs(eps,
                          EPS_SMALLEST_REAL);
    EPSSetDimensions(eps,
                     1,
                     PETSC_DEFAULT,
                     PETSC_DEFAULT);
    EPSSolve(eps);
    EPSGetEigenvalue(eps,
                     0,
                     &lowerBound,
                     NULL);
    EPSDestroy(&eps);

    MatDestroy(&TMat);
    norm2TMat = std::abs(norm2TMat);

    upperBound = beta + norm2TMat;

    // just to ensure every proc has the same value
    MPI_Bcast(&upperBound,
              1,
              MPI_DOUBLE,
              0,
              MPI_COMM_WORLD);
    MPI_Bcast(&lowerBound,
              1,
              MPI_DOUBLE,
              0,
              MPI_COMM_WORLD);

    VecDestroy(&v);
    VecDestroy(&f);
    VecDestroy(&v0);
    VecDestroy(&vloc);
    VecDestroy(&vcnl);
    VecDestroy(&vnl);

}