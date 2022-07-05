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

#include "ChebyShevFilter.h"
#include "../blas_lapack/clinalg.h"
#include "../utils/Utils.h"
#include "Mult.h"
#include <slepcbv.h>

/* The algorithm please refer to Algorithm 4.3 in
 * Zhou, Y., Saad, Y., Tiago, M. L., &Chelikowsky, J. R. (2006).
 * Parallel Self-Consistent-Field Calculations via Chebyshev-Filtered Subspace Acceleration (Unpublished).
 * Retrieved from http://www-users.cs.umn.edu/~saad/PDF/umsi-2006-101.pdf
 * For more detail.*/

void ChebyShevFilter::computeFilteredSubspace(ProjectHamiltonianSparse &H,
                                              AX &ax,
                                              Mat &X,
                                              int basisSize,
                                              int m,
                                              double a,
                                              double b,
                                              double a0,
                                              int num_X_blocks) {

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    PetscInt XM, XN, Xm, CM;
    MatGetSize(X,
               &XM,
               &XN);
    MatGetLocalSize(X,
                    &Xm,
                    PETSC_NULL);
    MatGetSize(H.C_nloc,
               &CM,
               PETSC_NULL);
    std::vector<PetscInt> columns_each_chunk(num_X_blocks,
                                             XN / num_X_blocks);
    for (int j = 0; j < XN % num_X_blocks; ++j) {
        columns_each_chunk[j] += 1;
    }

    PetscScalar *X_data;
    MatDenseGetArray(X,
                     &X_data);

    for (int i = 0; i < num_X_blocks; ++i) {
        double e = (b - a) / 2;
        double c = (b + a) / 2;
        double sigma = e / (a0 - c);
        double sigma1 = sigma;
        double gamma = 2.0 / sigma1;

        double alpha = sigma1 / e;
        double beta = alpha * (-c);
        double sigma2, delta;
        PetscInt num_entries_chunk = Xm * columns_each_chunk[i];
        Mat Xp, Yp, HlocXp, CnlocXp, Ynewp;
        MatCreateDense(PETSC_COMM_WORLD,
                       PETSC_DECIDE,
                       PETSC_DECIDE,
                       XM,
                       columns_each_chunk[i],
                       X_data,
                       &Xp);
        MatAssemblyBegin(Xp,
                         MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(Xp,
                       MAT_FINAL_ASSEMBLY);
        MatCreateDense(PETSC_COMM_WORLD,
                       PETSC_DECIDE,
                       PETSC_DECIDE,
                       XM,
                       columns_each_chunk[i],
                       PETSC_NULL,
                       &HlocXp);
        MatZeroEntries(HlocXp);
        MatAssemblyBegin(HlocXp,
                         MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(HlocXp,
                       MAT_FINAL_ASSEMBLY);
        ax.perform_ax(Xp, HlocXp);
        MatMatMult(H.C_nloc,
                   Xp,
                   MAT_INITIAL_MATRIX,
                   PETSC_DEFAULT,
                   &CnlocXp);
        MPI_Barrier(MPI_COMM_WORLD);
        MatMatMult(H.C_nloc_trans,
                   CnlocXp,
                   MAT_INITIAL_MATRIX,
                   PETSC_DEFAULT,
                   &Yp);
        MPI_Barrier(MPI_COMM_WORLD);
        MatAXPY(Yp,
                1.0,
                HlocXp,
                SAME_NONZERO_PATTERN);
        MatScale(Yp,
                 alpha);
        MatAXPY(Yp,
                beta,
                Xp,
                SAME_NONZERO_PATTERN);

        MatDuplicate(Yp,
                     MAT_DO_NOT_COPY_VALUES,
                     &Ynewp);
        for (int i = 2; i <= m; ++i) {
            sigma2 = 1.0 / (gamma - sigma);
            alpha = 2.0 * sigma2 / e;
            beta = -c * alpha;
            delta = -sigma * sigma2;
            // Ynew = (2*sigma2/e)*(HY-cY)-sigma*sigma2*X
            MatZeroEntries(HlocXp);
            MatAssemblyBegin(HlocXp, MAT_FINAL_ASSEMBLY);
            MatAssemblyEnd(HlocXp, MAT_FINAL_ASSEMBLY);
            ax.perform_ax(Yp, HlocXp);
            MatMatMult(H.C_nloc,
                       Yp,
                       MAT_REUSE_MATRIX,
                       PETSC_DEFAULT,
                       &CnlocXp);
            MatMatMult(H.C_nloc_trans,
                       CnlocXp,
                       MAT_REUSE_MATRIX,
                       PETSC_DEFAULT,
                       &Ynewp);
            MatAXPY(Ynewp,
                    1.0,
                    HlocXp,
                    SAME_NONZERO_PATTERN);
            MatScale(Ynewp,
                     alpha);
            MatAXPY(Ynewp,
                    beta,
                    Yp,
                    SAME_NONZERO_PATTERN);
            MatAXPY(Ynewp,
                    delta,
                    Xp,
                    SAME_NONZERO_PATTERN);
            MatCopy(Yp,
                    Xp,
                    SAME_NONZERO_PATTERN);
            MatCopy(Ynewp,
                    Yp,
                    SAME_NONZERO_PATTERN);
            sigma = sigma2;
        }

        PetscScalar *Xp_data;
        MatDenseGetArray(Xp,
                         &Xp_data);
        for (PetscInt j = 0; j < num_entries_chunk; ++j) X_data[j] = Xp_data[j];
        MatDenseRestoreArray(Xp,
                             &Xp_data);

        MatDestroy(&Xp);
        MatDestroy(&Yp);
        MatDestroy(&HlocXp);
        MatDestroy(&CnlocXp);
        MatDestroy(&Ynewp);
        X_data += num_entries_chunk;
    }

    MatDenseRestoreArray(X,
                         &X_data);
}