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

#ifndef BLASLAPACK_TTDFT_H
#define BLASLAPACK_TTDFT_H

namespace clinalg {

// lapack
    void dpotri_(const char &uplo,
                 const int &n,
                 double *ma,
                 const int &lda,
                 const int &info);

    void dpotrf_(const char &uplo,
                 const int &n,
                 double *ma,
                 const int &lda,
                 const int &info);

    void dgemv_(const char &trans,
                const int &m,
                const int &n,
                const double &da,
                const double *ma,
                const int &lda,
                const double *vx,
                const int &incx,
                const double &db,
                double *vy,
                const int &incy);

    void dgemm_(const char &transa,
                const char &transb,
                const int &m,
                const int &n,
                const int &k,
                const double &da,
                double *ma,
                const int &lda,
                double *mb,
                const int &ldb,
                const double &dc,
                double *mc,
                const int &ldc);

    void dcopy_(const int &n,
                const double *vx,
                const int &incx,
                double *vy,
                const int &incy);

    double ddot_(const int &n,
                 const double *vx,
                 const int &incx,
                 const double *vy,
                 const int &incy);

    void daxpy_(const int &n,
                const double &da,
                const double *vx,
                const int &incx,
                double *vy,
                const int &incy);

    void dscal_(const int &n,
                const double &da,
                double *vx,
                const int &incx);

    double dlamch_(const char &cmach);

/// @brief compute selected e-vals and e-vecs of a symmetric generalized eigenvalue problem
    void dsygvx_(const int &itype,
                 const char &jobz,
                 const char &range,
                 const char &uplo,
                 const int &n,
                 double *ma,
                 const int &lda,
                 double *mb,
                 const int &ldb,
                 const double &dvl,
                 const double &dvu,
                 const int &il,
                 const int &iu,
                 const double &abstol,
                 int *m,
                 double *w,
                 double *z,
                 const int &ldz,
                 double *work,
                 int *lwork,
                 int *iwork,
                 int *ifail,
                 int *info);

/// @brief compute the solution X of A*X=B for general matrix A
    void dgesv_(const int &n,
                const int &nrhs,
                double *a,
                const int &lda,
                int *ipiv,
                double *b,
                const int &ldb,
                int *info);
}

#endif //BLASLAPACK_TTDFT_H
