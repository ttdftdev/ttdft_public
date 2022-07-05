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

#include "clinalg.h"

namespace blaslapack {
    extern "C" void dpotri_(const char *uplo,
                            const int *n,
                            double *ma,
                            const int *lda,
                            const int *info);
    extern "C" void dpotrf_(const char *uplo,
                            const int *n,
                            double *ma,
                            const int *lda,
                            const int *info);
    extern "C" void dgemv_(const char *trans,
                           const int *m,
                           const int *n,
                           const double *da,
                           const double *ma,
                           const int *lda,
                           const double *vx,
                           const int *incx,
                           const double *db,
                           double *vy,
                           const int *incy);
    extern "C" void dgemm_(const char *transa,
                           const char *transb,
                           const int *m,
                           const int *n,
                           const int *k,
                           const double *da,
                           const double *ma,
                           const int *lda,
                           const double *mb,
                           const int *ldb,
                           const double *dc,
                           double *mc,
                           const int *ldc);
    extern "C" void dcopy_(const int *n,
                           const double *vx,
                           const int *incx,
                           double *vy,
                           const int *incy);
    extern "C" double ddot_(const int *n,
                            const double *vx,
                            const int *incx,
                            const double *vy,
                            const int *incy);
    extern "C" void daxpy_(const int *n,
                           const double *da,
                           const double *vx,
                           const int *incx,
                           double *vy,
                           const int *incy);
    extern "C" void dscal_(const int *n,
                           const double *da,
                           double *vx,
                           const int *incx);
    extern "C" double dlamch_(const char *cmach);
    extern "C" void dsygvx_(const int *itype,
                            const char *jobz,
                            const char *range,
                            const char *uplo,
                            const int *n,
                            double *mz,
                            const int *lda,
                            double *mb,
                            const int *ldb,
                            const double *dvl,
                            const double *dvu,
                            const int *il,
                            const int *iu,
                            const double *abstol,
                            int *m,
                            double *w,
                            double *z,
                            const int *ldz,
                            double *work,
                            int *lwork,
                            int *iwork,
                            int *ifail,
                            int *info);
    extern "C" void dgesv_(const int *n,
                           const int *nrhs,
                           double *a,
                           const int *lda,
                           int *ipiv,
                           double *b,
                           const int *ldb,
                           int *info);
}

void clinalg::dpotri_(const char &uplo,
                      const int &n,
                      double *ma,
                      const int &lda,
                      const int &info) {
    blaslapack::dpotri_(&uplo,
                        &n,
                        ma,
                        &lda,
                        &info);
}

void clinalg::dpotrf_(const char &uplo,
                      const int &n,
                      double *ma,
                      const int &lda,
                      const int &info) {
    blaslapack::dpotrf_(&uplo,
                        &n,
                        ma,
                        &lda,
                        &info);
}

void clinalg::dgemv_(const char &trans,
                     const int &m,
                     const int &n,
                     const double &da,
                     const double *ma,
                     const int &lda,
                     const double *vx,
                     const int &incx,
                     const double &db,
                     double *vy,
                     const int &incy) {
    blaslapack::dgemv_(&trans,
                       &m,
                       &n,
                       &da,
                       ma,
                       &lda,
                       vx,
                       &incx,
                       &db,
                       vy,
                       &incy);
}

void clinalg::dgemm_(const char &transa,
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
                     const int &ldc) {
    blaslapack::dgemm_(&transa,
                       &transb,
                       &m,
                       &n,
                       &k,
                       &da,
                       ma,
                       &lda,
                       mb,
                       &ldb,
                       &dc,
                       mc,
                       &ldc);
}

void clinalg::dcopy_(const int &n,
                     const double *vx,
                     const int &incx,
                     double *vy,
                     const int &incy) {
    blaslapack::dcopy_(&n,
                       vx,
                       &incx,
                       vy,
                       &incy);
}

double clinalg::ddot_(const int &n,
                      const double *vx,
                      const int &incx,
                      const double *vy,
                      const int &incy) {
    return blaslapack::ddot_(&n,
                             vx,
                             &incx,
                             vy,
                             &incy);
}

void clinalg::daxpy_(const int &n,
                     const double &da,
                     const double *vx,
                     const int &incx,
                     double *vy,
                     const int &incy) {
    blaslapack::daxpy_(&n,
                       &da,
                       vx,
                       &incx,
                       vy,
                       &incy);
}

void clinalg::dscal_(const int &n,
                     const double &da,
                     double *vx,
                     const int &incx) {
    blaslapack::dscal_(&n,
                       &da,
                       vx,
                       &incx);
}

double clinalg::dlamch_(const char &cmach) {
    return blaslapack::dlamch_(&cmach);
}

void clinalg::dsygvx_(const int &itype,
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
                      int *info) {
    blaslapack::dsygvx_(&itype,
                        &jobz,
                        &range,
                        &uplo,
                        &n,
                        ma,
                        &lda,
                        mb,
                        &ldb,
                        &dvl,
                        &dvu,
                        &il,
                        &iu,
                        &abstol,
                        m,
                        w,
                        z,
                        &ldz,
                        work,
                        lwork,
                        iwork,
                        ifail,
                        info);
}

void clinalg::dgesv_(const int &n,
                     const int &nrhs,
                     double *a,
                     const int &lda,
                     int *ipiv,
                     double *b,
                     const int &ldb,
                     int *info) {
    blaslapack::dgesv_(&n,
                       &nrhs,
                       a,
                       &lda,
                       ipiv,
                       b,
                       &ldb,
                       info);
}