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

#ifndef FunctionalRayleighQuotientSeperable_H
#define FunctionalRayleighQuotientSeperable_H

#include "../../fem/FEM.h"
#include <petsc.h>
#include <TuckerMPI.hpp>

/**
 * @brief Seperable Functional for constructing nonlinear solver
 *
 * This class is used to construct Hamiltonian and Jacobian for non-linear solver in PETSc.
 */
class FunctionalRayleighQuotientSeperable {
public:
    FunctionalRayleighQuotientSeperable(const FEM &_femX,
                                        const FEM &_femY,
                                        const FEM &_femZ,
                                        const TuckerMPI::TuckerTensor &_tuckerDecomposedVeffMPI);

    const FEM &get_femX() const;

    const FEM &get_femY() const;

    const FEM &get_femZ() const;

    virtual void computeVectorizedForce(const std::vector<double> &nodalFieldsX,
                                        const std::vector<double> &nodalFieldsY,
                                        const std::vector<double> &nodalFieldsZ,
                                        double lagrangeMultiplier,
                                        std::vector<double> &F);

    virtual void generateHamiltonianGenericPotential(const std::vector<double> &nodalFieldsX,
                                                     const std::vector<double> &nodalFieldsY,
                                                     const std::vector<double> &nodalFieldsZ,
                                                     Mat &Hx,
                                                     Mat &Hy,
                                                     Mat &Hz,
                                                     Mat &Mx,
                                                     Mat &My,
                                                     Mat &Mz);

    void solveHamiltonianGenericPotential(Mat &Hx,
                                          Mat &Hy,
                                          Mat &Hz,
                                          Mat &Mx,
                                          Mat &My,
                                          Mat &Mz,
                                          const int rankTuckerBasisX,
                                          const int rankTuckerBasisY,
                                          const int rankTuckerBasisZ,
                                          std::vector<std::vector<double> > &eigX,
                                          std::vector<std::vector<double> > &eigY,
                                          std::vector<std::vector<double> > &eigZ);

    void solveGHEP(Mat &A,
                   Mat &B,
                   int numeigs,
                   std::vector<std::vector<double> > &solution);

    void solveGHEPLapack(Mat &A,
                         Mat &B,
                         int numeigs,
                         std::vector<std::vector<double> > &solution);

    void solveGHEP(std::vector<double> &A,
                   std::vector<double> &B,
                   int size,
                   int numeigs,
                   std::vector<std::vector<double> > &solution);

    void constructGHEP(const FEM &fem,
                       const std::vector<double> &v,
                       const double potConst,
                       std::vector<double> &H,
                       std::vector<double> &M);

    void constructGHEP(const FEM &fem,
                       const std::vector<double> &v,
                       const double potConst,
                       Mat &H,
                       Mat &M);

protected:
    const FEM &_femX;
    const FEM &_femY;
    const FEM &_femZ;
    const TuckerMPI::TuckerTensor &_tuckerDecomposedVeffMPI;
    std::shared_ptr<Tucker::Tensor> veff_core_seq;
    Tucker::Matrix *veff_ux, *veff_uy, *veff_uz;

    void computeIntegralPsiSquare(const std::vector<double> &psiQuadValuesX,
                                  const std::vector<double> &psiQuadValuesY,
                                  const std::vector<double> &psiQuadValuesZ,
                                  const std::vector<double> &DPsiQuadValuesX,
                                  const std::vector<double> &DPsiQuadValuesY,
                                  const std::vector<double> &DPsiQuadValuesZ,
                                  double &normPsiX,
                                  double &normPsiY,
                                  double &normPsiZ,
                                  double &normDPsiX,
                                  double &normDPsiY,
                                  double &normDPsiZ);

    void computeSeparablePotentialUsingTuckerVeff(const std::vector<double> &psiQuadValuesX,
                                                  const std::vector<double> &psiQuadValuesY,
                                                  const std::vector<double> &psiQuadValuesZ,
                                                  std::vector<double> &vx,
                                                  std::vector<double> &vy,
                                                  std::vector<double> &vz,
                                                  std::vector<double> &umatpsiSquare,
                                                  std::vector<double> &vmatpsiSquare,
                                                  std::vector<double> &wmatpsiSquare);

    void computeGlobalForceVector(const std::vector<double> &psiQuadValuesX,
                                  const std::vector<double> &psiQuadValuesY,
                                  const std::vector<double> &psiQuadValuesZ,
                                  const std::vector<double> &DPsiQuadValuesX,
                                  const std::vector<double> &DPsiQuadValuesY,
                                  const std::vector<double> &DPsiQuadValuesZ,
                                  std::vector<double> &FxShapeDerTimesPsiXDer,
                                  std::vector<double> &FyShapeDerTimesPsiYDer,
                                  std::vector<double> &FzShapeDerTimesPsiZDer,
                                  std::vector<double> &FxShapeTimesPsiX,
                                  std::vector<double> &FyShapeTimesPsiY,
                                  std::vector<double> &FzShapeTimesPsiZ,
                                  std::vector<std::vector<double> > &FxUTimesPsixTimesShp,
                                  std::vector<std::vector<double> > &FyVTimesPsiyTimesShp,
                                  std::vector<std::vector<double> > &FzWTimesPsizTimesShp);

    void computeOverlapIntegral(const std::vector<double> &vx,
                                const std::vector<double> &vy,
                                const std::vector<double> &vz,
                                const std::vector<double> &umatpsiSquare,
                                const std::vector<double> &vmatpsiSquare,
                                const std::vector<double> &wmatpsiSquare,
                                const std::vector<std::vector<double> > &FxTucker,
                                const std::vector<std::vector<double> > &FyTucker,
                                const std::vector<std::vector<double> > &FzTucker,
                                std::vector<std::vector<std::vector<double> > > &vxShpOverlap,
                                std::vector<std::vector<std::vector<double> > > &vyShpOverlap,
                                std::vector<std::vector<std::vector<double> > > &vzShpOverlap,
                                std::vector<std::vector<double> > &crossTermsXYVeffIntegral,
                                std::vector<std::vector<double> > &crossTermsYZVeffIntegral,
                                std::vector<std::vector<double> > &crossTermsXZVeffIntegral);

    void computeIntegralMatPsiSquare(const int rank,
                                     const FEM &fem,
                                     const std::vector<double> &mat,
                                     const std::vector<double> &psiQuadValues,
                                     std::vector<double> &matpsiSquare);

    void computeProductCoreMatIntMatPsiSquare(const int rankX,
                                              const int rankY,
                                              const int rankZ,
                                              const int *rankABCOrder,
                                              const std::vector<double> &core,
                                              const std::vector<double> &matA,
                                              const std::vector<double> &matpsiSquareB,
                                              const std::vector<double> &matpsiSquareC,
                                              std::vector<double> &vA);

    void computeShapeFuncTimesPsiandPsider(const FEM &fem,
                                           const std::vector<double> &psiQuadValues,
                                           const std::vector<double> &DPsiQuadValues,
                                           std::vector<std::vector<double> > &shapeFuncTimesPsider,
                                           std::vector<std::vector<double> > &shapeFuncTimesPsi);

    void computeMatTimesPsiTimesShp(const FEM &fem,
                                    const int rank,
                                    const std::vector<double> &psiQuadValues,
                                    const std::vector<double> &mat,
                                    std::vector<std::vector<std::vector<double> > > &matTimesPsiTimeShp);

    void sumValuesAtQuadPointsForGlobalForceVector(const FEM &fem,
                                                   const int rank,
                                                   const std::vector<std::vector<double> > &shpTimesPsi,
                                                   const std::vector<std::vector<double> > &shpderTimesPsider,
                                                   const std::vector<std::vector<std::vector<double> > > &matTimesPsiTimesShp,
                                                   std::vector<double> &FunShapeDerTimesPsiDer,
                                                   std::vector<double> &FuncShapeTimesPsi,
                                                   std::vector<std::vector<double> > &FuncMatTimesPsiTimesShape);

    void computevShpOverlap(const FEM &fem,
                            const std::vector<double> &v,
                            std::vector<std::vector<std::vector<double> > > &vShpOverlap);

    void computecrossTermsABVeffIntegral(const FEM &femA,
                                         const FEM &femB,
                                         const int rankX,
                                         const int rankY,
                                         const int rankZ,
                                         const int *rankMatABOrder,
                                         const std::vector<double> &core,
                                         const std::vector<std::vector<double> > &FATucker,
                                         const std::vector<std::vector<double> > &FBTucker,
                                         const std::vector<double> &matpsiSquare,
                                         std::vector<std::vector<double> > &crossTermsABVeffIntegral);

    void computeOneDimForce(const FEM &fem,
                            std::vector<double> &psiQuadValues,
                            std::vector<double> &DPsiQuadValues,
                            const double normPsiProduct,
                            const double normPsiDpsiOuterProduct,
                            const std::vector<double> &v,
                            const double lagrangeMultiplier,
                            std::vector<std::vector<double> > &oneDimForce);

    void sumOneDimForceOverQuadPoints(const FEM &fem,
                                      const std::vector<std::vector<double> > &oneDimForce,
                                      std::vector<double> &summedOneDimForce);

};

#endif
