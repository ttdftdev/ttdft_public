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

#include "FunctionalRayleighQuotientSeperable.h"
#include "../../tensor/TensorUtils.h"
#include "../../blas_lapack/clinalg.h"
#include <numeric>
#include <algorithm>
#include <iterator>
#include <slepceps.h>

/**
 * @brief for use of STL algorithms, start from start, increase each element by inc
 */
struct add_inc {
    add_inc()
            : start(0),
              inc(1) {}

    add_inc(int start_value)
            : start(start_value),
              inc(1) {}

    add_inc(int start_value,
            int increment)
            : start(start_value),
              inc(increment) {}

    void operator()(int &n) {
        n += start;
        start += inc;
    }

private:
    int start;
    int inc;
};

/**
 * @brief for use of STL algorithm, add a constant con to each value
 */
struct add_const {
    add_const(int constant) : con(constant) {}

    void operator()(int &n) {
        n += con;
    }

private:
    int con;
};

void solveGeneralizedEigVecFromNodalfield(const std::vector<double> &nodalField);

FunctionalRayleighQuotientSeperable::FunctionalRayleighQuotientSeperable(const FEM &_femX,
                                                                         const FEM &_femY,
                                                                         const FEM &_femZ,
//                                                                        const TuckerTensor &_tuckerDecomposedVeff,
                                                                         const TuckerMPI::TuckerTensor &_tuckerDecomposedVeffMPI)
        : _femX(_femX),
          _femY(_femY),
          _femZ(_femZ),
          veff_ux(_tuckerDecomposedVeffMPI.U[0]),
          veff_uy(_tuckerDecomposedVeffMPI.U[1]),
          veff_uz(_tuckerDecomposedVeffMPI.U[2]),
//          _tuckerDecomposedVeff(_tuckerDecomposedVeff),
          _tuckerDecomposedVeffMPI(_tuckerDecomposedVeffMPI) {
    Tucker::SizeArray rank_veff(3);
    rank_veff[0] = _tuckerDecomposedVeffMPI.G->getGlobalSize(0);
    rank_veff[1] = _tuckerDecomposedVeffMPI.G->getGlobalSize(1);
    rank_veff[2] = _tuckerDecomposedVeffMPI.G->getGlobalSize(2);
    veff_core_seq = std::shared_ptr<Tucker::Tensor>(new Tucker::Tensor(rank_veff));
    TensorUtils::allgatherTensor(_tuckerDecomposedVeffMPI.G,
                                 veff_core_seq.get());
}

/**
  *@brief Functional for solving 1D nonlinear equation
  *
  *@param femX FEM object in X direction
  *
  *@tuckerDecomposedVeff Tucker-decomposed 3D potential, with the form \f$ V(x,y,z) = \sum_{i,j,k} \sigma_{i,j,k} u_i (x) v_j(y) w_k(z) \f$, where \f$ i,j,k = 1, ..., r \f$. REMEMBER that the eigen vectors \f$ u_i, v_j, w_k \f$ ought to be on quadrature points instead of nodal points.
  */

/**
  *@brief compute \f$\int\Psi^2dx\f$
  *
  *do the integration
  *
  *@param psiQuadValuesX a vector contains values of \f$\Psi_x(x)\f$ at quadrature points
  *
  *@param DPsiQuadValuesX a vector contains values of \f$\frac{d}{dx}\Psi_x(x)\f$ at quadrature points
  *
  *@return normPsiX the evaluation of \f$\int\Psi_x^2dx\f$
  *
  *@return normDPsiX the evaluation of \f$\int(\frac{d}{dx}\Psi_x(x))^2dx\f$
  */
void FunctionalRayleighQuotientSeperable::computeIntegralPsiSquare(const std::vector<double> &psiQuadValuesX,
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
                                                                   double &normDPsiZ) {

    /// initialize the container with the values at quadrature points
    std::vector<double> psiQuadValuesSquareX(psiQuadValuesX);
    std::vector<double> psiQuadValuesSquareY(psiQuadValuesY);
    std::vector<double> psiQuadValuesSquareZ(psiQuadValuesZ);
    std::vector<double> DPsiQuadValuesSquareX(DPsiQuadValuesX);
    std::vector<double> DPsiQuadValuesSquareY(DPsiQuadValuesY);
    std::vector<double> DPsiQuadValuesSquareZ(DPsiQuadValuesZ);

    /// square each element in the container
    auto square = [](double &d) { d *= d; };
    std::for_each(psiQuadValuesSquareX.begin(),
                  psiQuadValuesSquareX.end(),
                  square);
    std::for_each(psiQuadValuesSquareY.begin(),
                  psiQuadValuesSquareY.end(),
                  square);
    std::for_each(psiQuadValuesSquareZ.begin(),
                  psiQuadValuesSquareZ.end(),
                  square);
    std::for_each(DPsiQuadValuesSquareX.begin(),
                  DPsiQuadValuesSquareX.end(),
                  square);
    std::for_each(DPsiQuadValuesSquareY.begin(),
                  DPsiQuadValuesSquareY.end(),
                  square);
    std::for_each(DPsiQuadValuesSquareZ.begin(),
                  DPsiQuadValuesSquareZ.end(),
                  square);

    /// evaluate \f$ \int\Psi_x^2dx\f$
    normPsiX = _femX.integrate_by_quad_values(psiQuadValuesSquareX);
    normPsiY = _femY.integrate_by_quad_values(psiQuadValuesSquareY);
    normPsiZ = _femZ.integrate_by_quad_values(psiQuadValuesSquareZ);

    /// evaluate \f$ \int(\frac{d}{dx}\Psi_x)^2dx\f$
    normDPsiX = _femX.integrate_inv_by_quad_values(DPsiQuadValuesSquareX);
    normDPsiY = _femY.integrate_inv_by_quad_values(DPsiQuadValuesSquareY);
    normDPsiZ = _femZ.integrate_inv_by_quad_values(DPsiQuadValuesSquareZ);
}

/**
  *@breif compute the local potential
  *
  *Given \f$ V(x,y,z) = \sum_{i,j,k} \sigma_{i,j,k} u_i (x) v_j(y) w_k(z) \f$, where \f$ i,j,k = 1, ..., r \f$, calculate teh following return values.
  *
  *@param psiQuadValuesX \f$ \Psi_x (x_q) \f$, where \f$ x_q \f$'s are quadrature points
  *
  *@param vx return values storing \f$ \int V_x (x) dx \int V_y (y) \Psi_y(y)^2 dy \int V_z(z) \Psi_z(z)^2 \f$
  *
  *@param umatpsiSquare return values sotring \f$ \int u_i (x) \Psi_x (x)^2 dx \f$
  */
void
FunctionalRayleighQuotientSeperable::computeSeparablePotentialUsingTuckerVeff(const std::vector<double> &psiQuadValuesX,
                                                                              const std::vector<double> &psiQuadValuesY,
                                                                              const std::vector<double> &psiQuadValuesZ,
                                                                              std::vector<double> &vx,
                                                                              std::vector<double> &vy,
                                                                              std::vector<double> &vz,
                                                                              std::vector<double> &umatpsiSquare,
                                                                              std::vector<double> &vmatpsiSquare,
                                                                              std::vector<double> &wmatpsiSquare) {

    const std::vector<double> sigma(veff_core_seq->data(),
                                    veff_core_seq->data() + veff_core_seq->getNumElements());
    const std::vector<double> umat(veff_ux->data(),
                                   veff_ux->data() + veff_ux->getNumElements());
    const std::vector<double> vmat(veff_uy->data(),
                                   veff_uy->data() + veff_uy->getNumElements());
    const std::vector<double> wmat(veff_uz->data(),
                                   veff_uz->data() + veff_uz->getNumElements());
    int rankVeffX = veff_uz->ncols();
    int rankVeffY = veff_uz->ncols();
    int rankVeffZ = veff_uz->ncols();

    umatpsiSquare = std::vector<double>(rankVeffX,
                                        0.0);
    vmatpsiSquare = std::vector<double>(rankVeffY,
                                        0.0);
    wmatpsiSquare = std::vector<double>(rankVeffZ,
                                        0.0);
    computeIntegralMatPsiSquare(rankVeffX,
                                _femX,
                                umat,
                                psiQuadValuesX,
                                umatpsiSquare);
    computeIntegralMatPsiSquare(rankVeffY,
                                _femY,
                                vmat,
                                psiQuadValuesY,
                                vmatpsiSquare);
    computeIntegralMatPsiSquare(rankVeffZ,
                                _femZ,
                                wmat,
                                psiQuadValuesZ,
                                wmatpsiSquare);

    Tucker::Matrix umat_psi_square_tucker(1,
                                          rankVeffX);
    Tucker::Matrix vmat_psi_square_tucker(1,
                                          rankVeffY);
    Tucker::Matrix wmat_psi_square_tucker(1,
                                          rankVeffZ);
    std::copy(umatpsiSquare.begin(),
              umatpsiSquare.end(),
              umat_psi_square_tucker.data());
    std::copy(vmatpsiSquare.begin(),
              vmatpsiSquare.end(),
              vmat_psi_square_tucker.data());
    std::copy(wmatpsiSquare.begin(),
              wmatpsiSquare.end(),
              wmat_psi_square_tucker.data());

    vx = std::vector<double>(_femX.getTotalNumberQuadPoints(),
                             0.0);
    vy = std::vector<double>(_femY.getTotalNumberQuadPoints(),
                             0.0);
    vz = std::vector<double>(_femZ.getTotalNumberQuadPoints(),
                             0.0);

    Tucker::Tensor *temp;
    Tucker::Tensor *reconstructedTensor;

    temp = veff_core_seq.get();
    reconstructedTensor = Tucker::ttm(temp,
                                      2,
                                      &wmat_psi_square_tucker);
    temp = reconstructedTensor;
    reconstructedTensor = Tucker::ttm(temp,
                                      1,
                                      &vmat_psi_square_tucker);
    Tucker::MemoryManager::safe_delete(temp);
    temp = reconstructedTensor;
    reconstructedTensor = Tucker::ttm(temp,
                                      0,
                                      _tuckerDecomposedVeffMPI.U[0]);
    Tucker::MemoryManager::safe_delete(temp);
    std::copy(reconstructedTensor->data(),
              reconstructedTensor->data() + reconstructedTensor->getNumElements(),
              vx.begin());
    Tucker::MemoryManager::safe_delete(reconstructedTensor);

    temp = veff_core_seq.get();
    reconstructedTensor = Tucker::ttm(temp,
                                      0,
                                      &umat_psi_square_tucker);
    temp = reconstructedTensor;
    reconstructedTensor = Tucker::ttm(temp,
                                      2,
                                      &wmat_psi_square_tucker);
    Tucker::MemoryManager::safe_delete(temp);
    temp = reconstructedTensor;
    reconstructedTensor = Tucker::ttm(temp,
                                      1,
                                      _tuckerDecomposedVeffMPI.U[1]);
    Tucker::MemoryManager::safe_delete(temp);
    std::copy(reconstructedTensor->data(),
              reconstructedTensor->data() + reconstructedTensor->getNumElements(),
              vy.begin());
    Tucker::MemoryManager::safe_delete(reconstructedTensor);

    temp = veff_core_seq.get();
    reconstructedTensor = Tucker::ttm(temp,
                                      0,
                                      &umat_psi_square_tucker);
    temp = reconstructedTensor;
    reconstructedTensor = Tucker::ttm(temp,
                                      1,
                                      &vmat_psi_square_tucker);
    Tucker::MemoryManager::safe_delete(temp);
    temp = reconstructedTensor;
    reconstructedTensor = Tucker::ttm(temp,
                                      2,
                                      _tuckerDecomposedVeffMPI.U[2]);
    Tucker::MemoryManager::safe_delete(temp);
    std::copy(reconstructedTensor->data(),
              reconstructedTensor->data() + reconstructedTensor->getNumElements(),
              vz.begin());
    Tucker::MemoryManager::safe_delete(reconstructedTensor);
}

/**
 * @brief copmute the following integrals for building Jacobian matrix
 * @param FxShapeDerTimesPsiXDer computing \f$ \int \frac{dN_I}{dx} \frac{d \Psi_x}{dx} dx \f$
 * @param FyShapeDerTimesPsiYDer computing \f$ \int \frac{dN_J}{dy} \frac{d \Psi_y}{dy} dy \f$
 * @param FzShapeDerTimesPsiZDer computing \f$ \int \frac{dN_K}{dz} \frac{d \Psi_z}{dz} dz \f$
 * @param FxShapeTimesPsiX computing \f$ \int N_I \Psi_x dx \f$
 * @param FyShapeTimesPsiY computing \f$ \int N_J \Psi_y dy \f$
 * @param FzShapeTimesPsiZ computing \f$ \int N_K \Psi_z dz \f$
 * @param FxUTimesPsixTimesShp computing \f$ \int u_i N_I \Psi_x dx \f$
 * @param FyVTimesPsiyTimesShp computing \f$ \int u_j N_J \Psi_y dy \f$
 * @param FzWTimesPsizTimesShp computing \f$ \int u_k N_K \Psi_z dz \f$
 */
void FunctionalRayleighQuotientSeperable::computeGlobalForceVector(const std::vector<double> &psiQuadValuesX,
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
                                                                   std::vector<std::vector<double> > &FzWTimesPsizTimesShp) {

    FxShapeDerTimesPsiXDer = std::vector<double>(_femX.getTotalNumberNodes(),
                                                 0.0);
    FyShapeDerTimesPsiYDer = std::vector<double>(_femY.getTotalNumberNodes(),
                                                 0.0);
    FzShapeDerTimesPsiZDer = std::vector<double>(_femZ.getTotalNumberNodes(),
                                                 0.0);
    FxShapeTimesPsiX = std::vector<double>(_femX.getTotalNumberNodes(),
                                           0.0);
    FyShapeTimesPsiY = std::vector<double>(_femY.getTotalNumberNodes(),
                                           0.0);
    FzShapeTimesPsiZ = std::vector<double>(_femZ.getTotalNumberNodes(),
                                           0.0);

    std::vector<std::vector<double> > shpderXTimesPsiXder(_femX.getNumberNodesPerElement(),
                                                          _femX.getWeightQuadPointValues());
    std::vector<std::vector<double> > shpderYTimesPsiYder(_femY.getNumberNodesPerElement(),
                                                          _femY.getWeightQuadPointValues());
    std::vector<std::vector<double> > shpderZTimesPsiZder(_femZ.getNumberNodesPerElement(),
                                                          _femZ.getWeightQuadPointValues());

    std::vector<std::vector<double> > shpTimesPsiX(shpderXTimesPsiXder);
    std::vector<std::vector<double> > shpTimesPsiY(shpderYTimesPsiYder);
    std::vector<std::vector<double> > shpTimesPsiZ(shpderZTimesPsiZder);

    // This part is computing the the following quadrature values for each node
    // shpderXTimesPsiXder = weightQuadPointValuesX .* invJacobianQuadPointValuesX .* DPsiQuadValuesX .* DShapeFunctionX(repeatedly on each element)
    // shpTimesPsiX = shpderXTimesPsiXder .* jacobianQuadPointValuesX .* psiQuadValuesX .* shapeFunctionX(repeatedly on each element)
    computeShapeFuncTimesPsiandPsider(_femX,
                                      psiQuadValuesX,
                                      DPsiQuadValuesX,
                                      shpderXTimesPsiXder,
                                      shpTimesPsiX);
    computeShapeFuncTimesPsiandPsider(_femY,
                                      psiQuadValuesY,
                                      DPsiQuadValuesY,
                                      shpderYTimesPsiYder,
                                      shpTimesPsiY);
    computeShapeFuncTimesPsiandPsider(_femZ,
                                      psiQuadValuesZ,
                                      DPsiQuadValuesZ,
                                      shpderZTimesPsiZder,
                                      shpTimesPsiZ);

    Tucker::Tensor *core = Tucker::MemoryManager::safe_new<Tucker::Tensor>(_tuckerDecomposedVeffMPI.G->getGlobalSize());
    TensorUtils::allgatherTensor(_tuckerDecomposedVeffMPI.G,
                                 core);

    const std::vector<double> sigma(core->data(),
                                    core->data() + core->getNumElements());
    const std::vector<double> umat(_tuckerDecomposedVeffMPI.U[0]->data(),
                                   _tuckerDecomposedVeffMPI.U[0]->data() +
                                   _tuckerDecomposedVeffMPI.U[0]->getNumElements());
    const std::vector<double> vmat(_tuckerDecomposedVeffMPI.U[1]->data(),
                                   _tuckerDecomposedVeffMPI.U[1]->data() +
                                   _tuckerDecomposedVeffMPI.U[1]->getNumElements());
    const std::vector<double> wmat(_tuckerDecomposedVeffMPI.U[2]->data(),
                                   _tuckerDecomposedVeffMPI.U[2]->data() +
                                   _tuckerDecomposedVeffMPI.U[2]->getNumElements());
    int rankVeffX = _tuckerDecomposedVeffMPI.U[0]->ncols();
    int rankVeffY = _tuckerDecomposedVeffMPI.U[1]->ncols();
    int rankVeffZ = _tuckerDecomposedVeffMPI.U[2]->ncols();
    Tucker::MemoryManager::safe_delete(core);

    FxUTimesPsixTimesShp = std::vector<std::vector<double> >(_femX.getTotalNumberNodes(),
                                                             std::vector<double>(rankVeffX,
                                                                                 0.0));
    FyVTimesPsiyTimesShp = std::vector<std::vector<double> >(_femY.getTotalNumberNodes(),
                                                             std::vector<double>(rankVeffY,
                                                                                 0.0));
    FzWTimesPsizTimesShp = std::vector<std::vector<double> >(_femZ.getTotalNumberNodes(),
                                                             std::vector<double>(rankVeffZ,
                                                                                 0.0));

    std::vector<std::vector<std::vector<double> > > umatTimesPsixTimesShp(_femX.getNumberNodesPerElement(),
                                                                          std::vector<std::vector<double> >(rankVeffX,
                                                                                                            _femX.getWeightQuadPointValues()));
    std::vector<std::vector<std::vector<double> > > vmatTimesPsiyTimesShp(_femY.getNumberNodesPerElement(),
                                                                          std::vector<std::vector<double> >(rankVeffY,
                                                                                                            _femY.getWeightQuadPointValues()));
    std::vector<std::vector<std::vector<double> > > wmatTimesPsizTimesShp(_femZ.getNumberNodesPerElement(),
                                                                          std::vector<std::vector<double> >(rankVeffZ,
                                                                                                            _femZ.getWeightQuadPointValues()));

    // This part is computing the the following quadrature values for each node and rank for a Tucker decomposed potential
    // umatTimesPsixTimesShp = weightQuadPointValuesX .* jacobianQuadPointValuesX .* psiQuadValuesX .* umat[irank] .* shapeFunctionX(repeatedly on each element)

    computeMatTimesPsiTimesShp(_femX,
                               rankVeffX,
                               psiQuadValuesX,
                               umat,
                               umatTimesPsixTimesShp);
    computeMatTimesPsiTimesShp(_femY,
                               rankVeffY,
                               psiQuadValuesY,
                               vmat,
                               vmatTimesPsiyTimesShp);
    computeMatTimesPsiTimesShp(_femZ,
                               rankVeffZ,
                               psiQuadValuesZ,
                               wmat,
                               wmatTimesPsizTimesShp);


    // summing up values at each quadrature points
    sumValuesAtQuadPointsForGlobalForceVector(_femX,
                                              rankVeffX,
                                              shpTimesPsiX,
                                              shpderXTimesPsiXder,
                                              umatTimesPsixTimesShp,
                                              FxShapeDerTimesPsiXDer,
                                              FxShapeTimesPsiX,
                                              FxUTimesPsixTimesShp);
    sumValuesAtQuadPointsForGlobalForceVector(_femY,
                                              rankVeffY,
                                              shpTimesPsiY,
                                              shpderYTimesPsiYder,
                                              vmatTimesPsiyTimesShp,
                                              FyShapeDerTimesPsiYDer,
                                              FyShapeTimesPsiY,
                                              FyVTimesPsiyTimesShp);
    sumValuesAtQuadPointsForGlobalForceVector(_femZ,
                                              rankVeffZ,
                                              shpTimesPsiZ,
                                              shpderZTimesPsiZder,
                                              wmatTimesPsizTimesShp,
                                              FzShapeDerTimesPsiZDer,
                                              FzShapeTimesPsiZ,
                                              FzWTimesPsizTimesShp);
}

void FunctionalRayleighQuotientSeperable::computeOverlapIntegral(const std::vector<double> &vx,
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
                                                                 std::vector<std::vector<double> > &crossTermsXZVeffIntegral) {

    vxShpOverlap = std::vector<std::vector<std::vector<double> > >(_femX.getNumberElements(),
                                                                   std::vector<std::vector<double> >(
                                                                           _femX.getNumberNodesPerElement(),
                                                                           std::vector<double>(
                                                                                   _femX.getNumberNodesPerElement(),
                                                                                   0.0)));
    vyShpOverlap = std::vector<std::vector<std::vector<double> > >(_femY.getNumberElements(),
                                                                   std::vector<std::vector<double> >(
                                                                           _femY.getNumberNodesPerElement(),
                                                                           std::vector<double>(
                                                                                   _femY.getNumberNodesPerElement(),
                                                                                   0.0)));
    vzShpOverlap = std::vector<std::vector<std::vector<double> > >(_femZ.getNumberElements(),
                                                                   std::vector<std::vector<double> >(
                                                                           _femZ.getNumberNodesPerElement(),
                                                                           std::vector<double>(
                                                                                   _femZ.getNumberNodesPerElement(),
                                                                                   0.0)));
    crossTermsXYVeffIntegral = std::vector<std::vector<double> >(_femX.getTotalNumberNodes(),
                                                                 std::vector<double>(_femX.getTotalNumberNodes(),
                                                                                     0.0));
    crossTermsYZVeffIntegral = std::vector<std::vector<double> >(_femY.getTotalNumberNodes(),
                                                                 std::vector<double>(_femY.getTotalNumberNodes(),
                                                                                     0.0));
    crossTermsXZVeffIntegral = std::vector<std::vector<double> >(_femZ.getTotalNumberNodes(),
                                                                 std::vector<double>(_femZ.getTotalNumberNodes(),
                                                                                     0.0));

    computevShpOverlap(_femX,
                       vx,
                       vxShpOverlap);
    computevShpOverlap(_femY,
                       vy,
                       vyShpOverlap);
    computevShpOverlap(_femZ,
                       vz,
                       vzShpOverlap);

    Tucker::Tensor *core = Tucker::MemoryManager::safe_new<Tucker::Tensor>(_tuckerDecomposedVeffMPI.G->getGlobalSize());
    TensorUtils::allgatherTensor(_tuckerDecomposedVeffMPI.G,
                                 core);

    const std::vector<double> sigma(core->data(),
                                    core->data() + core->getNumElements());
    int rankVeffX = _tuckerDecomposedVeffMPI.U[0]->ncols();
    int rankVeffY = _tuckerDecomposedVeffMPI.U[1]->ncols();
    int rankVeffZ = _tuckerDecomposedVeffMPI.U[2]->ncols();

    Tucker::MemoryManager::safe_delete(core);

    int rankMatABOrder[3];
    rankMatABOrder[0] = 2;
    rankMatABOrder[1] = 0;
    rankMatABOrder[2] = 1;
    computecrossTermsABVeffIntegral(_femX,
                                    _femY,
                                    rankVeffX,
                                    rankVeffY,
                                    rankVeffZ,
                                    rankMatABOrder,
                                    sigma,
                                    FxTucker,
                                    FyTucker,
                                    wmatpsiSquare,
                                    crossTermsXYVeffIntegral);
    rankMatABOrder[0] = 0;
    rankMatABOrder[1] = 1;
    rankMatABOrder[2] = 2;
    computecrossTermsABVeffIntegral(_femY,
                                    _femZ,
                                    rankVeffX,
                                    rankVeffY,
                                    rankVeffZ,
                                    rankMatABOrder,
                                    sigma,
                                    FyTucker,
                                    FzTucker,
                                    umatpsiSquare,
                                    crossTermsYZVeffIntegral);
    rankMatABOrder[0] = 1;
    rankMatABOrder[1] = 0;
    rankMatABOrder[2] = 2;
    computecrossTermsABVeffIntegral(_femX,
                                    _femZ,
                                    rankVeffX,
                                    rankVeffY,
                                    rankVeffZ,
                                    rankMatABOrder,
                                    sigma,
                                    FxTucker,
                                    FzTucker,
                                    vmatpsiSquare,
                                    crossTermsXZVeffIntegral);
}

void FunctionalRayleighQuotientSeperable::computeVectorizedForce(const std::vector<double> &nodalFieldsX,
                                                                 const std::vector<double> &nodalFieldsY,
                                                                 const std::vector<double> &nodalFieldsZ,
                                                                 double lagrangeMultiplier,
                                                                 std::vector<double> &F) {
    std::vector<double> psiQuadValuesX, DPsiQuadValuesX;
    _femX.computeFieldAndDiffFieldAtAllQuadPoints(nodalFieldsX,
                                                  psiQuadValuesX,
                                                  DPsiQuadValuesX);
    std::vector<double> psiQuadValuesY, DPsiQuadValuesY;
    _femY.computeFieldAndDiffFieldAtAllQuadPoints(nodalFieldsY,
                                                  psiQuadValuesY,
                                                  DPsiQuadValuesY);
    std::vector<double> psiQuadValuesZ, DPsiQuadValuesZ;
    _femZ.computeFieldAndDiffFieldAtAllQuadPoints(nodalFieldsZ,
                                                  psiQuadValuesZ,
                                                  DPsiQuadValuesZ);

    double normPsiX, normPsiY, normPsiZ, normDPsiX, normDPsiY, normDPsiZ;
    computeIntegralPsiSquare(psiQuadValuesX,
                             psiQuadValuesY,
                             psiQuadValuesZ,
                             DPsiQuadValuesX,
                             DPsiQuadValuesY,
                             DPsiQuadValuesZ,
                             normPsiX,
                             normPsiY,
                             normPsiZ,
                             normDPsiX,
                             normDPsiY,
                             normDPsiZ);

    std::vector<double> vx, vy, vz;
    std::vector<double> umatpsiSquare, vmatpsiSquare, wmatpsiSquare;
    computeSeparablePotentialUsingTuckerVeff(psiQuadValuesX,
                                             psiQuadValuesY,
                                             psiQuadValuesZ,
                                             vx,
                                             vy,
                                             vz,
                                             umatpsiSquare,
                                             vmatpsiSquare,
                                             wmatpsiSquare);

    double cx = normPsiY * normPsiZ;
    double cy = normPsiX * normPsiZ;
    double cz = normPsiX * normPsiY;
    double Tx = 0.5 * (normDPsiY * normPsiZ + normDPsiZ * normPsiY);
    double Ty = 0.5 * (normDPsiX * normPsiZ + normDPsiZ * normPsiX);
    double Tz = 0.5 * (normDPsiX * normPsiY + normDPsiY * normPsiX);

    const int numberTotalQuadPointsX = _femX.getTotalNumberQuadPoints();
    const int numberTotalQuadPointsY = _femY.getTotalNumberQuadPoints();
    const int numberTotalQuadPointsZ = _femZ.getTotalNumberQuadPoints();
    const int numberTotalNodesX = _femX.getTotalNumberNodes();
    const int numberTotalNodesY = _femY.getTotalNumberNodes();
    const int numberTotalNodesZ = _femZ.getTotalNumberNodes();
    const int numberNodesPerEleX = _femX.getNumberNodesPerElement();
    const int numberNodesPerEleY = _femY.getNumberNodesPerElement();
    const int numberNodesPerEleZ = _femZ.getNumberNodesPerElement();

    std::vector<std::vector<double> > X0(numberNodesPerEleX,
                                         std::vector<double>(numberTotalQuadPointsX,
                                                             0.0));
    std::vector<std::vector<double> > Y0(numberNodesPerEleY,
                                         std::vector<double>(numberTotalQuadPointsY,
                                                             0.0));
    std::vector<std::vector<double> > Z0(numberNodesPerEleZ,
                                         std::vector<double>(numberTotalQuadPointsZ,
                                                             0.0));

    computeOneDimForce(_femX,
                       psiQuadValuesX,
                       DPsiQuadValuesX,
                       cx,
                       Tx,
                       vx,
                       lagrangeMultiplier,
                       X0);
    computeOneDimForce(_femY,
                       psiQuadValuesY,
                       DPsiQuadValuesY,
                       cy,
                       Ty,
                       vy,
                       lagrangeMultiplier,
                       Y0);
    computeOneDimForce(_femZ,
                       psiQuadValuesZ,
                       DPsiQuadValuesZ,
                       cz,
                       Tz,
                       vz,
                       lagrangeMultiplier,
                       Z0);

    std::vector<double> Fx(numberTotalNodesX,
                           0.0);
    std::vector<double> Fy(numberTotalNodesY,
                           0.0);
    std::vector<double> Fz(numberTotalNodesZ,
                           0.0);

    // Force assembly
    sumOneDimForceOverQuadPoints(_femX,
                                 X0,
                                 Fx);
    sumOneDimForceOverQuadPoints(_femY,
                                 Y0,
                                 Fy);
    sumOneDimForceOverQuadPoints(_femZ,
                                 Z0,
                                 Fz);

    F = std::vector<double>(numberTotalNodesX + numberTotalNodesY + numberTotalNodesZ - 5,
                            0.0);
    std::copy(Fx.begin() + 1,
              Fx.end() - 1,
              F.begin());
    std::copy(Fy.begin() + 1,
              Fy.end() - 1,
              F.begin() + numberTotalNodesX - 2);
    std::copy(Fz.begin() + 1,
              Fz.end() - 1,
              F.begin() + numberTotalNodesX + numberTotalNodesY - 4);
    // insert constraint force
    F.back() = 0.5 * (normPsiX * normPsiY * normPsiZ - 1.0);
}

void
FunctionalRayleighQuotientSeperable::constructGHEP(const FEM &fem,
                                                   const std::vector<double> &v,
                                                   const double potConst,
                                                   std::vector<double> &H,
                                                   std::vector<double> &M) {
    const std::vector<std::vector<std::vector<double> > > &elementalMass = fem.getShapeFunctionOverlapIntegral();
    const std::vector<std::vector<std::vector<double> > > &elementalKinetic = fem.getShapeFunctionGradientIntegral();
    std::vector<std::vector<std::vector<double> > > elementalPot;
    // compute elemental integration of vxNINj
    fem.computeQuadExternalFunctionShapeFunctionOverlapIntegral(v,
                                                                elementalPot);

    int numberNodesPerElement = fem.getNumberNodesPerElement();
    int numberElements = fem.getNumberElements();
    auto &elementConnectivity = fem.getElementConnectivity();
    int m = fem.getTotalNumberNodes() - 2, n = fem.getTotalNumberNodes() - 2;
    int lengthMat = m * n;
    H = std::vector<double>(lengthMat,
                            0.0);
    M = std::vector<double>(lengthMat,
                            0.0);

    // Set values on the boundary
    for (int jNode = 1; jNode < numberNodesPerElement; ++jNode) {
        for (int iNode = 1; iNode < numberNodesPerElement; ++iNode) {
            int globalI = elementConnectivity.front()[iNode] - 1, globalJ = elementConnectivity.front()[jNode] - 1;
            int flattenIdx = globalI + globalJ * m;
            if (iNode >= jNode) {
                H[flattenIdx] +=
                        0.5 * elementalKinetic.front()[iNode][jNode] + potConst * elementalPot.front()[iNode][jNode];
                M[flattenIdx] += elementalMass.front()[iNode][jNode];
            } else {
                H[flattenIdx] += H[globalJ + globalI * m];
                M[flattenIdx] += M[globalJ + globalI * m];
            }
        }
    }
    for (int ele = 1; ele < numberElements - 1; ++ele) {
        for (int iNode = 0; iNode < numberNodesPerElement; ++iNode) {
            for (int jNode = 0; jNode < numberNodesPerElement; ++jNode) {
                int globalI = elementConnectivity[ele][iNode] - 1, globalJ = elementConnectivity[ele][jNode] - 1;
                int flattenIdx = globalI + globalJ * m;
                if (iNode >= jNode) {
                    H[flattenIdx] += 0.5 * elementalKinetic[ele][iNode][jNode] +
                                     potConst * elementalPot[ele][iNode][jNode];
                    M[flattenIdx] += elementalMass[ele][iNode][jNode];
                } else {
                    H[flattenIdx] += H[globalJ + globalI * m];
                    M[flattenIdx] += M[globalJ + globalI * m];
                }
            }
        }
    }
    for (int iNode = 0; iNode < numberNodesPerElement - 1; ++iNode) {
        for (int jNode = 0; jNode < numberNodesPerElement - 1; ++jNode) {
            int globalI = elementConnectivity.back()[iNode] - 1, globalJ = elementConnectivity.back()[jNode] - 1;
            int flattenIdx = globalI + globalJ * m;
            if (iNode >= jNode) {
                H[flattenIdx] +=
                        0.5 * elementalKinetic.back()[iNode][jNode] + potConst * elementalPot.back()[iNode][jNode];
                M[flattenIdx] += elementalMass.back()[iNode][jNode];
            } else {
                H[flattenIdx] += H[globalJ + globalI * m];
                M[flattenIdx] += M[globalJ + globalI * m];
            }
        }
    }
    std::for_each(H.begin(),
                  H.end(),
                  [](double &d) { std::cout << d << " "; });
    std::cout << std::endl;
    std::for_each(M.begin(),
                  M.end(),
                  [](double &d) { std::cout << d << " "; });
    std::cout << std::endl;
}

void
FunctionalRayleighQuotientSeperable::solveGHEP(std::vector<double> &A,
                                               std::vector<double> &B,
                                               int size,
                                               int numeigs,
                                               std::vector<std::vector<double> > &solution) {
    int itype = 1;
    char jobz = 'V';
    char range = 'I';
    char uplo = 'U';
    int n = size;
    int il = 1;
    int iu = numeigs;
    double vl = 0.0, vu = 0.0;
    char cmach = 'S';
    double abstol = 1e-3;//dlamch_(&cmach)*2.;
    int m = iu - il + 1;
    int ldz = n;
    int lda = n;
    int ldb = n;
    int lwork = 8 * n;
    std::vector<double> w(m,
                          0.0);
    std::vector<double> z(m * n,
                          0.0);
    std::vector<double> work(8 * n,
                             0.0);
    std::vector<int> iwork(5 * n,
                           0);
    std::vector<int> ifail(n,
                           0);
    int info = 0;
    std::cout << "before eigensolver";

    clinalg::dsygvx_(1,
                     'V',
                     'I',
                     'U',
                     size,
                     A.data(),
                     size,
                     B.data(),
                     size,
                     0.0,
                     0.0,
                     1,
                     numeigs,
                     abstol,
                     &m,
                     w.data(),
                     z.data(),
                     size,
                     work.data(),
                     &lwork,
                     iwork.data(),
                     ifail.data(),
                     &info);
    std::cout << std::endl << "n: " << n << "   info: " << info << std::endl;
}

void
FunctionalRayleighQuotientSeperable::constructGHEP(const FEM &fem,
                                                   const std::vector<double> &v,
                                                   const double potConst,
                                                   Mat &H,
                                                   Mat &M) {
    const std::vector<std::vector<std::vector<double> > > &elementalMass = fem.getShapeFunctionOverlapIntegral();
    const std::vector<std::vector<std::vector<double> > > &elementalKinetic = fem.getShapeFunctionGradientIntegral();
    std::vector<std::vector<std::vector<double> > > elementalPot;
    // compute elemental integration of vxNINj

    fem.computeQuadExternalFunctionShapeFunctionOverlapIntegral(v,
                                                                elementalPot);

    int numberNodesPerElement = fem.getNumberNodesPerElement();
    std::vector<std::vector<PetscInt>> elementConnectivity(fem.getElementConnectivity().size());
    // Remove the boundary, thus minus every index by one
    for (int i = 0; i < elementConnectivity.size(); ++i) {
        const auto &conn = fem.getElementConnectivity()[i];
        elementConnectivity[i].resize(conn.size());
        for (int j = 0; j < conn.size(); ++j) {
            elementConnectivity[i][j] = conn[j] - 1;
        }
    }
    // Set the index of the other boundary to be negative for PETSc to neglect it
    elementConnectivity.back().back() *= (-1);

    // Clean up the matrices
    MatZeroEntries(H);
    MatZeroEntries(M);
    // Allocate the vector

    std::vector<PetscScalar> elementalHxFlatten(numberNodesPerElement * numberNodesPerElement);
    std::vector<PetscScalar> elementalMxFlatten(numberNodesPerElement * numberNodesPerElement);
    for (auto ele = 0; ele < fem.getNumberElements(); ++ele) {
        for (auto iNode = 0; iNode < numberNodesPerElement; ++iNode) {
            for (auto jNode = 0; jNode < numberNodesPerElement; ++jNode) {
                size_t flattenOffset = jNode + iNode * numberNodesPerElement;
                elementalHxFlatten[flattenOffset] =
                        0.5 * elementalKinetic[ele][iNode][jNode] + potConst * elementalPot[ele][iNode][jNode];
                elementalMxFlatten[flattenOffset] = elementalMass[ele][iNode][jNode];
            }
        }
        MatSetValues(H,
                     numberNodesPerElement,
                     &elementConnectivity[ele][0],
                     numberNodesPerElement,
                     &elementConnectivity[ele][0],
                     &elementalHxFlatten[0],
                     ADD_VALUES);
        MatSetValues(M,
                     numberNodesPerElement,
                     &elementConnectivity[ele][0],
                     numberNodesPerElement,
                     &elementConnectivity[ele][0],
                     &elementalMxFlatten[0],
                     ADD_VALUES);
    }
    MatAssemblyBegin(H,
                     MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(H,
                   MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(M,
                     MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M,
                   MAT_FINAL_ASSEMBLY);
}

void FunctionalRayleighQuotientSeperable::solveGHEP(Mat &A,
                                                    Mat &B,
                                                    int numeigs,
                                                    std::vector<std::vector<double> > &solution) {

    /* construct the eigenproblem solve*/
    EPS eps;
    EPSCreate(PETSC_COMM_SELF,
              &eps);
    EPSSetOperators(eps,
                    A,
                    B);
    /* set up the number of eigen vectors/values to be solved*/
    EPSSetDimensions(eps,
                     numeigs,
                     PETSC_DEFAULT,
                     PETSC_DEFAULT);
    /* solve from the smallest value up to numeig-th eigen vectors/values*/
    EPSSetWhichEigenpairs(eps,
                          EPS_SMALLEST_REAL);
    EPSSetType(eps,
               EPSLAPACK);
    EPSSetProblemType(eps,
                      EPS_GHEP);
    EPSSetFromOptions(eps);
    EPSSolve(eps);

    /* retrieve and copy the solutions */
    Vec vec;
    MatCreateVecs(A,
                  NULL,
                  &vec);
    PetscInt vecLength;
    MatGetSize(A,
               &vecLength,
               NULL);
    solution = std::vector<std::vector<double> >(numeigs,
                                                 std::vector<double>(vecLength + 2,
                                                                     0.0));
    for (int i = 0; i < numeigs; ++i) {
        EPSGetEigenvector(eps,
                          i,
                          vec,
                          PETSC_NULL);
        PetscScalar *vv;
        VecGetArray(vec,
                    &vv);
        std::copy(&vv[0],
                  &vv[vecLength],
                  solution[i].begin() + 1);
        VecRestoreArray(vec,
                        &vv);
    }
    /* release memory */
    EPSDestroy(&eps);
    VecDestroy(&vec);
}

// TODO rename the function
void FunctionalRayleighQuotientSeperable::solveGHEPLapack(Mat &A,
                                                          Mat &B,
                                                          int numeigs,
                                                          std::vector<std::vector<double> > &solution) {

    PetscInt sizeM, sizeN;
    MatGetSize(A,
               &sizeM,
               &sizeN);
    std::vector<double> matA;
    std::vector<double> matB;
    std::vector<double> temp(sizeM * sizeN,
                             0.0);
    solution = std::vector<std::vector<double> >(numeigs,
                                                 std::vector<double>(sizeM + 2,
                                                                     0.0));

    std::vector<PetscInt> idx(sizeM);
    for (int i = 0; i < sizeM; ++i) {
        idx[i] = i;
    }

    MatGetValues(A,
                 sizeM,
                 idx.data(),
                 sizeN,
                 idx.data(),
                 temp.data());

    int cnt = 0;
    for (int j = 0; j < sizeN; ++j) {
        for (int i = 0; i < sizeM; ++i) {
            matA.push_back(temp[j + i * sizeN]);
        }
    }

    MatGetValues(B,
                 sizeM,
                 idx.data(),
                 sizeN,
                 idx.data(),
                 temp.data());

    for (int j = 0; j < sizeN; ++j) {
        for (int i = 0; i < sizeM; ++i) {
            matB.push_back(temp[j + i * sizeN]);
        }
    }

    int itype = 1;
    char jobz = 'V';
    char range = 'I';
    char uplo = 'U';
    int n = sizeM;
    int il = 1;
    int iu = numeigs;
    double vl = 0.0, vu = 100.0;
    double abstol = clinalg::dlamch_('S') * 2.;//1.0e-8
    int m = iu - il + 1;
    int ldz = n;
    int lda = n;
    int ldb = n;
    int lwork = 20 * n;
    std::vector<double> w(n,
                          0.0);
    std::vector<double> z(n * m,
                          0.0);
    std::vector<double> work(lwork,
                             0.0);
    std::vector<int> iwork(5 * n,
                           0);
    std::vector<int> ifail(n,
                           0);
    int info = 0;

    clinalg::dsygvx_(1,
                     'V',
                     'I',
                     'U',
                     n,
                     matA.data(),
                     n,
                     matB.data(),
                     n,
                     0.0,
                     100.0,
                     1,
                     numeigs,
                     abstol,
                     &m,
                     w.data(),
                     z.data(),
                     n,
                     work.data(),
                     &lwork,
                     iwork.data(),
                     ifail.data(),
                     &info);

    int taskId, rootId = 0;
    std::vector<double> evals(numeigs,
                              0.0);
    if (numeigs > 1) {
        for (int i = 0; i < m; ++i) {
            printf("e-val %d: %e\n",
                   i,
                   w[i]);
        }
    }
    for (int i = 0; i < m; ++i) {
        std::copy(z.begin() + i * n,
                  z.begin() + (i + 1) * n,
                  solution[i].begin() + 1);

    }

}

void FunctionalRayleighQuotientSeperable::generateHamiltonianGenericPotential(const std::vector<double> &nodalFieldsX,
                                                                              const std::vector<double> &nodalFieldsY,
                                                                              const std::vector<double> &nodalFieldsZ,
                                                                              Mat &Hx,
                                                                              Mat &Hy,
                                                                              Mat &Hz,
                                                                              Mat &Mx,
                                                                              Mat &My,
                                                                              Mat &Mz) {
    std::vector<double> psiQuadValuesX, DPsiQuadValuesX, psiQuadValuesY, DPsiQuadValuesY, psiQuadValuesZ, DPsiQuadValuesZ;
    _femX.computeFieldAndDiffFieldAtAllQuadPoints(nodalFieldsX,
                                                  psiQuadValuesX,
                                                  DPsiQuadValuesX);
    _femY.computeFieldAndDiffFieldAtAllQuadPoints(nodalFieldsY,
                                                  psiQuadValuesY,
                                                  DPsiQuadValuesY);
    _femZ.computeFieldAndDiffFieldAtAllQuadPoints(nodalFieldsZ,
                                                  psiQuadValuesZ,
                                                  DPsiQuadValuesZ);

    double normPsiX, normPsiY, normPsiZ, normDPsiX, normDPsiY, normDPsiZ;
    computeIntegralPsiSquare(psiQuadValuesX,
                             psiQuadValuesY,
                             psiQuadValuesZ,
                             DPsiQuadValuesX,
                             DPsiQuadValuesY,
                             DPsiQuadValuesZ,
                             normPsiX,
                             normPsiY,
                             normPsiZ,
                             normDPsiX,
                             normDPsiY,
                             normDPsiZ);

    std::vector<double> vx, vy, vz;
    std::vector<double> umatpsiSquare, vmatpsiSquare, wmatpsiSquare;
    computeSeparablePotentialUsingTuckerVeff(psiQuadValuesX,
                                             psiQuadValuesY,
                                             psiQuadValuesZ,
                                             vx,
                                             vy,
                                             vz,
                                             umatpsiSquare,
                                             vmatpsiSquare,
                                             wmatpsiSquare);

    double cx = 1.0 / (normPsiY * normPsiZ), cy = 1.0 / (normPsiX * normPsiZ), cz = 1.0 / (normPsiX * normPsiY);
    std::vector<double> H, M;

    constructGHEP(_femX,
                  vx,
                  cx,
                  Hx,
                  Mx);
    constructGHEP(_femY,
                  vy,
                  cy,
                  Hy,
                  My);
    constructGHEP(_femZ,
                  vz,
                  cz,
                  Hz,
                  Mz);
}

void FunctionalRayleighQuotientSeperable::solveHamiltonianGenericPotential(Mat &Hx,
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
                                                                           std::vector<std::vector<double> > &eigZ) {
    solveGHEPLapack(Hx,
                    Mx,
                    rankTuckerBasisX,
                    eigX);
    solveGHEPLapack(Hy,
                    My,
                    rankTuckerBasisY,
                    eigY);
    solveGHEPLapack(Hz,
                    Mz,
                    rankTuckerBasisZ,
                    eigZ);
}

const FEM &FunctionalRayleighQuotientSeperable::get_femX() const {
    return _femX;
}

const FEM &FunctionalRayleighQuotientSeperable::get_femY() const {
    return _femY;
}

const FEM &FunctionalRayleighQuotientSeperable::get_femZ() const {
    return _femZ;
}

void FunctionalRayleighQuotientSeperable::computeIntegralMatPsiSquare(const int rank,
                                                                      const FEM &fem,
                                                                      const std::vector<double> &mat,
                                                                      const std::vector<double> &psiQuadValues,
                                                                      std::vector<double> &matpsiSquare) {
    int numberTotalNumberQuadPoints = fem.getTotalNumberQuadPoints();
    auto u_tims_vsquare = [](const double &u,
                             const double &v) -> double { return u * v * v; };

    for (int p = 0; p != rank; ++p) {
        int offset = p * numberTotalNumberQuadPoints;
        std::vector<double> temp(numberTotalNumberQuadPoints);
        std::transform(mat.begin() + offset,
                       mat.begin() + offset + numberTotalNumberQuadPoints,
                       psiQuadValues.begin(),
                       temp.begin(),
                       u_tims_vsquare);
        matpsiSquare[p] = fem.integrate_by_quad_values(temp);
    }
}

/**
 *
 * @param rankX rank in X direction
 * @param rankY rank in Y direction
 * @param rankZ rank in Z direction
 * @param rankABCOrder specify the the order of your input, e.g. matv, umatpsiSquare, matpsiSquarew require rankABCOrder = [1, 0, 2]
 * @param core core tensor
 * @param matA vmat, for example
 * @param matpsiSquareB umatpsiSquareB, for example
 * @param matpsiSquareC wmatpsiSquareB, for example
 * @param vA vy, for example
 */
void FunctionalRayleighQuotientSeperable::computeProductCoreMatIntMatPsiSquare(const int rankX,
                                                                               const int rankY,
                                                                               const int rankZ,
                                                                               const int *rankABCOrder,
                                                                               const std::vector<double> &core,
                                                                               const std::vector<double> &matA,
                                                                               const std::vector<double> &matpsiSquareB,
                                                                               const std::vector<double> &matpsiSquareC,
                                                                               std::vector<double> &vA) {
    // vx.size() is equal to the total number of quadrature points
    int offsetMat = vA.size();
    int rank[3] = {rankX, rankY, rankZ};
    int coreOffsetMultiplier[3];
    for (int i = 0; i < 3; ++i) {
        switch (rankABCOrder[i]) {
            case 0:
                coreOffsetMultiplier[i] = 1;
                break;
            case 1:
                coreOffsetMultiplier[i] = rank[0];
                break;
            case 2:
                coreOffsetMultiplier[i] = rank[0] * rank[1];
                break;
        }
    }

    for (int idx = 0; idx != vA.size(); ++idx) {
        for (int i = 0; i != rank[rankABCOrder[0]]; ++i) {
            for (int j = 0; j != rank[rankABCOrder[1]]; ++j) {
                for (int k = 0; k != rank[rankABCOrder[2]]; ++k) {
                    int coreOffset = i * coreOffsetMultiplier[0] + j * coreOffsetMultiplier[1] +
                                     k * coreOffsetMultiplier[2];
                    vA[idx] += core[coreOffset] * matA[offsetMat * i + idx] *
                               matpsiSquareB[j] * matpsiSquareC[k];
                }
            }
        }
    }
}

void FunctionalRayleighQuotientSeperable::computeShapeFuncTimesPsiandPsider(const FEM &fem,
                                                                            const std::vector<double> &psiQuadValues,
                                                                            const std::vector<double> &DPsiQuadValues,
                                                                            std::vector<std::vector<double> > &shapeFuncTimesPsider,
                                                                            std::vector<std::vector<double> > &shapeFuncTimesPsi) {

    const std::vector<double> &jacobQuadPointValues = fem.getJacobQuadPointValues();
    const std::vector<double> &invJacobQuadPointValues = fem.getInvJacobQuadPointValues();
    const std::vector<std::vector<double> > &shapeFunction = fem.getShapeFunctionAtQuadPoints();
    const std::vector<std::vector<double> > &DShapeFunction = fem.getShapeFunctionDerivativeAtQuadPoints();
    const int numberQuadPointsPerElements = fem.getNumberQuadPointsPerElement();
    const int numberElements = fem.getNumberElements();

    for (int iNode = 0; iNode != fem.getNumberNodesPerElement(); ++iNode) {
        std::transform(shapeFuncTimesPsider[iNode].begin(),
                       shapeFuncTimesPsider[iNode].end(),
                       invJacobQuadPointValues.begin(),
                       shapeFuncTimesPsider[iNode].begin(),
                       std::multiplies<double>());
        std::transform(shapeFuncTimesPsider[iNode].begin(),
                       shapeFuncTimesPsider[iNode].end(),
                       DPsiQuadValues.begin(),
                       shapeFuncTimesPsider[iNode].begin(),
                       std::multiplies<double>());
        for (int ele = 0; ele != numberElements; ++ele) {
            int offset = ele * numberQuadPointsPerElements;
            std::transform(DShapeFunction[iNode].begin(),
                           DShapeFunction[iNode].end(),
                           shapeFuncTimesPsider[iNode].begin() + offset,
                           shapeFuncTimesPsider[iNode].begin() + offset,
                           std::multiplies<double>());
        }

        std::transform(shapeFuncTimesPsi[iNode].begin(),
                       shapeFuncTimesPsi[iNode].end(),
                       jacobQuadPointValues.begin(),
                       shapeFuncTimesPsi[iNode].begin(),
                       std::multiplies<double>());
        std::transform(shapeFuncTimesPsi[iNode].begin(),
                       shapeFuncTimesPsi[iNode].end(),
                       psiQuadValues.begin(),
                       shapeFuncTimesPsi[iNode].begin(),
                       std::multiplies<double>());
        for (int ele = 0; ele != numberElements; ++ele) {
            int offset = ele * numberQuadPointsPerElements;
            std::transform(shapeFunction[iNode].begin(),
                           shapeFunction[iNode].end(),
                           shapeFuncTimesPsi[iNode].begin() + offset,
                           shapeFuncTimesPsi[iNode].begin() + offset,
                           std::multiplies<double>());
        }
    }
}

void FunctionalRayleighQuotientSeperable::computeMatTimesPsiTimesShp(const FEM &fem,
                                                                     const int rank,
                                                                     const std::vector<double> &psiQuadValues,
                                                                     const std::vector<double> &mat,
                                                                     std::vector<std::vector<std::vector<double> > > &matTimesPsiTimeShp) {
    const int numberNodesPerEle = fem.getNumberNodesPerElement();
    const int numberTotalQuadPoints = fem.getTotalNumberQuadPoints();
    const int numberElements = fem.getNumberElements();
    const int numberQuadPointsPerEle = fem.getNumberQuadPointsPerElement();
    const std::vector<double> &jacobQuadPointValues = fem.getJacobQuadPointValues();
    const std::vector<std::vector<double> > &shapeFunction = fem.getShapeFunctionAtQuadPoints();
    for (int iNode = 0; iNode != numberNodesPerEle; ++iNode) {
        for (int irank = 0; irank != rank; ++irank) {
            std::transform(psiQuadValues.begin(),
                           psiQuadValues.end(),
                           matTimesPsiTimeShp[iNode][irank].begin(),
                           matTimesPsiTimeShp[iNode][irank].begin(),
                           std::multiplies<double>());
            std::transform(jacobQuadPointValues.begin(),
                           jacobQuadPointValues.end(),
                           matTimesPsiTimeShp[iNode][irank].begin(),
                           matTimesPsiTimeShp[iNode][irank].begin(),
                           std::multiplies<double>());
            std::transform(mat.begin() + irank * numberTotalQuadPoints,
                           mat.begin() + (irank + 1) * numberTotalQuadPoints,
                           matTimesPsiTimeShp[iNode][irank].begin(),
                           matTimesPsiTimeShp[iNode][irank].begin(),
                           std::multiplies<double>());
            for (int ele = 0; ele != numberElements; ++ele) {
                int offset = ele * numberQuadPointsPerEle;
                std::transform(shapeFunction[iNode].begin(),
                               shapeFunction[iNode].end(),
                               matTimesPsiTimeShp[iNode][irank].begin() + offset,
                               matTimesPsiTimeShp[iNode][irank].begin() + offset,
                               std::multiplies<double>());
            }
        }
    }
}

void FunctionalRayleighQuotientSeperable::sumValuesAtQuadPointsForGlobalForceVector(const FEM &fem,
                                                                                    const int rank,
                                                                                    const std::vector<std::vector<double> > &shpTimesPsi,
                                                                                    const std::vector<std::vector<double> > &shpderTimesPsider,
                                                                                    const std::vector<std::vector<std::vector<
                                                                                            double> > > &matTimesPsiTimesShp,
                                                                                    std::vector<double> &FunShapeDerTimesPsiDer,
                                                                                    std::vector<double> &FuncShapeTimesPsi,
                                                                                    std::vector<std::vector<double> > &FuncMatTimesPsiTimesShape) {
    const int numberElements = fem.getNumberElements();
    const int numberNodesPerElement = fem.getNumberNodesPerElement();
    const int numberQuadPointsPerElement = fem.getNumberQuadPointsPerElement();
    const std::vector<std::vector<int> > &elementConnectivity = fem.getElementConnectivity();
    for (int ele = 0; ele != numberElements; ++ele) {
        for (int iNode = 0; iNode != numberNodesPerElement; ++iNode) {
            int iGlobal = elementConnectivity[ele][iNode];
            FunShapeDerTimesPsiDer[iGlobal] += std::accumulate(
                    shpderTimesPsider[iNode].begin() + ele * numberQuadPointsPerElement,
                    shpderTimesPsider[iNode].begin() + (ele + 1) * numberQuadPointsPerElement,
                    0.0);
            FuncShapeTimesPsi[iGlobal] += std::accumulate(shpTimesPsi[iNode].begin() + ele * numberQuadPointsPerElement,
                                                          shpTimesPsi[iNode].begin() +
                                                          (ele + 1) * numberQuadPointsPerElement,
                                                          0.0);
            for (int irank = 0; irank != rank; ++irank) {
                FuncMatTimesPsiTimesShape[iGlobal][irank] += std::accumulate(
                        matTimesPsiTimesShp[iNode][irank].begin() + ele * numberQuadPointsPerElement,
                        matTimesPsiTimesShp[iNode][irank].begin() + (ele + 1) * numberQuadPointsPerElement,
                        0.0);
            }
        }
    }
}

void FunctionalRayleighQuotientSeperable::computevShpOverlap(const FEM &fem,
                                                             const std::vector<double> &v,
                                                             std::vector<std::vector<std::vector<double> > > &vShpOverlap) {
    int numberElements = fem.getNumberElements();
    int numberNodesPerElement = fem.getNumberNodesPerElement();
    int numberQuadPointsPerElement = fem.getNumberQuadPointsPerElement();
    const std::vector<double> &weightQuadPointValues = fem.getWeightQuadPointValues();
    const std::vector<std::vector<double> > &shapeFunction = fem.getShapeFunctionAtQuadPoints();
    const std::vector<double> &jacobianQuadPointValues = fem.getJacobQuadPointValues();
    for (int ele = 0; ele != numberElements; ++ele) {
        for (int i = 0; i != numberNodesPerElement; ++i) {
            for (int j = 0; j != numberNodesPerElement; ++j) {
                for (int iquad = 0; iquad != numberQuadPointsPerElement; ++iquad) {
                    vShpOverlap[ele][i][j] += weightQuadPointValues[numberQuadPointsPerElement * ele + iquad] *
                                              v[numberQuadPointsPerElement * ele + iquad] * shapeFunction[i][iquad] *
                                              shapeFunction[j][iquad] *
                                              jacobianQuadPointValues[numberQuadPointsPerElement * ele + iquad];
                }
            }
        }
    }
}

void FunctionalRayleighQuotientSeperable::computecrossTermsABVeffIntegral(const FEM &femA,
                                                                          const FEM &femB,
                                                                          const int rankX,
                                                                          const int rankY,
                                                                          const int rankZ,
                                                                          const int *rankMatABOrder,
                                                                          const std::vector<double> &core,
                                                                          const std::vector<std::vector<double> > &FATucker,
                                                                          const std::vector<std::vector<double> > &FBTucker,
                                                                          const std::vector<double> &matpsiSquare,
                                                                          std::vector<std::vector<double> > &crossTermsABVeffIntegral) {
    // vx.size() is equal to the total number of quadrature points
    int rank[3] = {rankX, rankY, rankZ};
    int coreOffsetMultiplier[3];
    for (int i = 0; i < 3; ++i) {
        switch (rankMatABOrder[i]) {
            case 0:
                coreOffsetMultiplier[i] = 1;
                break;
            case 1:
                coreOffsetMultiplier[i] = rank[0];
                break;
            case 2:
                coreOffsetMultiplier[i] = rank[0] * rank[1];
                break;
        }
    }

    const int numberTotalNodesA = femA.getTotalNumberNodes();
    const int numberTotalNodesB = femB.getTotalNumberNodes();
    for (int inode = 0; inode != numberTotalNodesA; ++inode) {
        for (int jnode = 0; jnode != numberTotalNodesB; ++jnode) {
            for (int krank = 0; krank != rank[rankMatABOrder[2]]; ++krank) {
                for (int jrank = 0; jrank != rank[rankMatABOrder[1]]; ++jrank) {
                    for (int irank = 0; irank != rank[rankMatABOrder[0]]; ++irank) {
                        int coreOffset = irank * coreOffsetMultiplier[0] + jrank * coreOffsetMultiplier[1] +
                                         krank * coreOffsetMultiplier[2];
                        crossTermsABVeffIntegral[inode][jnode] +=
                                core[coreOffset] *
                                FATucker[inode][irank] * FBTucker[jnode][jrank] * matpsiSquare[krank];
                    }
                }
            }
        }
    }
}

void FunctionalRayleighQuotientSeperable::computeOneDimForce(const FEM &fem,
                                                             std::vector<double> &psiQuadValues,
                                                             std::vector<double> &DPsiQuadValues,
                                                             const double normPsiProduct,
                                                             const double normPsiDpsiOuterProduct,
                                                             const std::vector<double> &v,
                                                             const double lagrangeMultiplier,
                                                             std::vector<std::vector<double> > &oneDimForce) {

    const int numberNodePerElement = fem.getNumberNodesPerElement();
    const int numberElements = fem.getNumberElements();
    const int numberQuadPointsPerElement = fem.getNumberQuadPointsPerElement();
    const std::vector<double> &jacobianQuadPointValues = fem.getJacobQuadPointValues();
    const std::vector<double> &invJacobianQuadPointValues = fem.getInvJacobQuadPointValues();
    const std::vector<double> &weightQuadPointValues = fem.getWeightQuadPointValues();
    const std::vector<std::vector<double> > &shapeFunction = fem.getShapeFunctionAtQuadPoints();
    const std::vector<std::vector<double> > &DShapeFunction = fem.getShapeFunctionDerivativeAtQuadPoints();
    for (int iNode = 0; iNode != numberNodePerElement; ++iNode) {
        std::vector<double> kinetic(DPsiQuadValues);
        std::vector<double> other(psiQuadValues);
        double alpha = 0.5 * normPsiProduct;
        clinalg::dscal_(DPsiQuadValues.size(),
                        alpha,
                        kinetic.data(),
                        1);
        alpha = lagrangeMultiplier * normPsiProduct + normPsiDpsiOuterProduct;
        std::transform(other.begin(),
                       other.end(),
                       v.begin(),
                       other.begin(),
                       std::multiplies<double>());
        clinalg::daxpy_(DPsiQuadValues.size(),
                        alpha,
                        psiQuadValues.data(),
                        1,
                        other.data(),
                        1);
        for (int ele = 0; ele != numberElements; ++ele) {
            int offset = ele * numberQuadPointsPerElement;
            std::transform(shapeFunction[iNode].begin(),
                           shapeFunction[iNode].end(),
                           other.begin() + offset,
                           other.begin() + offset,
                           std::multiplies<double>());
        }
        std::transform(jacobianQuadPointValues.begin(),
                       jacobianQuadPointValues.end(),
                       other.begin(),
                       other.begin(),
                       std::multiplies<double>());
        for (int ele = 0; ele != numberElements; ++ele) {
            int offset = ele * numberQuadPointsPerElement;
            std::transform(DShapeFunction[iNode].begin(),
                           DShapeFunction[iNode].end(),
                           kinetic.begin() + offset,
                           kinetic.begin() + offset,
                           std::multiplies<double>());
        }
        std::transform(invJacobianQuadPointValues.begin(),
                       invJacobianQuadPointValues.end(),
                       kinetic.begin(),
                       kinetic.begin(),
                       std::multiplies<double>());

        std::transform(kinetic.begin(),
                       kinetic.end(),
                       other.begin(),
                       oneDimForce[iNode].begin(),
                       std::plus<double>());
        std::transform(weightQuadPointValues.begin(),
                       weightQuadPointValues.end(),
                       oneDimForce[iNode].begin(),
                       oneDimForce[iNode].begin(),
                       std::multiplies<double>());
    }
}

void FunctionalRayleighQuotientSeperable::sumOneDimForceOverQuadPoints(const FEM &fem,
                                                                       const std::vector<std::vector<double> > &oneDimForce,
                                                                       std::vector<double> &summedOneDimForce) {
    const int numberElements = fem.getNumberElements();
    const int numberNodesPerElement = fem.getNumberNodesPerElement();
    const int numberQuadPointsPerElement = fem.getNumberQuadPointsPerElement();
    const std::vector<std::vector<int> > &elementConnectivity = fem.getElementConnectivity();
    for (int ele = 0; ele != numberElements; ++ele) {
        int start = ele * numberQuadPointsPerElement;
        int end = (ele + 1) * numberQuadPointsPerElement;
        for (int iNode = 0; iNode != numberNodesPerElement; ++iNode) {
            int iGlobal = elementConnectivity[ele][iNode];
            for (int iquad = start; iquad != end; ++iquad) {
                summedOneDimForce[iGlobal] += oneDimForce[iNode][iquad];
            }
        }
    }
}