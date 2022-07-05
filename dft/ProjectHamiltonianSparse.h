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

#ifndef TUCKERDFTSPARSE_DFT_PROJECTHAMILTONIANSPARSE_H_
#define TUCKERDFTSPARSE_DFT_PROJECTHAMILTONIANSPARSE_H_

#include "../fem/FEM.h"
#include "../tensor/TuckerTensor.h"
#include "../atoms/NonLocalPSPData.h"
#include "../atoms/NonLocalMapManager.h"
#include "../atoms/AtomInformation.h"
#include "../atoms/NonLocalMap1D.h"

struct NonLocRowInfo {
    NonLocRowInfo(int atom_type,
                  int atom_type_local,
                  int atom_id,
                  int atom_id_local,
                  int lm_id,
                  PetscInt global_row_idx)
            : atom_type(atom_type),
              atom_type_local(atom_type_local),
              atom_id(atom_id),
              atom_id_local(atom_id_local),
              lm_id(lm_id),
              global_row_idx(global_row_idx) {}

    int atom_type;
    int atom_type_local;
    int atom_id;
    int atom_id_local;
    int lm_id;
    PetscInt global_row_idx;
};

class ProjectHamiltonianSparse {
public:
    ProjectHamiltonianSparse(const FEM &femX,
                             const FEM &femY,
                             const FEM &femZ,
                             const FEM &femLinearX,
                             const FEM &femLinearY,
                             const FEM &femLinearZ,
                             const std::vector<std::vector<double>> &eigX,
                             const std::vector<std::vector<double>> &eigY,
                             const std::vector<std::vector<double>> &eigZ);

    void Create_Hloc(const int rankVeffX,
                     const int rankVeffY,
                     const int rankVeffZ,
                     const int rankTuckerBasisX,
                     const int rankTuckerBasisY,
                     const int rankTuckerBasisZ,
                     const TuckerMPI::TuckerTensor *tuckerDecomposedVeff,
                     double truncation);

    void Create_Cnloc(const FEM &femNonLocX,
                      const FEM &femNonLocY,
                      const FEM &femNonLocZ,
                      const FEM &femNonLocLinearX,
                      const FEM &femNonLocLinearY,
                      const FEM &femNonLocLinearZ,
                      const int rankTuckerBasisX,
                      const int rankTuckerBasisY,
                      const int rankTuckerBasisZ,
                      const std::vector<std::shared_ptr<NonLocalPSPData>> &nonLocalPSPData,
                      const AtomInformation &atom_information,
                      double truncation);

    void computeRhoOut(Mat &eigenVectorTucker,
                       const std::vector<double> &occupancyFactor,
                       Tensor3DMPI &rhoNodalOut,
                       Tensor3DMPI &rhoGridOut);

    void Destroy();

    virtual ~ProjectHamiltonianSparse();

    Mat H_loc, C_nloc, C_nloc_trans;
private:
    void Create_Cnloc_mat_info(const AtomInformation &atom_information,
                               PetscInt &num_global_rows,
                               PetscInt &num_local_rows,
                               PetscInt &owned_row_start,
                               PetscInt &owned_row_end,
                               std::vector<NonLocRowInfo> &owned_rows_info,
                               std::vector<int> &owned_atom_id,
                               std::map<int, int> &owned_atom_id_g2l,
                               std::vector<int> &owned_atom_types,
                               std::map<int, int> &owned_atom_types_g2l);

    void Compute_nnz_loc(const TuckerMPI::TuckerTensor *tuckerDecomposedVeff,
                         const int rankVeffX,
                         const int rankVeffY,
                         const int rankVeffZ,
                         const int rankTuckerBasisX,
                         const int rankTuckerBasisY,
                         const int rankTuckerBasisZ,
                         double truncation,
                         std::vector<PetscInt> &Hloc_dnz,
                         std::vector<PetscInt> &Hloc_onz);

    void computeOverlapKineticTuckerBasis(const FEM &femX,
                                          const FEM &femY,
                                          const FEM &femZ,
                                          int rankTuckerBasisX,
                                          int rankTuckerBasisY,
                                          int rankTuckerBasisZ,
                                          const std::vector<std::vector<double>> &basisXQuadValues,
                                          const std::vector<std::vector<double>> &basisDXQuadValues,
                                          const std::vector<std::vector<double>> &basisYQuadValues,
                                          const std::vector<std::vector<double>> &basisDYQuadValues,
                                          const std::vector<std::vector<double>> &basisZQuadValues,
                                          const std::vector<std::vector<double>> &basisDZQuadValues,
                                          std::vector<std::vector<double> > &MatX,
                                          std::vector<std::vector<double> > &MatY,
                                          std::vector<std::vector<double> > &MatZ,
                                          std::vector<std::vector<double> > &MatDX,
                                          std::vector<std::vector<double> > &MatDY,
                                          std::vector<std::vector<double> > &MatDZ);

    void computeOverlapPotentialTuckerBasis(const FEM &femX,
                                            const FEM &femY,
                                            const FEM &femZ,
                                            const TuckerMPI::TuckerTensor *tuckerDecomposedVeff,
                                            const std::vector<std::vector<double>> &basisXQuadValues,
                                            const std::vector<std::vector<double>> &basisYQuadValues,
                                            const std::vector<std::vector<double>> &basisZQuadValues,
                                            int rankVeffX,
                                            int rankVeffY,
                                            int rankVeffZ,
                                            int rankTuckerBasisX,
                                            int rankTuckerBasisY,
                                            int rankTuckerBasisZ,
                                            std::vector<std::vector<std::vector<double> > > &MatPotX,
                                            std::vector<std::vector<std::vector<double> > > &MatPotY,
                                            std::vector<std::vector<std::vector<double> > > &MatPotZ);

    void computeFEMNL(const FEM &fem,
                      const FEM &femNonLoc,
                      const std::vector<std::vector<double>> &basis,
                      const NonLocalMap::NonLocalMap1D &nonloc_map1d,
                      std::vector<std::vector<double> > &basis_nonloc);

    void computeFactorsProduct(const FEM &fem,
                               const FEM &femNonLoc,
                               const std::vector<std::vector<double>> &basis,
                               const int rankNloc,
                               const int lMax,
                               const std::vector<Tucker::Matrix *> &factor_nloc,
                               std::vector<std::vector<std::vector<double> > > &factor_product);

    void computeOverlapNonLocPotentialTuckerBasis(int nonloc_compact_support_size,
                                                  std::vector<std::vector<double>> &basis_nonloc,
                                                  std::vector<std::vector<double>> &matNlocTimesJacobTimesWeight,
                                                  Tucker::Matrix *&mat_nonloc);

    const FEM &femX;
    const FEM &femY;
    const FEM &femZ;
    const FEM &femLinearX;
    const FEM &femLinearY;
    const FEM &femLinearZ;
    const std::vector<std::vector<double> > &eigX;
    const std::vector<std::vector<double> > &eigY;
    const std::vector<std::vector<double> > &eigZ;
    std::vector<std::vector<double> > basisXQuadValues, basisDXQuadValues;
    std::vector<std::vector<double> > basisYQuadValues, basisDYQuadValues;
    std::vector<std::vector<double> > basisZQuadValues, basisDZQuadValues;

    std::vector<PetscInt> Hloc_dnz;
    std::vector<PetscInt> Hloc_onz;
};

#endif //TUCKERDFTSPARSE_DFT_PROJECTHAMILTONIANSPARSE_H_
