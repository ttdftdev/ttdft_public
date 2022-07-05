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
#include "InitialGuess.h"
#include "../alglib/src/ap.h"
#include "../alglib/src/interpolation.h"
#include "KSDFTEnergyFunctional.h"

namespace {
    void construct3DSplineObject(const std::string &nodalFileX,
                                 const std::string &nodalFileY,
                                 const std::string &nodalFileZ,
                                 const std::string &fieldFile,
                                 alglib::spline3dinterpolant &spline3dinterpolant);

    void read_in_factor_matrices(const std::string &filename,
                                 std::vector<alglib::spline1dinterpolant> &spline1dinterpolant);

    void construct_local_factor_matrices(const std::vector<double> &coord,
                                         int rank,
                                         int istart,
                                         int iend,
                                         std::vector<alglib::spline1dinterpolant> &function,
                                         double *data);

    double find_largest_value(const std::string &nodes_filename) {
        std::fstream fin(nodes_filename);
        double temp, largest_value = 0;
        while (fin >> temp) {
            if (std::abs(temp) > largest_value) largest_value = temp;
        }
        fin.clear();
        fin.close();
        return largest_value;
    }
}

void InitialGuess::initializeRhoNodal(const FEM &femX,
                                      const FEM &femY,
                                      const FEM &femZ,
                                      Tensor3DMPI &rhoNodal) {
    std::array<int, 6> rhoNodalGlobalIdx;
    double *rhoNodalLocal = rhoNodal.getLocalData(rhoNodalGlobalIdx);

    const std::vector<double> &globalCoordX = femX.getGlobalNodalCoord();
    const std::vector<double> &globalCoordY = femY.getGlobalNodalCoord();
    const std::vector<double> &globalCoordZ = femZ.getGlobalNodalCoord();

    int cnt = 0;
    for (int k = rhoNodalGlobalIdx[4]; k < rhoNodalGlobalIdx[5]; ++k) {
        for (int j = rhoNodalGlobalIdx[2]; j < rhoNodalGlobalIdx[3]; ++j) {
            for (int i = rhoNodalGlobalIdx[0]; i < rhoNodalGlobalIdx[1]; ++i) {
                double r = std::sqrt(globalCoordX[i] * globalCoordX[i] + globalCoordY[j] * globalCoordY[j] +
                                     globalCoordZ[k] * globalCoordZ[k]);
//                rhoNodalLocal[cnt] = (1.0 / M_PI) * std::exp(-2.0 * r);
                rhoNodalLocal[cnt] = r * r * std::exp(-2.0 * r);
                cnt = cnt + 1;
            }
        }
    }
}

void InitialGuess::initializeRhoGrid(const FEM &femX,
                                     const FEM &femY,
                                     const FEM &femZ,
                                     Tensor3DMPI &rhoGrid) {
    std::array<int, 6> rhoGridGlobalIdx;
    double *rhoGridLocal = rhoGrid.getLocalData(rhoGridGlobalIdx);

    const std::vector<double> &quadCoordX = femX.getPositionQuadPointValues();
    const std::vector<double> &quadCoordY = femY.getPositionQuadPointValues();
    const std::vector<double> &quadCoordZ = femZ.getPositionQuadPointValues();

    int cnt = 0;
    for (int k = rhoGridGlobalIdx[4]; k < rhoGridGlobalIdx[5]; ++k) {
        for (int j = rhoGridGlobalIdx[2]; j < rhoGridGlobalIdx[3]; ++j) {
            for (int i = rhoGridGlobalIdx[0]; i < rhoGridGlobalIdx[1]; ++i) {
                double r = std::sqrt(quadCoordX[i] * quadCoordX[i] + quadCoordY[j] * quadCoordY[j] +
                                     quadCoordZ[k] * quadCoordZ[k]);
//                rhoGridLocal[cnt] = (1.0 / M_PI) * std::exp(-2.0 * r);
                rhoGridLocal[cnt] = r * r * std::exp(-2.0 * r);
                cnt = cnt + 1;
            }
        }
    }
}

void InitialGuess::initialize_rho_by_hydrogen(const FEM &femX,
                                              const FEM &femY,
                                              const FEM &femZ,
                                              const std::vector<std::vector<double> > &nuclei,
                                              Tensor3DMPI &rhoNodal) {
    std::array<int, 6> rhoNodalGlobalIdx;
    double *rhoNodalLocal = rhoNodal.getLocalData(rhoNodalGlobalIdx);

    const std::vector<double> &globalCoordX = femX.getGlobalNodalCoord();
    const std::vector<double> &globalCoordY = femY.getGlobalNodalCoord();
    const std::vector<double> &globalCoordZ = femZ.getGlobalNodalCoord();

    int cnt = 0;
    for (int k = rhoNodalGlobalIdx[4]; k < rhoNodalGlobalIdx[5]; ++k) {
        for (int j = rhoNodalGlobalIdx[2]; j < rhoNodalGlobalIdx[3]; ++j) {
            for (int i = rhoNodalGlobalIdx[0]; i < rhoNodalGlobalIdx[1]; ++i) {
                for (auto &atom: nuclei) {
                    double x = globalCoordX[i] - atom[1];
                    double y = globalCoordY[j] - atom[2];
                    double z = globalCoordZ[k] - atom[3];
                    double r = std::sqrt(x * x + y * y + z * z);
                    rhoNodalLocal[cnt] = (1.0 / M_PI) * std::exp(-2.0 * r);
//          rhoNodalLocal[cnt] += r * r * std::exp(-2.0 * r);
                }
                cnt = cnt + 1;
            }
        }
    }

}

void InitialGuess::initialize_rho_grid_by_hydrogen(const FEM &femX,
                                                   const FEM &femY,
                                                   const FEM &femZ,
                                                   const std::vector<std::vector<double> > &nuclei,
                                                   Tensor3DMPI &rhoGrid) {
    std::array<int, 6> rhoGridGlobalIdx;
    double *rhoGridLocal = rhoGrid.getLocalData(rhoGridGlobalIdx);

    const std::vector<double> &quadCoordX = femX.getPositionQuadPointValues();
    const std::vector<double> &quadCoordY = femY.getPositionQuadPointValues();
    const std::vector<double> &quadCoordZ = femZ.getPositionQuadPointValues();

    int cnt = 0;
    for (int k = rhoGridGlobalIdx[4]; k < rhoGridGlobalIdx[5]; ++k) {
        for (int j = rhoGridGlobalIdx[2]; j < rhoGridGlobalIdx[3]; ++j) {
            for (int i = rhoGridGlobalIdx[0]; i < rhoGridGlobalIdx[1]; ++i) {
                for (auto &atom: nuclei) {
                    double x = quadCoordX[i] - atom[1];
                    double y = quadCoordY[j] - atom[2];
                    double z = quadCoordZ[k] - atom[3];
                    double r = std::sqrt(x * x + y * y + z * z);
                    rhoGridLocal[cnt] = (1.0 / M_PI) * std::exp(-2.0 * r);
//          rhoGridLocal[cnt] += r * r * std::exp(-2.0 * r);
                }
                cnt = cnt + 1;
            }
        }
    }
}

void InitialGuess::initilize_rho_from_single_atom_fem(const FEM &femX,
                                                      const FEM &femY,
                                                      const FEM &femZ,
                                                      const std::vector<std::vector<double> > &nuclei,
                                                      std::string nodalFileX,
                                                      std::string nodalFileY,
                                                      std::string nodalFileZ,
                                                      std::string fieldFile,
                                                      Tensor3DMPI &rhoNodal,
                                                      Tensor3DMPI &rhoGrid) {
    rhoNodal.setEntriesZero();
    rhoGrid.setEntriesZero();
    double largest_x = find_largest_value(nodalFileX);
    double largest_y = find_largest_value(nodalFileY);
    double largest_z = find_largest_value(nodalFileZ);

    alglib::spline3dinterpolant spline3dinterpolant;
    construct3DSplineObject(nodalFileX,
                            nodalFileY,
                            nodalFileZ,
                            fieldFile,
                            spline3dinterpolant);

    const std::vector<double> &nodalX = femX.getGlobalNodalCoord();
    const std::vector<double> &nodalY = femY.getGlobalNodalCoord();
    const std::vector<double> &nodalZ = femZ.getGlobalNodalCoord();
    const std::vector<double> &quadX = femX.getPositionQuadPointValues();
    const std::vector<double> &quadY = femY.getPositionQuadPointValues();
    const std::vector<double> &quadZ = femZ.getPositionQuadPointValues();

    std::array<int, 6> nodalIdx, quadIdx;
    double *nodalFieldData = rhoNodal.getLocalData(nodalIdx);
    double *quadFieldData = rhoGrid.getLocalData(quadIdx);
    int cnt = 0;
    for (int k = nodalIdx[4]; k < nodalIdx[5]; ++k) {
        for (int j = nodalIdx[2]; j < nodalIdx[3]; ++j) {
            for (int i = nodalIdx[0]; i < nodalIdx[1]; ++i) {
                for (auto &atom: nuclei) {
                    double xx = nodalX[i] - atom[1];
                    double yy = nodalY[j] - atom[2];
                    double zz = nodalZ[k] - atom[3];
                    double val = 0.0;
//          if(std::abs(xx) < largest_x && std::abs(yy) < largest_y && std::abs(zz) < largest_z)
                    val = alglib::spline3dcalc(spline3dinterpolant,
                                               xx,
                                               yy,
                                               zz);
                    nodalFieldData[cnt] += val;
                }
                cnt++;
            }
        }
    }
    cnt = 0;
    for (int k = quadIdx[4]; k < quadIdx[5]; ++k) {
        for (int j = quadIdx[2]; j < quadIdx[3]; ++j) {
            for (int i = quadIdx[0]; i < quadIdx[1]; ++i) {
                for (auto &atom: nuclei) {
                    double xx = quadX[i] - atom[1];
                    double yy = quadY[j] - atom[2];
                    double zz = quadZ[k] - atom[3];
                    double val = 0.0;
//          if(std::abs(xx) < largest_x && std::abs(yy) < largest_y && std::abs(zz) < largest_z)
                    val = alglib::spline3dcalc(spline3dinterpolant,
                                               xx,
                                               yy,
                                               zz);
                    quadFieldData[cnt] += val;
                }
                cnt++;
            }
        }
    }

}

void InitialGuess::initilize_rho_from_previous_calculation_fem(const FEM &femX,
                                                               const FEM &femY,
                                                               const FEM &femZ,
                                                               std::string nodalFileX,
                                                               std::string nodalFileY,
                                                               std::string nodalFileZ,
                                                               std::string fieldFile,
                                                               Tensor3DMPI &rhoNodal,
                                                               Tensor3DMPI &rhoGrid) {
    rhoNodal.setEntriesZero();
    rhoGrid.setEntriesZero();

    double largest_x = find_largest_value(nodalFileX);
    double largest_y = find_largest_value(nodalFileY);
    double largest_z = find_largest_value(nodalFileZ);

    alglib::spline3dinterpolant spline3dinterpolant;
    construct3DSplineObject(nodalFileX,
                            nodalFileY,
                            nodalFileZ,
                            fieldFile,
                            spline3dinterpolant);

    const std::vector<double> &nodalX = femX.getGlobalNodalCoord();
    const std::vector<double> &nodalY = femY.getGlobalNodalCoord();
    const std::vector<double> &nodalZ = femZ.getGlobalNodalCoord();
    const std::vector<double> &quadX = femX.getPositionQuadPointValues();
    const std::vector<double> &quadY = femY.getPositionQuadPointValues();
    const std::vector<double> &quadZ = femZ.getPositionQuadPointValues();

    std::array<int, 6> nodalIdx, quadIdx;
    double *nodalFieldData = rhoNodal.getLocalData(nodalIdx);
    double *quadFieldData = rhoGrid.getLocalData(quadIdx);
    int cnt = 0;
    for (int k = nodalIdx[4]; k < nodalIdx[5]; ++k) {
        for (int j = nodalIdx[2]; j < nodalIdx[3]; ++j) {
            for (int i = nodalIdx[0]; i < nodalIdx[1]; ++i) {
                double xx = nodalX[i];
                double yy = nodalY[j];
                double zz = nodalZ[k];
                double val = 0.0;
//        if(std::abs(xx) < largest_x && std::abs(yy) < largest_y && std::abs(zz) < largest_z)
                val = alglib::spline3dcalc(spline3dinterpolant,
                                           xx,
                                           yy,
                                           zz);
                nodalFieldData[cnt] += val;
                cnt++;
            }
        }
    }
    cnt = 0;
    for (int k = quadIdx[4]; k < quadIdx[5]; ++k) {
        for (int j = quadIdx[2]; j < quadIdx[3]; ++j) {
            for (int i = quadIdx[0]; i < quadIdx[1]; ++i) {
                double xx = quadX[i];
                double yy = quadY[j];
                double zz = quadZ[k];
                double val = 0.0;
//        if(std::abs(xx) < largest_x && std::abs(yy) < largest_y && std::abs(zz) < largest_z)
                val = alglib::spline3dcalc(spline3dinterpolant,
                                           xx,
                                           yy,
                                           zz);
                quadFieldData[cnt] += val;
                cnt++;
            }
        }
    }

}

void InitialGuess::initializeRhoFromFile(const FEM &femX,
                                         const FEM &femY,
                                         const FEM &femZ,
                                         const std::vector<std::vector<double> > &nuclei,
                                         const int numberAtomType,
                                         const std::vector<std::string> &nodalFileX,
                                         const std::vector<std::string> &nodalFileY,
                                         const std::vector<std::string> &nodalFileZ,
                                         const std::vector<std::string> &fieldFile,
                                         Tensor3DMPI &rhoNodal,
                                         Tensor3DMPI &rhoGrid) {
    rhoNodal.setEntriesZero();
    rhoGrid.setEntriesZero();

    std::vector<alglib::spline3dinterpolant> rhoFunc(numberAtomType);
    for (int iAtomType = 0; iAtomType < numberAtomType; ++iAtomType) {
        construct3DSplineObject(nodalFileX[iAtomType],
                                nodalFileY[iAtomType],
                                nodalFileZ[iAtomType],
                                fieldFile[iAtomType],
                                rhoFunc[iAtomType]);
    }

    const std::vector<double> &nodalX = femX.getGlobalNodalCoord();
    const std::vector<double> &nodalY = femY.getGlobalNodalCoord();
    const std::vector<double> &nodalZ = femZ.getGlobalNodalCoord();
    const std::vector<double> &quadX = femX.getPositionQuadPointValues();
    const std::vector<double> &quadY = femY.getPositionQuadPointValues();
    const std::vector<double> &quadZ = femZ.getPositionQuadPointValues();

    std::array<int, 6> nodalIdx, quadIdx;
    double *nodalFieldData = rhoNodal.getLocalData(nodalIdx);
    double *quadFieldData = rhoGrid.getLocalData(quadIdx);
    for (auto &atom: nuclei) {
        int atomType = atom[4];
        int cnt = 0;
        for (int k = nodalIdx[4]; k < nodalIdx[5]; ++k) {
            for (int j = nodalIdx[2]; j < nodalIdx[3]; ++j) {
                for (int i = nodalIdx[0]; i < nodalIdx[1]; ++i) {
                    double xx = nodalX[i] - atom[1];
                    double yy = nodalY[j] - atom[2];
                    double zz = nodalZ[k] - atom[3];
                    nodalFieldData[cnt] += alglib::spline3dcalc(rhoFunc[atomType],
                                                                xx,
                                                                yy,
                                                                zz);
                    cnt++;
                }
            }
        }
    }

    for (auto &atom: nuclei) {
        int atomType = atom[4];
        int cnt = 0;
        for (int k = quadIdx[4]; k < quadIdx[5]; ++k) {
            for (int j = quadIdx[2]; j < quadIdx[3]; ++j) {
                for (int i = quadIdx[0]; i < quadIdx[1]; ++i) {
                    double xx = quadX[i] - atom[1];
                    double yy = quadY[j] - atom[2];
                    double zz = quadZ[k] - atom[3];
                    quadFieldData[cnt] += alglib::spline3dcalc(rhoFunc[atomType],
                                                               xx,
                                                               yy,
                                                               zz);
                    cnt++;
                }
            }
        }
    }

}

void InitialGuess::initializePsi(const FEM &femX,
                                 const FEM &femY,
                                 const FEM &femZ,
                                 const std::vector<std::vector<double> > &nuclei,
                                 Tensor3DMPI &psiNodal) {
    std::array<int, 6> psiNodalGlobalIdx;
    double *psiNodalLocal = psiNodal.getLocalData(psiNodalGlobalIdx);

    const std::vector<double> &globalCoordX = femX.getGlobalNodalCoord();
    const std::vector<double> &globalCoordY = femY.getGlobalNodalCoord();
    const std::vector<double> &globalCoordZ = femZ.getGlobalNodalCoord();

    int cnt = 0;
    for (int k = psiNodalGlobalIdx[4]; k < psiNodalGlobalIdx[5]; ++k) {
        for (int j = psiNodalGlobalIdx[2]; j < psiNodalGlobalIdx[3]; ++j) {
            for (int i = psiNodalGlobalIdx[0]; i < psiNodalGlobalIdx[1]; ++i) {
                for (auto &atom: nuclei) {
                    double x = globalCoordX[i] - atom[1];
                    double y = globalCoordY[j] - atom[2];
                    double z = globalCoordZ[k] - atom[3];
                    double r = std::sqrt(x * x + y * y + z * z);
//                    rhoNodalLocal[cnt] = (1.0 / M_PI) * std::exp(-2.0 * r);
                    psiNodalLocal[cnt] = (1.0 / std::sqrt(M_PI)) * std::exp(-r);//r * std::exp(-r);
                }
                cnt = cnt + 1;
            }
        }
    }

}

void InitialGuess::initializePsiFromFile(const FEM &femX,
                                         const FEM &femY,
                                         const FEM &femZ,
                                         const std::vector<std::vector<double> > &nuclei,
                                         std::string nodalFileX,
                                         std::string nodalFileY,
                                         std::string nodalFileZ,
                                         std::string fieldFile,
                                         int option, // 1: use superposition wrt atom 2: interpolate as it is
                                         Tensor3DMPI &psiNodal) {
    psiNodal.setEntriesZero();

    double largest_x = find_largest_value(nodalFileX);
    double largest_y = find_largest_value(nodalFileY);
    double largest_z = find_largest_value(nodalFileZ);

    alglib::spline3dinterpolant spline3dinterpolant;
    construct3DSplineObject(nodalFileX,
                            nodalFileY,
                            nodalFileZ,
                            fieldFile,
                            spline3dinterpolant);

    const std::vector<double> &nodalX = femX.getGlobalNodalCoord();
    const std::vector<double> &nodalY = femY.getGlobalNodalCoord();
    const std::vector<double> &nodalZ = femZ.getGlobalNodalCoord();

    std::array<int, 6> nodalIdx;
    double *nodalFieldData = psiNodal.getLocalData(nodalIdx);
    int cnt = 0;
    if (option == 1) {
        for (int k = nodalIdx[4]; k < nodalIdx[5]; ++k) {
            for (int j = nodalIdx[2]; j < nodalIdx[3]; ++j) {
                for (int i = nodalIdx[0]; i < nodalIdx[1]; ++i) {
                    for (auto &atom: nuclei) {
                        double xx = nodalX[i] - atom[1];
                        double yy = nodalY[j] - atom[2];
                        double zz = nodalZ[k] - atom[3];
                        double val = 0.0;
//            if (std::abs(xx) < largest_x && std::abs(yy) < largest_y && std::abs(zz) < largest_z)
                        val = alglib::spline3dcalc(spline3dinterpolant,
                                                   xx,
                                                   yy,
                                                   zz);
                        nodalFieldData[cnt] += val;
                    }
                    cnt++;
                }
            }
        }
    } else {
        for (int k = nodalIdx[4]; k < nodalIdx[5]; ++k) {
            for (int j = nodalIdx[2]; j < nodalIdx[3]; ++j) {
                for (int i = nodalIdx[0]; i < nodalIdx[1]; ++i) {
                    double xx = nodalX[i];
                    double yy = nodalY[j];
                    double zz = nodalZ[k];
                    double val = 0.0;
//          if (std::abs(xx) < largest_x && std::abs(yy) < largest_y && std::abs(zz) < largest_z)
                    val = alglib::spline3dcalc(spline3dinterpolant,
                                               xx,
                                               yy,
                                               zz);
                    nodalFieldData[cnt] += val;
                    cnt++;
                }
            }
        }
    }

    Tensor3DMPI psiSquareNodal(psiNodal);
    double *psiSquareNodalData = psiSquareNodal.getLocalData();
    for (int i = 0; i < psiSquareNodal.getLocalNumberEntries(); ++i) {
        psiSquareNodalData[i] = nodalFieldData[i] * nodalFieldData[i];
    }

    double normalization_factor =
            KSDFTEnergyFunctional::compute3DIntegralTuckerCuboidNodal(psiSquareNodal,
                                                                      25,
                                                                      25,
                                                                      25,
                                                                      femX,
                                                                      femY,
                                                                      femZ);
    normalization_factor = std::sqrt(normalization_factor);
    for (int i = 0; i < psiNodal.getLocalNumberEntries(); ++i) {
        nodalFieldData[i] = nodalFieldData[i] / normalization_factor;
    }

}

void InitialGuess::initializePsiFromFile(const FEM &femX,
                                         const FEM &femY,
                                         const FEM &femZ,
                                         const std::vector<std::vector<double> > &nuclei,
                                         const int numberAtomType,
                                         const std::vector<std::string> &nodalFileX,
                                         const std::vector<std::string> &nodalFileY,
                                         const std::vector<std::string> &nodalFileZ,
                                         const std::vector<std::string> &fieldFile,
                                         Tensor3DMPI &psiNodal) {
    psiNodal.setEntriesZero();

    int taskId;
    MPI_Comm_rank(MPI_COMM_WORLD,
                  &taskId);

    std::vector<alglib::spline3dinterpolant> psiFunc(numberAtomType);
    for (int iAtomType = 0; iAtomType < numberAtomType; ++iAtomType) {
        construct3DSplineObject(nodalFileX[iAtomType],
                                nodalFileY[iAtomType],
                                nodalFileZ[iAtomType],
                                fieldFile[iAtomType],
                                psiFunc[iAtomType]);
    }

    const std::vector<double> &nodalX = femX.getGlobalNodalCoord();
    const std::vector<double> &nodalY = femY.getGlobalNodalCoord();
    const std::vector<double> &nodalZ = femZ.getGlobalNodalCoord();

    std::array<int, 6> nodalIdx;
    double *nodalFieldData = psiNodal.getLocalData(nodalIdx);
    for (auto &atom: nuclei) {
        double atomCharge = atom[0];
        int atomType = atom[4];
        int cnt = 0;
        for (int k = nodalIdx[4]; k < nodalIdx[5]; ++k) {
            for (int j = nodalIdx[2]; j < nodalIdx[3]; ++j) {
                for (int i = nodalIdx[0]; i < nodalIdx[1]; ++i) {
                    double xx = nodalX[i] - atom[1];
                    double yy = nodalY[j] - atom[2];
                    double zz = nodalZ[k] - atom[3];
                    nodalFieldData[cnt] += atomCharge * alglib::spline3dcalc(psiFunc[atomType],
                                                                             xx,
                                                                             yy,
                                                                             zz);
                }
                cnt++;
            }
        }
    }

}

void InitialGuess::initializeLocPSPGrid(const FEM &femX,
                                        const FEM &femY,
                                        const FEM &femZ,
                                        const std::vector<std::vector<double> > &nuclei,
                                        double cutoffRadius,
                                        Tensor3DMPI &localPSP) {

}

void InitialGuess::initializeRhoFromRadiusFile(const FEM &femX,
                                               const FEM &femY,
                                               const FEM &femZ,
                                               const std::vector<std::vector<double> > &nuclei,
                                               std::string filename,
                                               Tensor3DMPI &rhoNodal,
                                               Tensor3DMPI &rhoGrid) {
    std::vector<double> vec_radius, vec_rho;
    int taskId;
    MPI_Comm_rank(MPI_COMM_WORLD,
                  &taskId);

    if (taskId == 0) {
        double temp_r, temp_rho;
        std::ifstream fin;
        fin.open(filename);
        while (fin >> temp_r >> temp_rho) {
            vec_radius.emplace_back(temp_r);
            vec_rho.emplace_back(temp_rho);
        }
        fin.close();
        fin.clear();
    }
    int vec_size = vec_radius.size();
    MPI_Bcast(&vec_size,
              1,
              MPI_INT,
              0,
              MPI_COMM_WORLD);
    if (taskId != 0) {
        vec_radius = std::vector<double>(vec_size,
                                         0.0);
        vec_rho = std::vector<double>(vec_size,
                                      0.0);
    }
    MPI_Bcast(vec_radius.data(),
              vec_size,
              MPI_DOUBLE,
              0,
              MPI_COMM_WORLD);
    MPI_Bcast(vec_rho.data(),
              vec_size,
              MPI_DOUBLE,
              0,
              MPI_COMM_WORLD);

    alglib::real_1d_array radius, rho;
    radius.setcontent(vec_radius.size(),
                      vec_radius.data());
    rho.setcontent(vec_rho.size(),
                   vec_rho.data());

    alglib::spline1dinterpolant spline1dinterpolant;
    alglib::spline1dbuildcubic(radius,
                               rho,
                               vec_size,
                               1,
                               0,
                               1,
                               0,
                               spline1dinterpolant);


    const std::vector<double> &nodalX = femX.getGlobalNodalCoord();
    const std::vector<double> &nodalY = femY.getGlobalNodalCoord();
    const std::vector<double> &nodalZ = femZ.getGlobalNodalCoord();
    const std::vector<double> &quadX = femX.getPositionQuadPointValues();
    const std::vector<double> &quadY = femY.getPositionQuadPointValues();
    const std::vector<double> &quadZ = femZ.getPositionQuadPointValues();

    std::array<int, 6> nodalIdx, quadIdx;
    double *nodalFieldData = rhoNodal.getLocalData(nodalIdx);
    double *quadFieldData = rhoGrid.getLocalData(quadIdx);
    int cnt = 0;
    for (int k = nodalIdx[4]; k < nodalIdx[5]; ++k) {
        for (int j = nodalIdx[2]; j < nodalIdx[3]; ++j) {
            for (int i = nodalIdx[0]; i < nodalIdx[1]; ++i) {
                for (auto &atom: nuclei) {
//        for (int q = 0; q < 10; ++q) { auto &atom = nuclei[q];
                    double xx = nodalX[i] - atom[1];
                    double yy = nodalY[j] - atom[2];
                    double zz = nodalZ[k] - atom[3];
                    double rr = std::sqrt(xx * xx + yy * yy + zz * zz);
                    double val = alglib::spline1dcalc(spline1dinterpolant,
                                                      rr);
                    nodalFieldData[cnt] += val;
                }
                cnt++;
            }
        }
    }
    cnt = 0;
    for (int k = quadIdx[4]; k < quadIdx[5]; ++k) {
        for (int j = quadIdx[2]; j < quadIdx[3]; ++j) {
            for (int i = quadIdx[0]; i < quadIdx[1]; ++i) {
                for (auto &atom: nuclei) {
                    double xx = quadX[i] - atom[1];
                    double yy = quadY[j] - atom[2];
                    double zz = quadZ[k] - atom[3];
                    double rr = std::sqrt(xx * xx + yy * yy + zz * zz);
                    double val = alglib::spline1dcalc(spline1dinterpolant,
                                                      rr);
                    quadFieldData[cnt] += val;
                }
                cnt++;
            }
        }
    }
}

void InitialGuess::initialize_rho(const FEM &femX,
                                  const FEM &femY,
                                  const FEM &femZ,
                                  InputParameter &input_parameter,
                                  AtomInformation &atomInformation,
                                  Tensor3DMPI &rhoNodal,
                                  Tensor3DMPI &rhoGrid) {
    rhoNodal.setEntriesZero();
    rhoGrid.setEntriesZero();
    if (input_parameter.using_initial_guess_electron_density == RhoIGType::RADIALDATA) {
        PetscPrintf(PETSC_COMM_WORLD,
                    "using initial guesses from superposition of single atoms with radius data.\n");
        const std::vector<std::vector<std::vector<double>>> &nuclei = atomInformation.nuclei;
        for (int atom_type = 0; atom_type < nuclei.size(); ++atom_type) {
            std::string density_filename = std::string("Density_AT") + std::to_string(atom_type);
            initializeRhoFromRadiusFile(femX,
                                        femY,
                                        femZ,
                                        nuclei[atom_type],
                                        density_filename,
                                        rhoNodal,
                                        rhoGrid);
        }

    } else if (input_parameter.using_initial_guess_electron_density == RhoIGType::FEM_SINGLEATOM_DATA) {
        PetscPrintf(PETSC_COMM_WORLD,
                    "using initial guesses from superposition of single atoms on fem grid.\n");
        initilize_rho_from_single_atom_fem(femX,
                                           femY,
                                           femZ,
                                           atomInformation.all_nuclei,
                                           input_parameter.ig_fem_x_electron_density_filename,
                                           input_parameter.ig_fem_y_electron_density_filename,
                                           input_parameter.ig_fem_z_electron_density_filename,
                                           input_parameter.ig_electron_density_3d_filename,
                                           rhoNodal,
                                           rhoGrid);
    } else if (input_parameter.using_initial_guess_electron_density == RhoIGType::FEM_READ_IN) {
        PetscPrintf(PETSC_COMM_WORLD,
                    "using initial guesses from previous calculation on fem grid.\n");
        initilize_rho_from_previous_calculation_fem(femX,
                                                    femY,
                                                    femZ,
                                                    input_parameter.ig_fem_x_electron_density_filename,
                                                    input_parameter.ig_fem_y_electron_density_filename,
                                                    input_parameter.ig_fem_z_electron_density_filename,
                                                    input_parameter.ig_electron_density_3d_filename,
                                                    rhoNodal,
                                                    rhoGrid);
    } else if (input_parameter.using_initial_guess_electron_density == RhoIGType::TUCKER_READ_IN) {
        PetscPrintf(PETSC_COMM_WORLD,
                    "using initial guesses from previous calculation with decomposed tucker rho.\n");
        initialize_rho_tucker(femX,
                              femY,
                              femZ,
                              rhoNodal,
                              rhoGrid);
    } else if (input_parameter.using_initial_guess_electron_density == RhoIGType::DFTFE) {
        PetscPrintf(PETSC_COMM_WORLD,
                    "using initial guesses from DFT FE results.\n");
        initilize_rho_from_dftfe(femX,
                                 femY,
                                 femZ,
                                 rhoNodal,
                                 rhoGrid);
    } else {
        PetscPrintf(PETSC_COMM_WORLD,
                    "using initial guesses constructed from superposition of hydrogen atom.\n");
        initialize_rho_by_hydrogen(femX,
                                   femY,
                                   femZ,
                                   atomInformation.all_nuclei,
                                   rhoNodal);
        initialize_rho_grid_by_hydrogen(femX,
                                        femY,
                                        femZ,
                                        atomInformation.all_nuclei,
                                        rhoGrid);
    }
}

void InitialGuess::initialize_rho_tucker(const FEM &femX,
                                         const FEM &femY,
                                         const FEM &femZ,
                                         Tensor3DMPI &rhoNodal,
                                         Tensor3DMPI &rhoGrid) {
    std::vector<alglib::spline1dinterpolant> factor_x, factor_y, factor_z;
    read_in_factor_matrices("tucker_input_factor_x",
                            factor_x);
    read_in_factor_matrices("tucker_input_factor_y",
                            factor_y);
    read_in_factor_matrices("tucker_input_factor_z",
                            factor_z);

    int taskId;
    MPI_Comm_rank(MPI_COMM_WORLD,
                  &taskId);

    Tucker::SizeArray core_rank(3);
    std::fstream fin;
    if (taskId == 0) {
        fin.open("tucker_input_core");
        fin >> core_rank[0] >> core_rank[1] >> core_rank[2];
    }
    MPI_Bcast(core_rank.data(),
              3,
              MPI_INT,
              0,
              MPI_COMM_WORLD);

    Tucker::Tensor core(core_rank);
    if (taskId == 0) {
        for (int i = 0; i < core_rank[0] * core_rank[1] * core_rank[2]; ++i)
            fin >> core.data()[i];
    }
    MPI_Bcast(core.data(),
              core_rank[0] * core_rank[1] * core_rank[2],
              MPI_DOUBLE,
              0,
              MPI_COMM_WORLD);

    std::array<int, 6> rhoNodal_idx = rhoNodal.getGlobalIndex();
    Tucker::Matrix Ux_nodal(rhoNodal_idx[1] - rhoNodal_idx[0],
                            core_rank[0]);
    Tucker::Matrix Uy_nodal(rhoNodal_idx[3] - rhoNodal_idx[2],
                            core_rank[1]);
    Tucker::Matrix Uz_nodal(rhoNodal_idx[5] - rhoNodal_idx[4],
                            core_rank[2]);

    const std::vector<double> &coord_x = femX.getGlobalNodalCoord();
    const std::vector<double> &coord_y = femX.getGlobalNodalCoord();
    const std::vector<double> &coord_z = femX.getGlobalNodalCoord();

    construct_local_factor_matrices(coord_x,
                                    core_rank[0],
                                    rhoNodal_idx[0],
                                    rhoNodal_idx[1],
                                    factor_x,
                                    Ux_nodal.data());
    construct_local_factor_matrices(coord_y,
                                    core_rank[1],
                                    rhoNodal_idx[2],
                                    rhoNodal_idx[3],
                                    factor_y,
                                    Uy_nodal.data());
    construct_local_factor_matrices(coord_z,
                                    core_rank[2],
                                    rhoNodal_idx[4],
                                    rhoNodal_idx[5],
                                    factor_z,
                                    Uz_nodal.data());

    Tucker::Tensor *temp;
    Tucker::Tensor *temp_recon;
    temp = &core;
    temp_recon = Tucker::ttm(temp,
                             0,
                             &Ux_nodal);
    temp = temp_recon;
    temp_recon = Tucker::ttm(temp,
                             1,
                             &Uy_nodal);
    Tucker::MemoryManager::safe_delete(temp);
    temp = temp_recon;
    temp_recon = Tucker::ttm(temp,
                             2,
                             &Uz_nodal);
    Tucker::MemoryManager::safe_delete(temp);
    std::copy(temp_recon->data(),
              temp_recon->data() + temp_recon->getNumElements(),
              rhoNodal.getLocalData());
    Tucker::MemoryManager::safe_delete(temp_recon);

    std::array<int, 6> rhoGrid_idx = rhoGrid.getGlobalIndex();
    Tucker::Matrix Ux_quad(rhoGrid_idx[1] - rhoGrid_idx[0],
                           core_rank[0]);
    Tucker::Matrix Uy_quad(rhoGrid_idx[3] - rhoGrid_idx[2],
                           core_rank[1]);
    Tucker::Matrix Uz_quad(rhoGrid_idx[5] - rhoGrid_idx[4],
                           core_rank[2]);
    const std::vector<double> &quad_x = femX.getPositionQuadPointValues();
    const std::vector<double> &quad_y = femY.getPositionQuadPointValues();
    const std::vector<double> &quad_z = femZ.getPositionQuadPointValues();
    construct_local_factor_matrices(quad_x,
                                    core_rank[0],
                                    rhoGrid_idx[0],
                                    rhoGrid_idx[1],
                                    factor_x,
                                    Ux_quad.data());
    construct_local_factor_matrices(quad_y,
                                    core_rank[1],
                                    rhoGrid_idx[2],
                                    rhoGrid_idx[3],
                                    factor_y,
                                    Uy_quad.data());
    construct_local_factor_matrices(quad_z,
                                    core_rank[2],
                                    rhoGrid_idx[4],
                                    rhoGrid_idx[5],
                                    factor_z,
                                    Uz_quad.data());

    temp = &core;
    temp_recon = Tucker::ttm(temp,
                             0,
                             &Ux_quad);
    temp = temp_recon;
    temp_recon = Tucker::ttm(temp,
                             1,
                             &Uy_quad);
    Tucker::MemoryManager::safe_delete(temp);
    temp = temp_recon;
    temp_recon = Tucker::ttm(temp,
                             2,
                             &Uz_quad);
    Tucker::MemoryManager::safe_delete(temp);
    std::copy(temp_recon->data(),
              temp_recon->data() + temp_recon->getNumElements(),
              rhoGrid.getLocalData());
    Tucker::MemoryManager::safe_delete(temp_recon);

}

void InitialGuess::initilize_rho_from_dftfe(const FEM &femX,
                                            const FEM &femY,
                                            const FEM &femZ,
                                            Tensor3DMPI &rhoNodal,
                                            Tensor3DMPI &rhoGrid) {
    int taskid;
    MPI_Comm_rank(MPI_COMM_WORLD,
                  &taskid);
    std::string filename = "dftfe_nodal/tensor_nodal_proc" + std::to_string(taskid) + ".dat";
    std::fstream fin;
    fin.open(filename,
             std::ios::in);

    PetscPrintf(MPI_COMM_WORLD,
                "loaded files from dft-fe results\n");

    int num_local_entries = rhoNodal.getLocalNumberEntries();
    double *nodal_data = rhoNodal.getLocalData();
    for (int i = 0; i < num_local_entries; ++i) {
        fin >> nodal_data[i];
    }

    PetscPrintf(MPI_COMM_WORLD,
                "read in files finished\n");

    const TuckerMPI::TuckerTensor *ttensor = TuckerMPI::STHOSVD(rhoNodal.getTensor(),
                                                                1.0e-4);
    int core_dim_x = ttensor->G->getGlobalSize(0);
    int core_dim_y = ttensor->G->getGlobalSize(1);
    int core_dim_z = ttensor->G->getGlobalSize(2);

    PetscPrintf(MPI_COMM_WORLD,
                "Tucker decomposition.\n");

    Tucker::Tensor *core = Tucker::MemoryManager::safe_new<Tucker::Tensor>(ttensor->G->getGlobalSize());
    core->initialize();

    bool squeezed = false;
    const Tucker::SizeArray *offsetX = ttensor->G->getDistribution()->getMap(0,
                                                                             squeezed)->getOffsets();
    const Tucker::SizeArray *offsetY = ttensor->G->getDistribution()->getMap(1,
                                                                             squeezed)->getOffsets();
    const Tucker::SizeArray *offsetZ = ttensor->G->getDistribution()->getMap(2,
                                                                             squeezed)->getOffsets();
    int procGrid[3];
    int xidx0, xidx1, yidx0, yidx1, zidx0, zidx1;
    ttensor->G->getDistribution()->getProcessorGrid()->getCoordinates(procGrid);
    xidx0 = (*offsetX)[procGrid[0]];
    xidx1 = (*offsetX)[procGrid[0] + 1];
    yidx0 = (*offsetY)[procGrid[1]];
    yidx1 = (*offsetY)[procGrid[1] + 1];
    zidx0 = (*offsetZ)[procGrid[2]];
    zidx1 = (*offsetZ)[procGrid[2] + 1];
    int total_core_elements = core->size(0) * core->size(1) * core->size(2);
    std::vector<double> core_data(total_core_elements,
                                  0);
    double *mpi_core_local_data;
    if (ttensor->G->getLocalNumEntries() != 0) {
        mpi_core_local_data = ttensor->G->getLocalTensor()->data();
        for (int k = zidx0; k < zidx1; ++k) {
            for (int j = yidx0; j < yidx1; ++j) {
                for (int i = xidx0; i < xidx1; ++i) {
                    int idx = i + j * core_dim_x + k * core_dim_x * core_dim_y;
                    core_data[idx] = *mpi_core_local_data;
                    mpi_core_local_data++;
                }
            }
        }
    }
    MPI_Allreduce(core_data.data(),
                  core->data(),
                  total_core_elements,
                  MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD);

    PetscPrintf(MPI_COMM_WORLD,
                "rank: %d, %d, %d\n",
                core->size(0),
                core->size(1),
                core->size(2));
    PetscPrintf(MPI_COMM_WORLD,
                "finish all reduce, total number elements %d.\n",
                core->getNumElements());

    Tucker::Matrix U(rhoGrid.getTensor()->getLocalSize(0),
                     core->size(0));
    Tucker::Matrix V(rhoGrid.getTensor()->getLocalSize(1),
                     core->size(1));
    Tucker::Matrix W(rhoGrid.getTensor()->getLocalSize(2),
                     core->size(2));

    Tucker::SizeArray nodalSize(3), rank(3);
    nodalSize[0] = rhoNodal.getGlobalDimension(0);
    nodalSize[1] = rhoNodal.getGlobalDimension(1);
    nodalSize[2] = rhoNodal.getGlobalDimension(2);
    rank[0] = ttensor->G->getGlobalSize(0);
    rank[1] = ttensor->G->getGlobalSize(1);
    rank[2] = ttensor->G->getGlobalSize(2);
    int decompRankX = rank[0];
    int decompRankY = rank[1];
    int decompRankZ = rank[2];

    int numQuadPointsX = femX.getTotalNumberQuadPoints();
    int numQuadPointsY = femY.getTotalNumberQuadPoints();
    int numQuadPointsZ = femZ.getTotalNumberQuadPoints();
    int localQuadPointsX0 = rhoGrid.getIstartGlobal(), localQuadPointsX1 = rhoGrid.getIendGlobal();
    int localQuadPointsY0 = rhoGrid.getJstartGlobal(), localQuadPointsY1 = rhoGrid.getJendGlobal();
    int localQuadPointsZ0 = rhoGrid.getKstartGlobal(), localQuadPointsZ1 = rhoGrid.getKendGlobal();
    int numLocalQuadPointsX = localQuadPointsX1 - localQuadPointsX0;
    int numLocalQuadPointsY = localQuadPointsY1 - localQuadPointsY0;
    int numLocalQuadPointsZ = localQuadPointsZ1 - localQuadPointsZ0;
    int numNodesX = nodalSize[0];
    int numNodesY = nodalSize[1];
    int numNodesZ = nodalSize[2];

    double *nodal_U = ttensor->U[0]->data();
    double *nodal_V = ttensor->U[1]->data();
    double *nodal_W = ttensor->U[2]->data();

    PetscPrintf(MPI_COMM_WORLD,
                "requesting U V W data\n");

    double *quad_U, *quad_V, *quad_W;
    if (numLocalQuadPointsX > 0) {
        quad_U = U.data();
        for (int i = 0; i < decompRankX; ++i) {
            std::vector<double> nodalVal(numNodesX);
            std::vector<double> gridVal(numQuadPointsX);
            std::copy(nodal_U + i * numNodesX,
                      nodal_U + (i + 1) * numNodesX,
                      nodalVal.begin());
            femX.computeFieldAtAllQuadPoints(nodalVal,
                                             gridVal);
            std::copy(gridVal.begin() + localQuadPointsX0,
                      gridVal.begin() + localQuadPointsX1,
                      quad_U + i * numLocalQuadPointsX);
        }
    }
    PetscPrintf(MPI_COMM_WORLD,
                "finish projection X.\n");
    if (numLocalQuadPointsY > 0) {
        quad_V = V.data();
        for (int i = 0; i < decompRankY; ++i) {
            std::vector<double> nodalVal(numNodesY);
            std::vector<double> gridVal(numQuadPointsY);
            std::copy(nodal_V + i * numNodesY,
                      nodal_V + (i + 1) * numNodesY,
                      nodalVal.begin());
            femY.computeFieldAtAllQuadPoints(nodalVal,
                                             gridVal);
            std::copy(gridVal.begin() + localQuadPointsY0,
                      gridVal.begin() + localQuadPointsY1,
                      quad_V + i * numLocalQuadPointsY);
        }
    }
    PetscPrintf(MPI_COMM_WORLD,
                "finish projection Y.\n");
    if (numLocalQuadPointsZ > 0) {
        quad_W = W.data();
        for (int i = 0; i < decompRankZ; ++i) {
            std::vector<double> nodalVal(numNodesZ);
            std::vector<double> gridVal(numQuadPointsZ);
            std::copy(nodal_W + i * numNodesZ,
                      nodal_W + (i + 1) * numNodesZ,
                      nodalVal.begin());
            femZ.computeFieldAtAllQuadPoints(nodalVal,
                                             gridVal);
            std::copy(gridVal.begin() + localQuadPointsZ0,
                      gridVal.begin() + localQuadPointsZ1,
                      quad_W + i * numLocalQuadPointsZ);
        }
    }

    PetscPrintf(MPI_COMM_WORLD,
                "finish projection Z.\n");
    Tucker::MemoryManager::safe_delete(ttensor);

    Tucker::Tensor *temp;
    Tucker::Tensor *reconstructedTensor;
    temp = core;
    reconstructedTensor = Tucker::ttm(temp,
                                      0,
                                      &U);
    temp = reconstructedTensor;
    reconstructedTensor = Tucker::ttm(temp,
                                      1,
                                      &V);
    Tucker::MemoryManager::safe_delete(temp);
    temp = reconstructedTensor;
    reconstructedTensor = Tucker::ttm(temp,
                                      2,
                                      &W);
    Tucker::MemoryManager::safe_delete(temp);

    PetscPrintf(MPI_COMM_WORLD,
                "finiish reconstruction.\n");
    std::copy(reconstructedTensor->data(),
              reconstructedTensor->data() + reconstructedTensor->getNumElements(),
              rhoGrid.getLocalData());

    PetscPrintf(MPI_COMM_WORLD,
                "finish copying.\n");
    Tucker::MemoryManager::safe_delete(reconstructedTensor);

}

namespace {
    void construct3DSplineObject(const std::string &nodalFileX,
                                 const std::string &nodalFileY,
                                 const std::string &nodalFileZ,
                                 const std::string &fieldFile,
                                 alglib::spline3dinterpolant &spline3dinterpolant) {
        std::vector<double> vecx, vecy, vecz, vecfield;
        int taskId;
        MPI_Comm_rank(MPI_COMM_WORLD,
                      &taskId);

        if (taskId == 0) {
            double temp;
            std::ifstream fin;
            fin.open(nodalFileX);
            while (fin >> temp) {
                vecx.push_back(temp);
            }
            fin.close();
            fin.clear();
            fin.open(nodalFileY);
            while (fin >> temp) {
                vecy.push_back(temp);
            }
            fin.close();
            fin.clear();
            fin.open(nodalFileZ);
            while (fin >> temp) {
                vecz.push_back(temp);
            }
            fin.close();
            fin.clear();
            fin.open(fieldFile);
            while (fin >> temp) {
                vecfield.push_back(temp);
            }
            fin.close();
            fin.clear();
        }
        size_t sizes[] = {vecx.size(), vecy.size(), vecz.size(), vecfield.size()};
        MPI_Bcast(sizes,
                  4,
                  MPI_INT,
                  0,
                  MPI_COMM_WORLD);
        assert(sizes[0] * sizes[1] * sizes[2] == sizes[3]);
        if (taskId != 0) {
            vecx = std::vector<double>(sizes[0],
                                       0.0);
            vecy = std::vector<double>(sizes[1],
                                       0.0);
            vecz = std::vector<double>(sizes[2],
                                       0.0);
            vecfield = std::vector<double>(sizes[3],
                                           0.0);
        }
        MPI_Bcast(vecx.data(),
                  sizes[0],
                  MPI_DOUBLE,
                  0,
                  MPI_COMM_WORLD);
        MPI_Bcast(vecy.data(),
                  sizes[1],
                  MPI_DOUBLE,
                  0,
                  MPI_COMM_WORLD);
        MPI_Bcast(vecz.data(),
                  sizes[2],
                  MPI_DOUBLE,
                  0,
                  MPI_COMM_WORLD);
        MPI_Bcast(vecfield.data(),
                  sizes[3],
                  MPI_DOUBLE,
                  0,
                  MPI_COMM_WORLD);

        alglib::real_1d_array x, y, z, field;
        x.setcontent(vecx.size(),
                     vecx.data());
        y.setcontent(vecy.size(),
                     vecy.data());
        z.setcontent(vecz.size(),
                     vecz.data());
        field.setcontent(vecfield.size(),
                         vecfield.data());

        alglib::spline3dbuildtrilinearv(x,
                                        vecx.size(),
                                        y,
                                        vecy.size(),
                                        z,
                                        vecz.size(),
                                        field,
                                        1,
                                        spline3dinterpolant);
    }

    void read_in_factor_matrices(const std::string &filename,
                                 std::vector<alglib::spline1dinterpolant> &spline1dinterpolant) {
        int taskId;
        MPI_Comm_rank(MPI_COMM_WORLD,
                      &taskId);

        std::ifstream fin;
        int rank, num_nodes;
        if (taskId == 0) {
            fin.open(filename);
            std::string str;
            std::getline(fin,
                         str);
            std::istringstream ssin(str);
            ssin >> num_nodes >> rank;
            ssin.clear();
        }
        MPI_Bcast(&rank,
                  1,
                  MPI_INT,
                  0,
                  MPI_COMM_WORLD);
        MPI_Bcast(&num_nodes,
                  1,
                  MPI_INT,
                  0,
                  MPI_COMM_WORLD);

        std::vector<double> nodes_coord(num_nodes,
                                        0.0);
        if (taskId == 0) {
            std::string str;
            std::getline(fin,
                         str);
            std::istringstream ssin(str);
            for (int j = 0; j < num_nodes; ++j) {
                ssin >> nodes_coord[j];
            }
        }
        MPI_Bcast(nodes_coord.data(),
                  num_nodes,
                  MPI_DOUBLE,
                  0,
                  MPI_COMM_WORLD);

        spline1dinterpolant = std::vector<alglib::spline1dinterpolant>(rank);
        for (int i = 0; i < rank; ++i) {
            std::vector<double> data_temp(num_nodes,
                                          0.0);
            if (taskId == 0) {
                std::string str;
                std::getline(fin,
                             str);
                std::istringstream ssin(str);
                for (int j = 0; j < num_nodes; ++j) {
                    ssin >> data_temp[j];
                }
            }
            MPI_Bcast(data_temp.data(),
                      num_nodes,
                      MPI_DOUBLE,
                      0,
                      MPI_COMM_WORLD);
            alglib::real_1d_array x, y;
            x.setcontent(nodes_coord.size(),
                         nodes_coord.data());
            y.setcontent(data_temp.size(),
                         data_temp.data());
            alglib::spline1dbuildcubic(x,
                                       y,
                                       num_nodes,
                                       1,
                                       0.0,
                                       1,
                                       0.0,
                                       spline1dinterpolant[i]);
        }

    }

    void construct_local_factor_matrices(const std::vector<double> &coord,
                                         int rank,
                                         int istart,
                                         int iend,
                                         std::vector<alglib::spline1dinterpolant> &function,
                                         double *data) {
        int cnt = 0;
        for (int r = 0; r < rank; ++r) {
            alglib::spline1dinterpolant &interpolant = function[r];
            for (int i = istart; i < iend; ++i) {
                data[cnt] = alglib::spline1dcalc(interpolant,
                                                 coord[i]);
                cnt++;
            }
        }
    }
}
