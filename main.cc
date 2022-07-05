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

#include "basis/BasisFunctionReader.h"
#include "dft/AndersonMixing.h"
#include "dft/InitialGuess.h"
#include "dft/KSDFTEnergyFunctional.h"
#include "dft/KSDFTPotential.h"
#include "dft/ProjectHamiltonianSparse.h"
#include "dft/SeparableHamiltonian.h"
#include "eigensolver/ChebyShevFilter.h"
#include "eigensolver/EigenSolver.h"
#include "hartree/PoissonHartreePotentialSolver.h"
#include "tensor/TensorUtils.h"
#include "tensor/TuckerTensor.h"
#include "utils/Utils.h"
#include "utils/ttdft_timer.h"
#include "eigensolver/MultTm.h"

#include <algorithm>
#include <boost/filesystem.hpp>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <slepc.h>
#include <vector>

#undef __FUNCT__
#define __FUNCT__ "main"

namespace {
    void
    computeJacobWeight3DMat(const FEM &femX,
                            const FEM &femY,
                            const FEM &femZ,
                            Tensor3DMPI &jacob3DMat,
                            Tensor3DMPI &weight3DMat);

    void
    write_out_rho(const FEM &femX,
                  const FEM &femY,
                  const FEM &femZ,
                  int current_rank_x,
                  int current_rank_y,
                  int current_rank_z,
                  Tensor3DMPI &rho);
} // namespace

int
main(int argc,
     char **args) {
    SlepcInitialize(&argc,
                    &args,
                    (char *) 0,
                    (char *) 0);


    int taskId, nproc, rootId = 0;
    MPI_Comm_rank(MPI_COMM_WORLD,
                  &taskId);
    MPI_Comm_size(MPI_COMM_WORLD,
                  &nproc);


    InputParameter input_parameter(args[1]);
    double Hloc_truncation = std::stod(std::string(args[2]));
    double Cnloc_truncation = std::stod(std::string(args[3]));
    int tensor_proc_dim_x = 1, tensor_proc_dim_y = 1, tensor_proc_dim_z = nproc;
    if (argc > 4) {
        tensor_proc_dim_x = std::stoi(std::string(args[4]));
        tensor_proc_dim_y = std::stoi(std::string(args[5]));
        tensor_proc_dim_z = std::stoi(std::string(args[6]));
    }
    Tensor3DMPI::set_proc(tensor_proc_dim_x,
                          tensor_proc_dim_y,
                          tensor_proc_dim_z);
    int num_wavefunctions_block = 1;
    if (argc > 7) {
        num_wavefunctions_block = std::stod(std::string(args[7]));
    }

    std::string input_file_name = args[1];
    if (taskId == rootId) {
        input_parameter.PrintParameter();
    }

    timer timer1;
    timer scftimer;
    timer totalTimeTimer;

    totalTimeTimer.start();
    int numEleX = input_parameter.numberElementsX;
    int numEleY = input_parameter.numberElementsY;
    int numEleZ = input_parameter.numberElementsZ;
    int totalNumberElementsForNonLocX = input_parameter.numberNonLocalElementsX;
    int totalNumberElementsForNonLocY = input_parameter.numberNonLocalElementsY;
    int totalNumberElementsForNonLocZ = input_parameter.numberNonLocalElementsZ;
    std::string quadRule = input_parameter.quadRule;
    std::string quadRuleElectro = input_parameter.quadRuleElectro;
    std::string meshType = input_parameter.meshType;
    int numNodesPerElement = input_parameter.number_nodes_per_element;
    int innerElements = input_parameter.numberInnerElements;
    double innerDomainSize = input_parameter.innerDomainSize;
    double innerMeshSize = innerDomainSize / innerElements;
    double innerDomainSizeX = innerDomainSize, innerDomainSizeY = innerDomainSize,
            innerDomainSizeZ = innerDomainSize;
    double innerMeshSizeX = innerMeshSize, innerMeshSizeY = innerMeshSize,
            innerMeshSizeZ = innerMeshSize;
    double ratio = input_parameter.coarsingFactor;
    if (meshType == "adaptive") {
        double halfdomain = innerDomainSize / 2.0;
        double outerMeshSize = innerMeshSize;
        for (int i = 0; i < (numEleX - innerElements) / 2; ++i) {
            outerMeshSize *= ratio;
            halfdomain += outerMeshSize;
        }
        input_parameter.domainStart = -halfdomain;
        input_parameter.domainEnd = halfdomain;
        input_parameter.domainElectroStart = -halfdomain;
        input_parameter.domainElectroEnd = halfdomain;
    } else if (meshType == "manual1") {
        double halfdomain = innerDomainSize / 2.0;
        double outerMeshSize = 1.5;
        for (int i = 0; i < (numEleX - innerElements) / 2; ++i) {
            outerMeshSize *= ratio;
            halfdomain += outerMeshSize;
        }
        input_parameter.domainStart = -halfdomain;
        input_parameter.domainEnd = halfdomain;
        input_parameter.domainElectroStart = -halfdomain;
        input_parameter.domainElectroEnd = halfdomain;
    }
    double domainStart = input_parameter.domainStart;
    double domainEnd = input_parameter.domainEnd;
    double domainStartElectro = input_parameter.domainElectroStart;
    double domainEndElectro = input_parameter.domainElectroEnd;

    double coarsingFactor = input_parameter.coarsingFactor;
    bool electroFlag = 0;
    Utils::print_current_max_mem("memory usage after initializing parameters");

    AtomInformation atomInformation(input_parameter.system_name);
    // !!!!!!!!!!!!!CHECK!!!!!!!!
    std::vector<std::vector<double>> nuclei = atomInformation.all_nuclei;
    // std::vector<std::vector<double> > nonLocalNuclei =
    // atomInformation.all_nuclei; int numNonLocalAtoms =
    // atomInformation.all_nuclei.size();
    int numAtoms = atomInformation.numAtoms;
    // int numAtomTypes = atomInformation.numAtomType;
    int numberEigenValues = input_parameter.numEigenValues;
    int numberElectrons = atomInformation.numElectrons;

    Utils::print_current_max_mem(
            "memory usage after initializing atom information");

    double boltzmannConstant = 3.166811429e-06;
    double temperature = input_parameter.smearingTemerature;

    Tucker::SizeArray rankRho(3);
    rankRho[0] = input_parameter.rhoRankX;
    rankRho[1] = input_parameter.rhoRankY;
    rankRho[2] = input_parameter.rhoRankZ;

    Tucker::SizeArray rankVeff(3);
    rankVeff[0] = input_parameter.veffRankX;
    rankVeff[1] = input_parameter.veffRankY;
    rankVeff[2] = input_parameter.veffRankZ;

    int rankNloc = input_parameter.rankNloc;
    int tuckerRankX, tuckerRankY, tuckerRankZ;

    // bool usingKernelExpansion = input_parameter.usingKernelExpansion;
    bool chebFlag = input_parameter.chebFlag;
    double first_cheb_tolerance = 1.0e-3;
    int maxLanczosIter = input_parameter.numLanczosIter;
    int polynomialDegree = input_parameter.polynomialDegree;

    int maxSCFIter = input_parameter.maxScfIter;
    double alpha = input_parameter.alpha; // used by anderson mixing scheme
    int numberIterHistory =
            input_parameter.numberHistory; // number of steps to be mixed
    double scfTol = input_parameter.scfTol;
    double langMult = 0.551; // used in solving separable hamiltonian

    int linear_nodes_per_element = 2;
    FEM femX(numEleX,
             quadRule,
             numNodesPerElement,
             domainStart,
             domainEnd,
             innerDomainSizeX,
             innerMeshSizeX,
             meshType,
             electroFlag,
             coarsingFactor);
    FEM femLinearX(numEleX,
                   quadRule,
                   linear_nodes_per_element,
                   domainStart,
                   domainEnd,
                   innerDomainSizeX,
                   innerMeshSizeX,
                   meshType,
                   electroFlag,
                   coarsingFactor);
    FEM femY(numEleY,
             quadRule,
             numNodesPerElement,
             domainStart,
             domainEnd,
             innerDomainSizeY,
             innerMeshSizeY,
             meshType,
             electroFlag,
             coarsingFactor);
    FEM femLinearY(numEleY,
                   quadRule,
                   linear_nodes_per_element,
                   domainStart,
                   domainEnd,
                   innerDomainSizeY,
                   innerMeshSizeY,
                   meshType,
                   electroFlag,
                   coarsingFactor);
    FEM femZ(numEleZ,
             quadRule,
             numNodesPerElement,
             domainStart,
             domainEnd,
             innerDomainSizeZ,
             innerMeshSizeZ,
             meshType,
             electroFlag,
             coarsingFactor);
    FEM femLinearZ(numEleZ,
                   quadRule,
                   linear_nodes_per_element,
                   domainStart,
                   domainEnd,
                   innerDomainSizeZ,
                   innerMeshSizeZ,
                   meshType,
                   electroFlag,
                   coarsingFactor);


    FEM femElectroX(numEleX,
                    quadRuleElectro,
                    numNodesPerElement,
                    domainStartElectro,
                    domainEndElectro,
                    innerDomainSizeX,
                    innerMeshSizeX,
                    meshType,
                    electroFlag,
                    coarsingFactor);
    FEM femElectroY(numEleY,
                    quadRuleElectro,
                    numNodesPerElement,
                    domainStartElectro,
                    domainEndElectro,
                    innerDomainSizeY,
                    innerMeshSizeY,
                    meshType,
                    electroFlag,
                    coarsingFactor);
    FEM femElectroZ(numEleZ,
                    quadRuleElectro,
                    numNodesPerElement,
                    domainStartElectro,
                    domainEndElectro,
                    innerDomainSizeZ,
                    innerMeshSizeZ,
                    meshType,
                    electroFlag,
                    coarsingFactor);

    PetscPrintf(PETSC_COMM_WORLD,
                "linear nodes:\n(%.2f, ",
                femLinearX.getGlobalNodalCoord().front());
    for (int i = 1; i < femLinearX.getGlobalNodalCoord().size() - 1; ++i) {
        PetscPrintf(PETSC_COMM_WORLD,
                    "%.2f, ",
                    femLinearX.getGlobalNodalCoord()[i]);
    }
    PetscPrintf(PETSC_COMM_WORLD,
                "%.2f)\n",
                femLinearX.getGlobalNodalCoord().back());

    PetscPrintf(PETSC_COMM_WORLD,
                "global nodes:\n(%.2f, ",
                femX.getGlobalNodalCoord().front());
    for (int i = 1; i < femX.getGlobalNodalCoord().size() - 1; ++i) {
        PetscPrintf(PETSC_COMM_WORLD,
                    "%.2f, ",
                    femX.getGlobalNodalCoord()[i]);
    }
    PetscPrintf(PETSC_COMM_WORLD,
                "%.2f)\n",
                femX.getGlobalNodalCoord().back());

    Utils::print_current_max_mem("memory usage after initializing FEM objects");


    // Get nodal coordinates of 1-D FEM for physical grids & refined electro grids

    std::vector<double> globalCoordX = femX.getGlobalNodalCoord();
    std::vector<double> globalCoordY = femY.getGlobalNodalCoord();
    std::vector<double> globalCoordZ = femZ.getGlobalNodalCoord();

    std::vector<double> globalCoordElectroX = femElectroX.getGlobalNodalCoord();
    std::vector<double> globalCoordElectroY = femElectroY.getGlobalNodalCoord();
    std::vector<double> globalCoordElectroZ = femElectroZ.getGlobalNodalCoord();

    // Get quadrature points coordinates of 1-D FEM for physical grids & refined
    // electro grids
    std::vector<double> quadCoordX = femX.getPositionQuadPointValues();
    std::vector<double> quadCoordY = femY.getPositionQuadPointValues();
    std::vector<double> quadCoordZ = femZ.getPositionQuadPointValues();

    std::vector<double> quadCoordElectroX =
            femElectroX.getPositionQuadPointValues();
    std::vector<double> quadCoordElectroY =
            femElectroY.getPositionQuadPointValues();
    std::vector<double> quadCoordElectroZ =
            femElectroZ.getPositionQuadPointValues();

    Utils::print_current_max_mem(
            "memory usage after initializing nodal positions");

    int numberNodesX = femX.getTotalNumberNodes();
    int numberNodesY = femY.getTotalNumberNodes();
    int numberNodesZ = femZ.getTotalNumberNodes();
    int numberQuadPointsX = femX.getTotalNumberQuadPoints();
    int numberQuadPointsY = femY.getTotalNumberQuadPoints();
    int numberQuadPointsZ = femZ.getTotalNumberQuadPoints();

    double radiusDeltaVlx = input_parameter.nonloca_radius_delta_x;
    double domainStartNonLocX = -radiusDeltaVlx;
    double domainEndNonLocX = radiusDeltaVlx;
    double sizeOfDomainX = domainEndNonLocX - domainStartNonLocX;
    double nonLoclInnerMeshSizeX = sizeOfDomainX / totalNumberElementsForNonLocX;

    double radiusDeltaVly = input_parameter.nonloca_radius_delta_y;
    double domainStartNonLocY = -radiusDeltaVly;
    double domainEndNonLocY = radiusDeltaVly;
    double sizeOfDomainY = domainEndNonLocY - domainStartNonLocY;
    double nonLoclInnerMeshSizeY = sizeOfDomainY / totalNumberElementsForNonLocY;

    double radiusDeltaVlz = input_parameter.nonloca_radius_delta_z;
    double domainStartNonLocZ = -radiusDeltaVlz;
    double domainEndNonLocZ = radiusDeltaVlz;
    double sizeOfDomainZ = domainEndNonLocZ - domainStartNonLocZ;
    double nonLoclInnerMeshSizeZ = sizeOfDomainZ / totalNumberElementsForNonLocZ;


    FEM femNonLocX(totalNumberElementsForNonLocX,
                   quadRule,
                   numNodesPerElement,
                   domainStartNonLocX,
                   domainEndNonLocX,
                   sizeOfDomainX,
                   nonLoclInnerMeshSizeX,
                   "uniform",
                   1,
                   coarsingFactor);
    FEM femNonLocLinearX(totalNumberElementsForNonLocX,
                         quadRule,
                         linear_nodes_per_element,
                         domainStartNonLocX,
                         domainEndNonLocX,
                         sizeOfDomainX,
                         nonLoclInnerMeshSizeX,
                         "uniform",
                         1,
                         coarsingFactor);
    FEM femNonLocY(totalNumberElementsForNonLocY,
                   quadRule,
                   numNodesPerElement,
                   domainStartNonLocY,
                   domainEndNonLocY,
                   sizeOfDomainY,
                   nonLoclInnerMeshSizeY,
                   "uniform",
                   1,
                   coarsingFactor);
    FEM femNonLocLinearY(totalNumberElementsForNonLocY,
                         quadRule,
                         linear_nodes_per_element,
                         domainStartNonLocY,
                         domainEndNonLocY,
                         sizeOfDomainY,
                         nonLoclInnerMeshSizeY,
                         "uniform",
                         1,
                         coarsingFactor);
    FEM femNonLocZ(totalNumberElementsForNonLocZ,
                   quadRule,
                   numNodesPerElement,
                   domainStartNonLocZ,
                   domainEndNonLocZ,
                   sizeOfDomainZ,
                   nonLoclInnerMeshSizeZ,
                   "uniform",
                   1,
                   coarsingFactor);
    FEM femNonLocLinearZ(totalNumberElementsForNonLocZ,
                         quadRule,
                         linear_nodes_per_element,
                         domainStartNonLocZ,
                         domainEndNonLocZ,
                         sizeOfDomainZ,
                         nonLoclInnerMeshSizeZ,
                         "uniform",
                         1,
                         coarsingFactor);
    Utils::print_current_max_mem(
            "memory usage after initializing atoms FEM objects");

    MultTM multtm;

    int numRanks = input_parameter.tuckerRankX.size();
    std::shared_ptr<PoissonHartreePotentialSolver> poissonHartreePotentialSolver;
    if (input_parameter.which_using_kernel_expansion.size() < numRanks) {
        poissonHartreePotentialSolver =
                std::shared_ptr<PoissonHartreePotentialSolver>(
                        new PoissonHartreePotentialSolver(
                                femX,
                                femY,
                                femZ,
                                femElectroX,
                                femElectroY,
                                femElectroZ,
                                PETScLinearSolver::Solver::CG,
                                PETScLinearSolver::Preconditioner::BJACOBI,
                                nuclei,
                                input_parameter.rhoRankX,
                                input_parameter.rhoRankY,
                                input_parameter.rhoRankZ,
                                input_parameter.poisson_alphafile,
                                input_parameter.poisson_omegafile,
                                input_parameter.poisson_Asquare,
                                input_parameter.hartree_domain_start_x,
                                input_parameter.hartree_domain_end_x,
                                input_parameter.hartree_domain_start_y,
                                input_parameter.hartree_domain_end_y,
                                input_parameter.hartree_domain_start_z,
                                input_parameter.hartree_domain_end_z,
                                input_parameter.hartree_domain_coarsing_factor,
                                input_parameter.hartree_domain_num_additional_elements));
        Utils::print_current_max_mem(
                "memory usage after initializing hartree objects");
    }

    std::vector<std::vector<double>> basis_x, basis_y, basis_z;
    BasisFunctionReader::copy_basis_function_from_file("basis_x.txt",
                                                       basis_x);
    BasisFunctionReader::copy_basis_function_from_file("basis_y.txt",
                                                       basis_y);
    BasisFunctionReader::copy_basis_function_from_file("basis_z.txt",
                                                       basis_z);
    tuckerRankX = basis_x.size();
    tuckerRankY = basis_y.size();
    tuckerRankZ = basis_z.size();


    timer1.start();
    ProjectHamiltonianSparse project_hamiltonian_sparse(femX,
                                                        femY,
                                                        femZ,
                                                        femLinearX,
                                                        femLinearY,
                                                        femLinearZ,
                                                        basis_x,
                                                        basis_y,
                                                        basis_z);
    Tensor3DMPI potLoc(numberQuadPointsX,
                       numberQuadPointsY,
                       numberQuadPointsZ,
                       MPI_COMM_WORLD);
    {
        std::vector<std::shared_ptr<NonLocalPSPData>> nonLocalPSPData(
                atomInformation.numAtomType);
        for (int i = 0; i < atomInformation.numAtomType; ++i) {
            std::string local_potential_filename =
                    std::string("locPotential_AT") + std::to_string(i);
            std::string nonlocal_psp_filename =
                    std::string("nlpV_AT") + std::to_string(i);
            std::string nonlocal_pswfn_filename =
                    std::string("nlpWaveFun_AT") + std::to_string(i);
            nonLocalPSPData[i] =
                    std::make_shared<NonLocalPSPData>(femNonLocX,
                                                      femNonLocY,
                                                      femNonLocZ,
                                                      atomInformation.nuclei[i].size(),
                                                      atomInformation.lMax[i],
                                                      rankNloc,
                                                      local_potential_filename,
                                                      nonlocal_psp_filename,
                                                      nonlocal_pswfn_filename);
        }
        timer1.end("computing nonlocal PSP Data");
        Utils::print_current_max_mem(
                "memory usage after initializing nonlocal psp data");
        timer1.start();
        project_hamiltonian_sparse.Create_Cnloc(femNonLocX,
                                                femNonLocY,
                                                femNonLocZ,
                                                femNonLocLinearX,
                                                femNonLocLinearY,
                                                femNonLocLinearZ,
                                                tuckerRankX,
                                                tuckerRankY,
                                                tuckerRankZ,
                                                nonLocalPSPData,
                                                atomInformation,
                                                Cnloc_truncation);
        timer1.end("constructing Cnloc");
        Utils::print_current_max_mem(
                "memory usage after initializing atoms manager objects");

        std::string potloc_path = "restart/potloc";
        std::string potloc_filename =
                potloc_path + "/potloc_proc" + std::to_string(taskId);
        if (input_parameter.is_calculation_restart == false) {
            timer1.start();
            for (int i = 0; i < atomInformation.numAtomType; ++i) {
                std::string local_psp_filename =
                        std::string("locPotential_AT") + std::to_string(i);
                nonLocalPSPData[i]->computeLocalPart(femX,
                                                     femY,
                                                     femZ,
                                                     atomInformation.nuclei[i],
                                                     local_psp_filename,
                                                     potLoc);
            }
            timer1.end("computing local part");
            timer1.start();
            boost::filesystem::create_directories(potloc_path);
            int potloc_fd;
            PetscBinaryOpen(potloc_filename.c_str(),
                            FILE_MODE_WRITE,
                            &potloc_fd);
            PetscBinaryWrite(potloc_fd,
                             potLoc.getLocalData(),
                             potLoc.getLocalNumberEntries(),
                             PETSC_DOUBLE,
                             PETSC_FALSE);
            PetscBinaryClose(potloc_fd);
            timer1.end("write out local part");
        } else {
            if (taskId == 0)
                std::cout << "starting to read in local part psp.\n";
            timer1.start();
            int potloc_fd;
            PetscBinaryOpen(potloc_filename.c_str(),
                            FILE_MODE_READ,
                            &potloc_fd);
#if PETSC_VERSION_GE(3, 12, 0)
            PetscBinaryRead(potloc_fd,
                            potLoc.getLocalData(),
                            potLoc.getLocalNumberEntries(),
                            PETSC_NULL,
                            PETSC_REAL);
#else
            PetscBinaryRead(potloc_fd,
                            potLoc.getLocalData(),
                            potLoc.getLocalNumberEntries(),
                            PETSC_DOUBLE);
#endif
            PetscBinaryClose(potloc_fd);
            timer1.end("read in local part");
        }
        Utils::print_current_max_mem(
                "memory usage after initializing potloc objects");
    }


    KSDFTPotential ksdftPotential(femX,
                                  femY,
                                  femZ,
                                  femElectroX,
                                  femElectroY,
                                  femElectroZ,
                                  input_parameter.alphafile,
                                  input_parameter.omegafile,
                                  input_parameter.Asquare);
    Utils::print_current_max_mem(
            "memory usage after initializing ksdft potential");


    Tensor3DMPI rhoNodalIn(numberNodesX,
                           numberNodesY,
                           numberNodesZ,
                           MPI_COMM_WORLD);
    Tensor3DMPI rhoGridIn(numberQuadPointsX,
                          numberQuadPointsY,
                          numberQuadPointsZ,
                          MPI_COMM_WORLD);
    InitialGuess initialGuess;
    timer1.start();
    if (input_parameter.is_calculation_restart == true) {
        // Anderson mixing will take over the initialization
        if (taskId == 0)
            std::cout << "rho values will be read in Anderson mixing.\n";
    } else {
        initialGuess.initialize_rho(femX,
                                    femY,
                                    femZ,
                                    input_parameter,
                                    atomInformation,
                                    rhoNodalIn,
                                    rhoGridIn);
    }
    Utils::print_current_max_mem("memory usage after initializing rho in");
    timer1.end("computing initial values of rhoNodaIn and rhoGridIn");

    Tensor3DMPI rhoNodalOut(numberNodesX,
                            numberNodesY,
                            numberNodesZ,
                            MPI_COMM_WORLD);
    Tensor3DMPI rhoGridOut(numberQuadPointsX,
                           numberQuadPointsY,
                           numberQuadPointsZ,
                           MPI_COMM_WORLD);
    Utils::print_current_max_mem("memory usage after initializing rho out");

    Tensor3DMPI jacob3DMat(numberQuadPointsX,
                           numberQuadPointsY,
                           numberQuadPointsZ,
                           MPI_COMM_WORLD);
    Tensor3DMPI weight3DMat(numberQuadPointsX,
                            numberQuadPointsY,
                            numberQuadPointsZ,
                            MPI_COMM_WORLD);
    computeJacobWeight3DMat(femX,
                            femY,
                            femZ,
                            jacob3DMat,
                            weight3DMat);
    Utils::print_current_max_mem("memory usage after initializing jacob");

    Tensor3DMPI groundStateEnergyGrid(numberQuadPointsX,
                                      numberQuadPointsZ,
                                      numberQuadPointsZ,
                                      MPI_COMM_WORLD);
    Utils::print_current_max_mem("memory usage after initializing all grids");

    double groundStateEnergyCurrent, groundStateEnergyOld;

    double relErrorEnergy = 1.0e3;
    KSDFTEnergyFunctional ksdftEnergyFunctional;
    EigenSolver eigenSolver;

    double scaleFactor =
            ksdftEnergyFunctional.compute3DIntegral(rhoGridIn,
                                                    jacob3DMat,
                                                    weight3DMat);
    scaleFactor = numberElectrons / scaleFactor;
    rhoGridIn *= scaleFactor;
    rhoNodalIn *= scaleFactor;

    Utils::print_current_max_mem("memory usage after initializing Psi");


    double occupied_orbital_energy = 1.0e9;
    int occupied_orbital_number = int(numberElectrons / 2.0 + 10);
    if (occupied_orbital_number > numberEigenValues - 1)
        occupied_orbital_number = numberEigenValues - 1;

    // create folder for storing restart files
    std::string restart_path = "restart/rank_" + std::to_string(tuckerRankX) +
                               "_" + std::to_string(tuckerRankY) + "_" +
                               std::to_string(tuckerRankZ);
    std::string restart_path_rho = restart_path + "/rho";
    std::string restart_path_wfn = restart_path + "/wfn";
    boost::filesystem::create_directories(restart_path_rho);
    boost::filesystem::create_directories(restart_path_wfn);
    AndersonMixing andersonMixing(femX,
                                  femY,
                                  femZ,
                                  jacob3DMat,
                                  weight3DMat,
                                  restart_path_rho,
                                  alpha,
                                  numberIterHistory,
                                  input_parameter.is_calculation_restart,
                                  false);
    Utils::print_current_max_mem("memory usage after initializing anderson");

    bool is_using_kernel_expansion =
            std::find(input_parameter.which_using_kernel_expansion.begin(),
                      input_parameter.which_using_kernel_expansion.end(),
                      0) != input_parameter.which_using_kernel_expansion.end();

    if (is_using_kernel_expansion == false) {
        poissonHartreePotentialSolver->turn_on_initialize_hartree();
    }

    numberEigenValues = input_parameter.numEigenValues;
    if (numberEigenValues > tuckerRankX * tuckerRankY * tuckerRankZ) {
        numberEigenValues = tuckerRankX * tuckerRankY * tuckerRankZ;
        PetscPrintf(
                PETSC_COMM_WORLD,
                "number of eigenvalues in this iteration is shrunk to be: %d\n",
                numberEigenValues);
    }



    std::vector<double> eig_vals(numberEigenValues,
                                 0.0);
    relErrorEnergy = 1.0e3;
    double upperBoundUnWantedSpectrum = 1e16, error_in_upper_bound = 1e16;
    double lowerBoundWantedSpectrum = 0, lowerBoundUnWantedSpectrum = -1e16;
    int ks_scf_converge_count = 0;

    Utils::print_current_max_mem("memory usage before start");
    long int basis3d_size = tuckerRankX * tuckerRankY * tuckerRankZ;
    Mat eig_vecs;
    MatCreateDense(PETSC_COMM_WORLD,
                   PETSC_DECIDE,
                   PETSC_DECIDE,
                   basis3d_size,
                   numberEigenValues,
                   PETSC_NULL,
                   &eig_vecs);
    PetscInt wfn_local_rows, wfn_local_cols;
    MatGetLocalSize(eig_vecs,
                    &wfn_local_rows,
                    &wfn_local_cols);
    PetscInt wfn_local_elements = wfn_local_rows * numberEigenValues;
    std::string wfn_filenmae =
            restart_path_wfn + "/wfn_proc" + std::to_string(taskId);
    if (input_parameter.is_calculation_restart == true) {
        std::ifstream fin(restart_path_wfn + "/cheby_bounds.txt");
        fin >> upperBoundUnWantedSpectrum >> lowerBoundUnWantedSpectrum >>
            lowerBoundWantedSpectrum >> error_in_upper_bound >>
            occupied_orbital_energy >> groundStateEnergyOld;
        fin.close();

        PetscPrintf(MPI_COMM_WORLD,
                    "read in wavefunctions.\n");
        PetscPrintf(MPI_COMM_WORLD,
                    "read in upperBoundUnWantedSpectrum: %.8e\n",
                    upperBoundUnWantedSpectrum);
        PetscPrintf(MPI_COMM_WORLD,
                    "read in lowerBoundUnWantedSpectrum: %.8e\n",
                    lowerBoundUnWantedSpectrum);
        PetscPrintf(MPI_COMM_WORLD,
                    "read in lowerBoundWantedSpectrum: %.8e\n",
                    lowerBoundWantedSpectrum);
        PetscPrintf(MPI_COMM_WORLD,
                    "read in error_in_upper_bound: %.8e\n",
                    error_in_upper_bound);
        PetscPrintf(MPI_COMM_WORLD,
                    "read in occupied_orbital_energy: %.8e\n",
                    occupied_orbital_energy);

        int fd;
        double *wfn_data;
        timer1.start();
        MatDenseGetArray(eig_vecs,
                         &wfn_data);
        PetscBinaryOpen(wfn_filenmae.c_str(),
                        FILE_MODE_READ,
                        &fd);
#if PETSC_VERSION_GE(3, 12, 0)
        PetscBinaryRead(
          fd, wfn_data, wfn_local_elements, PETSC_NULL, PETSC_DOUBLE);
#else
        PetscBinaryRead(fd,
                        wfn_data,
                        wfn_local_elements,
                        PETSC_DOUBLE);
#endif
        MatDenseRestoreArray(eig_vecs,
                             &wfn_data);
        MatAssemblyBegin(eig_vecs,
                         MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(eig_vecs,
                       MAT_FINAL_ASSEMBLY);
        timer1.end("reading in wavefunctions.");
    } else {
        Utils::print_current_max_mem(
                "memory usage before initial wavefunction orthogonalization");

        timer1.start();
        MatSetRandom(eig_vecs,
                     PETSC_NULL);
        timer1.end("randomizing the matrix X");

        timer1.start();
        BV bv;
        BVCreateFromMat(eig_vecs,
                        &bv);
        BVSetType(bv,
                  BVMAT);
        BVOrthogonalize(bv,
                        PETSC_NULL);
        Mat orthogonal_basis;
        BVGetMat(bv,
                 &orthogonal_basis);
        MatCopy(orthogonal_basis,
                eig_vecs,
                SAME_NONZERO_PATTERN);
        BVRestoreMat(bv,
                     &orthogonal_basis);
        BVDestroy(&bv);
        timer1.end("orthogonalizing the matrix X befor start");
    }
    Utils::print_current_max_mem(
            "memory usage after wavefunction initialization");

    PetscPrintf(PETSC_COMM_WORLD,
                "************************************\n");
    PetscPrintf(PETSC_COMM_WORLD,
                "       Entering Rank %d\n",
                tuckerRankX);
    PetscPrintf(PETSC_COMM_WORLD,
                "************************************\n");

    int scfIter = 0;
    if (input_parameter.is_calculation_restart != 0) {
        scfIter = input_parameter.start_from_scf_iter;
    }
    for (; scfIter < maxSCFIter; ++scfIter) {
        scftimer.start();
        if (taskId == rootId) {
            std::cout << "====================================" << std::endl;
            std::cout << "       Entering SCF " << scfIter << std::endl;
            std::cout << "====================================" << std::endl;
        }

        double chargeIn = ksdftEnergyFunctional.compute3DIntegral(rhoGridIn,
                                                                  jacob3DMat,
                                                                  weight3DMat);
#ifndef NDEBUG
        PetscPrintf(PETSC_COMM_WORLD, "charge in: %e\n", chargeIn);
#endif

        timer1.start();
        andersonMixing.computeRhoIn(scfIter,
                                    rhoNodalIn,
                                    rhoGridIn);
        timer1.end("Anderson mixing");
        Utils::print_current_max_mem("memory usage after anderson mixing");

        chargeIn = ksdftEnergyFunctional.compute3DIntegral(rhoGridIn,
                                                           jacob3DMat,
                                                           weight3DMat);
#ifndef NDEBUG
        PetscPrintf(PETSC_COMM_WORLD, "charge in after mixing: %e\n", chargeIn);
#endif

        Tensor3DMPI potEffIn(numberQuadPointsX,
                             numberQuadPointsY,
                             numberQuadPointsZ,
                             PETSC_COMM_WORLD);

        std::string hartreeMethod;
        timer1.start();
        if (is_using_kernel_expansion == true || ks_scf_converge_count == 0) {
            timer decomp_timer;
            decomp_timer.start();
            const TuckerMPI::TuckerTensor *decomposedRhoGrid =
                    TensorUtils::computeSTHOSVDonQuadMPI(femElectroX,
                                                         femElectroY,
                                                         femElectroZ,
                                                         input_parameter.rhoRankX,
                                                         input_parameter.rhoRankY,
                                                         input_parameter.rhoRankZ,
                                                         rhoNodalIn);
            decomp_timer.end("decoposition in kernel expansion");
            ksdftPotential.computeHartreePotNew(decomposedRhoGrid,
                                                potEffIn,
                                                TENSOR_INSERT);

            Tucker::MemoryManager::safe_delete(decomposedRhoGrid);
            hartreeMethod = "kernel expansion";
        } else {
            poissonHartreePotentialSolver->solve(input_parameter.maxHartreeIter,
                                                 input_parameter.hartreeTolerance,
                                                 rhoNodalIn,
                                                 potEffIn);
            hartreeMethod = "poisson solver";
            ks_scf_converge_count++;
        }

        timer1.end("computing Hartree potential using " + hartreeMethod);
        Utils::print_current_max_mem("memory usage after Hartree potential");

        if (input_parameter.is_break_apart_energies == true) {
            timer1.start();
            Tensor3DMPI hartree_energy_grid(numberQuadPointsX,
                                            numberQuadPointsY,
                                            numberQuadPointsZ,
                                            PETSC_COMM_WORLD);

            // 0.5 int (rho*pot_hartree)
            ksdftEnergyFunctional.computeHartreeEnergy(rhoGridIn,
                                                       potEffIn,
                                                       hartree_energy_grid,
                                                       TENSOR_INSERT);

            double hartree_energy_in =
                    ksdftEnergyFunctional.compute3DIntegral(hartree_energy_grid,
                                                            jacob3DMat,
                                                            weight3DMat);
            timer1.end("computing Hartree energy in");

            Utils::print_current_max_mem(
                    "memory usage after break-apart energy computation");
        }

        timer1.start();
        ksdftPotential.computeLDACorrelationPot(rhoGridIn,
                                                potEffIn,
                                                TENSOR_ADD);
        ksdftPotential.computeLDAExchangePot(rhoGridIn,
                                             potEffIn,
                                             TENSOR_ADD);
        timer1.end("computing XC potential");
        Utils::print_current_max_mem("memory usage after XC potential");

        timer1.start();
        potEffIn += potLoc;
        timer1.end("adding potentials");
        Utils::print_current_max_mem("memory usage after adding potLoc");

        timer1.start();
        const TuckerMPI::TuckerTensor *decomposedPotEff =
                TuckerMPI::STHOSVD(potEffIn.tensor,
                                   &rankVeff,
                                   true,
                                   false);
        Utils::print_current_max_mem(
                "memory usage after Tucker decomposition of rho");

        timer1.end("reading in basis functions.");
        Utils::print_current_max_mem("memory usage after 1d basis construction");

        timer1.start();
        Utils::print_current_max_mem(
                "memory usage before initialization of atoms hamiltonian");
        project_hamiltonian_sparse.Create_Hloc(rankVeff[0],
                                               rankVeff[1],
                                               rankVeff[2],
                                               tuckerRankX,
                                               tuckerRankY,
                                               tuckerRankZ,
                                               decomposedPotEff,
                                               Hloc_truncation);
        timer1.end("initializing tucker hamiltonian");
        Utils::print_current_max_mem(
                "memory usage  after initialization of hamiltonian");

        timer1.start();
        int restart_times;
        if (scfIter == 0) {
            restart_times = input_parameter.chebyshev_restart_times_first;
        } else {
            restart_times = input_parameter.chebyshev_restart_times_other;
        }
        //      double lowerBoundWantedSpectrum, lowerBoundUnWantedSpectrum;
        double temp_lower_bound;

        if (error_in_upper_bound > 0.01) {
            double current_spectrum;
            eigenSolver.computeUpperBoundWithLanczos(project_hamiltonian_sparse,
                                                     maxLanczosIter,
                                                     current_spectrum,
                                                     temp_lower_bound);
            error_in_upper_bound =
                    (upperBoundUnWantedSpectrum - current_spectrum) /
                    upperBoundUnWantedSpectrum;
            if (error_in_upper_bound > 0.01) {
                upperBoundUnWantedSpectrum = current_spectrum;
            } else {
                upperBoundUnWantedSpectrum = current_spectrum * 1.05;
            }
        } else {
            // do nothing
        }
        timer1.end("Lanczos algorithm");
        Utils::print_current_max_mem("memory usage after Lanczos");

        if (scfIter == 0) {
            lowerBoundWantedSpectrum = temp_lower_bound;
            lowerBoundUnWantedSpectrum = 0.0; // 0.5 * lowerBoundWantedSpectrum +
            // 0.5 * upperBoundUnWantedSpectrum;
        } else if (scfIter != input_parameter.start_from_scf_iter) {
            lowerBoundWantedSpectrum = eig_vals[0];
            lowerBoundUnWantedSpectrum = eig_vals[numberEigenValues - 1];
        }

        timer ChfTime;
        ChfTime.start();
        ChebyShevFilter chebyshevFilter;
        Mat Q;
        MatCreateSeqDense(
                PETSC_COMM_SELF,
                numberEigenValues,
                numberEigenValues,
                PETSC_NULL,
                &Q);

        double upper, lower;
        eigenSolver.computeUpperBoundWithLanczos(project_hamiltonian_sparse,
                                                 maxLanczosIter,
                                                 upper,
                                                 lower);
        AX ax(1,
              numberEigenValues);
        ax.setup_system(project_hamiltonian_sparse.H_loc);
        for (int restart_iter = 0; restart_iter < restart_times; ++restart_iter) {
            PetscPrintf(PETSC_COMM_WORLD,
                        "upperBoundUnWantedSpectrum is %.18e.\n",
                        upperBoundUnWantedSpectrum);
            PetscPrintf(PETSC_COMM_WORLD,
                        "lowerBoundUnWantedSpectrum is %.18e.\n",
                        lowerBoundUnWantedSpectrum);
            PetscPrintf(PETSC_COMM_WORLD,
                        "lowerBoundWantedSpectrum is %.18e.\n",
                        lowerBoundWantedSpectrum);

            timer1.start();
            chebyshevFilter.computeFilteredSubspace(project_hamiltonian_sparse,
                                                    ax,
                                                    eig_vecs,
                                                    basis3d_size,
                                                    polynomialDegree,
                                                    lowerBoundUnWantedSpectrum,
                                                    upperBoundUnWantedSpectrum,
                                                    lowerBoundWantedSpectrum,
                                                    num_wavefunctions_block);
            timer1.end("Chebyshev filtering.");

            timer1.start();

            multtm.orth(eig_vecs);

            // projecting Hamiltonian onto orthogonalized filtered Chebyshev
            // subspace
            timer1.start();
            Mat PtHP, HPloc, HPnl, HP;
            MatCreateDense(PETSC_COMM_WORLD,
                           PETSC_DECIDE,
                           PETSC_DECIDE,
                           basis3d_size,
                           numberEigenValues,
                           PETSC_NULL,
                           &HPloc);
            MatZeroEntries(HPloc);
            MatAssemblyBegin(HPloc,
                             MAT_FINAL_ASSEMBLY);
            MatAssemblyEnd(HPloc,
                           MAT_FINAL_ASSEMBLY);
            ax.perform_ax(eig_vecs, HPloc);
            MatMatMult(project_hamiltonian_sparse.C_nloc,
                       eig_vecs,
                       MAT_INITIAL_MATRIX,
                       PETSC_DEFAULT,
                       &HPnl);
            MatMatMult(project_hamiltonian_sparse.C_nloc_trans,
                       HPnl,
                       MAT_INITIAL_MATRIX,
                       PETSC_DEFAULT,
                       &HP);
            MatAXPY(HP,
                    1.0,
                    HPloc,
                    SAME_NONZERO_PATTERN);

            std::vector<double> PtHP_temp;
            multtm.mult(eig_vecs, HP, PtHP_temp);
            MatDestroy(&HPloc);
            MatDestroy(&HPnl);
            MatDestroy(&HP);
            timer1.end("compute PtHP");
            Utils::print_current_max_mem("memory usage after Pt*H*P");

            timer1.start();
            MatCreateSeqDense(MPI_COMM_SELF, numberEigenValues, numberEigenValues, NULL, &PtHP);
            double *PtHP_ptr;
            MatDenseGetArray(PtHP, &PtHP_ptr);
            std::copy(PtHP_temp.begin(), PtHP_temp.end(), PtHP_ptr);
            MatDenseRestoreArray(PtHP, &PtHP_ptr);
            std::vector<double>().swap(PtHP_temp);
            eigenSolver.computeEigenPairsSeq(PtHP,
                                             numberEigenValues,
                                             numberEigenValues,
                                             eig_vals,
                                             Q,
                                             true);
            MatDestroy(&PtHP);

            timer1.end("solve eig problem for PtHP");
            Utils::print_current_max_mem("memory usage after solving Pt*H*P");

            PetscPrintf(PETSC_COMM_WORLD,
                        "--------------------------------------------\n");
            PetscPrintf(PETSC_COMM_WORLD,
                        "Projected Eigenvalues of Chebyshev Pass %d\n",
                        restart_iter);
            PetscPrintf(PETSC_COMM_WORLD,
                        "--------------------------------------------\n");
            lowerBoundWantedSpectrum = eig_vals[0];
            lowerBoundUnWantedSpectrum = eig_vals[numberEigenValues - 1];

            double current_occupied_orbital_energy =
                    eig_vals[occupied_orbital_number];
            double error_in_occupied_orbital_energy = std::abs(
                    (current_occupied_orbital_energy - occupied_orbital_energy) /
                    occupied_orbital_energy);
            occupied_orbital_energy = current_occupied_orbital_energy;

            /* write out the eigenfunctions for restarting */

            if (error_in_occupied_orbital_energy < first_cheb_tolerance) {
                PetscPrintf(
                        PETSC_COMM_WORLD,
                        "the iteration converged at step %d.\n",
                        restart_iter);
                break;
            }
            int wfn_fd;
            double *wfn_data;
            timer1.start();
            {
                std::ofstream fout(restart_path_wfn + "/cheby_bounds.txt");
                fout << std::setprecision(16) << std::scientific;
                fout << upperBoundUnWantedSpectrum << "\t"
                     << lowerBoundUnWantedSpectrum << "\t"
                     << lowerBoundWantedSpectrum << "\t" << error_in_upper_bound
                     << "\t" << occupied_orbital_energy << "\t"
                     << groundStateEnergyOld;
                fout.close();
            }
            MatDenseGetArray(eig_vecs,
                             &wfn_data);
            PetscBinaryOpen(wfn_filenmae.c_str(),
                            FILE_MODE_WRITE,
                            &wfn_fd);
            PetscBinaryWrite(
                    wfn_fd,
                    wfn_data,
                    wfn_local_elements,
                    PETSC_DOUBLE,
                    PETSC_FALSE);
            PetscBinaryClose(wfn_fd);
            MatDenseRestoreArray(eig_vecs,
                                 &wfn_data);
            timer1.end("writing out wavefunctions.");
#ifndef NDEBUG
            PetscPrintf(PETSC_COMM_WORLD,
                        "restarting step %d with error %e\n",
                        restart_iter,
                        error_in_occupied_orbital_energy);
#endif
        }
        ax.~AX();

        project_hamiltonian_sparse.Destroy();

        // Perform basis rotation
        timer1.start();
        Mat eig_vecs_copy, eig_vecs_copy_seq, eig_vecs_seq;
        MatDuplicate(eig_vecs,
                     MAT_COPY_VALUES,
                     &eig_vecs_copy);
        MatDenseGetLocalMatrix(eig_vecs_copy,
                               &eig_vecs_copy_seq);
        MatDenseGetLocalMatrix(eig_vecs,
                               &eig_vecs_seq);
        MatMatMult(
                eig_vecs_copy_seq,
                Q,
                MAT_REUSE_MATRIX,
                PETSC_DEFAULT,
                &eig_vecs_seq);


        timer1.end("performing basis rotation");

        MatDestroy(&eig_vecs_copy);
        MatDestroy(&Q);
        timer1.end("performing basis rotation");

        {
            int wfn_fd;
            double *wfn_data;
            std::ofstream fout(restart_path_wfn + "/cheby_bounds.txt");
            fout << std::setprecision(16) << std::scientific;
            fout << upperBoundUnWantedSpectrum << "\t" << lowerBoundUnWantedSpectrum
                 << "\t" << lowerBoundWantedSpectrum << "\t" << error_in_upper_bound
                 << "\t" << occupied_orbital_energy << "\t" << groundStateEnergyOld;
            fout.close();
            timer1.start();
            MatDenseGetArray(eig_vecs,
                             &wfn_data);
            PetscBinaryOpen(wfn_filenmae.c_str(),
                            FILE_MODE_WRITE,
                            &wfn_fd);
            PetscBinaryWrite(
                    wfn_fd,
                    wfn_data,
                    wfn_local_elements,
                    PETSC_DOUBLE,
                    PETSC_FALSE);
            PetscBinaryClose(wfn_fd);
            MatDenseRestoreArray(eig_vecs,
                                 &wfn_data);
            timer1.end("writing out rotated wavefunctions.");
        }

        ChfTime.end("ChFSI eigen solver");
        Utils::print_current_max_mem("memory usage after Chebyshev filter");

        double FermiEnergy = 0.0;
        std::vector<double> occupancyFactor(numberEigenValues,
                                            0.0);
        if (taskId == rootId) {
            FermiEnergy =
                    ksdftEnergyFunctional.computeFermiEnergy(eig_vals.data(),
                                                             numberEigenValues,
                                                             numberElectrons,
                                                             boltzmannConstant,
                                                             temperature);

            printf("%d :Fermi Energy: %.18e\n",
                   taskId,
                   FermiEnergy);
            for (int i = 0; i < numberEigenValues; ++i) {
                double term =
                        (eig_vals[i] - FermiEnergy) / (boltzmannConstant * temperature);
                if (term <= 0.0) {
                    occupancyFactor[i] = 1.0 / (1.0 + std::exp(term));
                } else {
                    occupancyFactor[i] =
                            std::exp(-term) / (1.0 + std::exp(-term));
                }
                printf("Occupancy Factor %d\t%.18e\n",
                       i,
                       occupancyFactor[i]);
            }
        }
        MPI_Bcast(&occupancyFactor[0],
                  numberEigenValues,
                  MPI_DOUBLE,
                  rootId,
                  PETSC_COMM_WORLD);
        Utils::print_current_max_mem("memory usage after solving Fermi energy");

        timer1.start();
        project_hamiltonian_sparse.computeRhoOut(eig_vecs,
                                                 occupancyFactor,
                                                 rhoNodalOut,
                                                 rhoGridOut);
        timer1.end("computing rhoOuts and psiXYZ with old method");
        Utils::print_current_max_mem("memory usage after computing rho");

        Tensor3DMPI potHartreeOut(numberQuadPointsX,
                                  numberQuadPointsY,
                                  numberQuadPointsZ,
                                  PETSC_COMM_WORLD);

        timer1.start();
        if (is_using_kernel_expansion == true || ks_scf_converge_count == 0) {
            const TuckerMPI::TuckerTensor *decomposedRhoGridOut =
                    TensorUtils::computeSTHOSVDonQuadMPI(femElectroX,
                                                         femElectroY,
                                                         femElectroZ,
                                                         rankRho[0],
                                                         rankRho[1],
                                                         rankRho[2],
                                                         rhoNodalOut);

            ksdftPotential.computeHartreePotNew(decomposedRhoGridOut,
                                                potHartreeOut.tensor,
                                                INSERT_VALUES);

            Tucker::MemoryManager::safe_delete(decomposedRhoGridOut);
        } else {
            poissonHartreePotentialSolver->solve(input_parameter.maxHartreeIter,
                                                 input_parameter.hartreeTolerance,
                                                 rhoNodalOut,
                                                 potHartreeOut);
            ks_scf_converge_count++;
        }
        timer1.end("computing output Hartree potential using " + hartreeMethod);
        Utils::print_current_max_mem(
                "memory usage after computing Hartree potential");

        // compute band energy
        double bandEnergy = 0.0;
        for (int i = 0; i < numberEigenValues; ++i) {
            bandEnergy += 2 * occupancyFactor[i] * eig_vals[i];
        }

        double hartree_energy, xc_energy, eff_minus_psp_energy;

        if (input_parameter.is_break_apart_energies == true) {
            timer1.start();
            Tensor3DMPI hartree_energy_grid(numberQuadPointsX,
                                            numberQuadPointsY,
                                            numberQuadPointsZ,
                                            PETSC_COMM_WORLD);
            Tensor3DMPI xc_energy_grid(numberQuadPointsX,
                                       numberQuadPointsY,
                                       numberQuadPointsZ,
                                       PETSC_COMM_WORLD);
            Tensor3DMPI eff_minus_psp_energy_grid(numberQuadPointsX,
                                                  numberQuadPointsY,
                                                  numberQuadPointsZ,
                                                  PETSC_COMM_WORLD);

            ksdftEnergyFunctional.computeLDAExchangeCorrelationEnergy(
                    rhoGridOut,
                    xc_energy_grid,
                    TENSOR_INSERT);
            // 0.5 int (rho*pot_hartree)
            ksdftEnergyFunctional.computeHartreeEnergy(rhoGridOut,
                                                       potHartreeOut,
                                                       hartree_energy_grid,
                                                       TENSOR_INSERT);
            ksdftEnergyFunctional.computeEffMinusPSP(rhoGridOut,
                                                     potEffIn,
                                                     potLoc,
                                                     eff_minus_psp_energy_grid,
                                                     TENSOR_INSERT);

            hartree_energy =
                    ksdftEnergyFunctional.compute3DIntegral(hartree_energy_grid,
                                                            jacob3DMat,
                                                            weight3DMat);
            xc_energy = ksdftEnergyFunctional.compute3DIntegral(xc_energy_grid,
                                                                jacob3DMat,
                                                                weight3DMat);
            eff_minus_psp_energy =
                    ksdftEnergyFunctional.compute3DIntegral(eff_minus_psp_energy_grid,
                                                            jacob3DMat,
                                                            weight3DMat);
            timer1.end("computing break-apart energies");
        }
        Utils::print_current_max_mem(
                "memory usage after computing breaking apart energies");

        timer1.start();
        Tensor3DMPI groundStateEnergyGrid(numberQuadPointsX,
                                          numberQuadPointsY,
                                          numberQuadPointsZ,
                                          PETSC_COMM_WORLD);
        ksdftEnergyFunctional.computeLDAExchangeCorrelationEnergy(
                rhoGridOut,
                groundStateEnergyGrid,
                TENSOR_INSERT);
        ksdftEnergyFunctional.computeHartreeEnergy(rhoGridOut,
                                                   potHartreeOut,
                                                   groundStateEnergyGrid,
                                                   TENSOR_ADD);
        ksdftEnergyFunctional.computeEffMinusPSP(
                rhoGridOut,
                potEffIn,
                potLoc,
                groundStateEnergyGrid,
                TENSOR_SUBTRACT);
        timer1.end("compute potential energy on grid");

        timer1.start();
        groundStateEnergyCurrent =
                ksdftEnergyFunctional.compute3DIntegral(groundStateEnergyGrid,
                                                        jacob3DMat,
                                                        weight3DMat);
        timer1.end("3D integral for potential energy");
        timer1.start();
        double nuclei_energy;
        nuclei_energy = ksdftEnergyFunctional.computeRepulsiveEnergy(nuclei);
        groundStateEnergyCurrent += nuclei_energy;
        timer1.end("computing repulsive energy");

        groundStateEnergyCurrent += bandEnergy;

        if (scfIter > 0) {
            relErrorEnergy =
                    std::abs((groundStateEnergyCurrent - groundStateEnergyOld) /
                             groundStateEnergyOld);
        }
        groundStateEnergyOld = groundStateEnergyCurrent;

        Tensor3DMPI diffrhoGridSquare(numberQuadPointsX,
                                      numberQuadPointsY,
                                      numberQuadPointsZ,
                                      PETSC_COMM_WORLD);
        int localNumEntriesDiffrhoGridSquare =
                diffrhoGridSquare.getLocalNumberEntries();
        double *diffrhoGridSquareData = diffrhoGridSquare.getLocalData();
        const double *rhoGridOutData = rhoGridOut.getLocalData();
        const double *rhoGridInData = rhoGridIn.getLocalData();
        for (int i = 0; i < localNumEntriesDiffrhoGridSquare; ++i) {
            diffrhoGridSquareData[i] = (rhoGridOutData[i] - rhoGridInData[i]) *
                                       (rhoGridOutData[i] - rhoGridInData[i]);
        }

        timer1.start();
        double squareL2Normdiff =
                ksdftEnergyFunctional.compute3DIntegral(diffrhoGridSquare,
                                                        jacob3DMat,
                                                        weight3DMat);
        timer1.end("compute squareL2Normdiff");
        timer1.start();
        double chargeOut = ksdftEnergyFunctional.compute3DIntegral(rhoGridOut,
                                                                   jacob3DMat,
                                                                   weight3DMat);
        timer1.end("compute charge out");
        Utils::print_current_max_mem("memory usage after iteration");
        timer1.start();
        andersonMixing.updateRho(
                rhoNodalIn,
                rhoGridIn,
                rhoNodalOut,
                rhoGridOut,
                scfIter);
        timer1.end("pushing back computed rho into anderon mixing");
        Utils::print_current_max_mem("memory usage after storing rho");

        scftimer.end("completing this iteration");
        {
            std::ofstream fout(restart_path_wfn + "/cheby_bounds.txt");
            fout << std::setprecision(16) << std::scientific;
            fout << upperBoundUnWantedSpectrum << "\t" << lowerBoundUnWantedSpectrum
                 << "\t" << lowerBoundWantedSpectrum << "\t" << error_in_upper_bound
                 << "\t" << occupied_orbital_energy << "\t" << groundStateEnergyOld;
            fout.close();
        }
        //      delete projectHamiltonian;
        if (taskId == rootId) {
            const double hartree_to_eV = 27.21138602;
            std::cout << std::setprecision(16) << std::scientific;
            std::cout << "=====energy output in Hartree====" << std::endl;
            std::cout << "band energy: " << bandEnergy << std::endl;
            if (input_parameter.is_break_apart_energies == true) {
                std::cout << "Hartree energy 0.5*int(rho*v_h): " << hartree_energy
                          << std::endl;
                std::cout << "XC energy int(rho*v_xc): " << xc_energy
                          << std::endl;
                std::cout << "int(veff-psp_loc): " << eff_minus_psp_energy
                          << std::endl;
                std::cout << "nuclei energy: " << nuclei_energy << std::endl;
            }
            std::cout << "ground state energy : " << groundStateEnergyCurrent
                      << std::endl;
            std::cout << "ground state energy per atom : "
                      << groundStateEnergyCurrent / numAtoms << std::endl;
#ifdef PRINT_EV
            std::cout << "=====energy output in eV====" << std::endl;
            std::cout << "band energy: " << bandEnergy * hartree_to_eV
                      << std::endl;
            if (input_parameter.is_break_apart_energies == true)
              {
                std::cout << "Hartree energy 0.5*int(rho*v_h): "
                          << hartree_energy * hartree_to_eV << std::endl;
                std::cout << "X C energy int(rho*v_xc): "
                          << xc_energy * hartree_to_eV << std::endl;
                std::cout << "int(veff-psp_loc): "
                          << eff_minus_psp_energy * hartree_to_eV << std::endl;
                std::cout << "nuclei energy: " << nuclei_energy * hartree_to_eV
                          << std::endl;
              }
            std::cout << "ground state energy : "
                      << groundStateEnergyCurrent * hartree_to_eV << std::endl;
            std::cout << "ground state energy per atom : "
                      << groundStateEnergyCurrent / numAtoms * hartree_to_eV
                      << std::endl;
#endif
            std::cout << "=====error computation=====" << std::endl;
            std::cout << std::setprecision(18) << std::scientific;
            std::cout << "square L2 diff norm: " << squareL2Normdiff << std::endl;
            std::cout << "relative error: " << relErrorEnergy << std::endl;
            std::cout << "charge out: " << chargeOut << std::endl;
            std::cout << "====================================" << std::endl;
            std::cout << "        End of SCF " << scfIter << std::endl;
            std::cout << "====================================" << std::endl;
        }
        if (taskId == rootId) {
            std::cout << "====================================" << std::endl;
            std::cout << "        End of Mem " << std::endl;
            std::cout << "====================================" << std::endl;
        }

        Tucker::MemoryManager::safe_delete(decomposedPotEff);

        if ((squareL2Normdiff < scfTol) && (relErrorEnergy < scfTol)) {
            if (taskId == 0) {
                std::cout << "====================================" << std::endl;
                std::cout << " SCF has converged at " << scfIter << "iterations."
                          << std::endl;
                std::cout << "====================================" << std::endl;
            }
            totalTimeTimer.end("total iterations");
            break;
        }
    }

    SlepcFinalize();
    return 0;
}

namespace {
    void
    write_out_rho(const FEM &femX,
                  const FEM &femY,
                  const FEM &femZ,
                  int current_rank_x,
                  int current_rank_y,
                  int current_rank_z,
                  Tensor3DMPI &rho) {
        int taskId;
        MPI_Comm_rank(MPI_COMM_WORLD,
                      &taskId);

        // assume rank35 is enough for decompose an electron density of grid size
        // 100(20 quintic elements)
        Tucker::SizeArray output_rank(3);
        output_rank[0] =
                int(100 * std::log(femX.getTotalNumberNodes()) / std::log(250));
        output_rank[1] =
                int(100 * std::log(femY.getTotalNumberNodes()) / std::log(250));
        output_rank[2] =
                int(100 * std::log(femZ.getTotalNumberNodes()) / std::log(250));

        if (output_rank[0] >= femX.getTotalNumberNodes())
            output_rank[0] = int(femX.getTotalNumberNodes() * 0.6);
        if (output_rank[1] >= femY.getTotalNumberNodes())
            output_rank[1] = int(femY.getTotalNumberNodes() * 0.6);
        if (output_rank[2] >= femZ.getTotalNumberNodes())
            output_rank[2] = int(femZ.getTotalNumberNodes() * 0.6);

        const TuckerMPI::TuckerTensor *decomposed_rho =
                TuckerMPI::STHOSVD(rho.getTensor(),
                                   &output_rank);
        Tucker::Tensor *seq_core_rho =
                Tucker::MemoryManager::safe_new<Tucker::Tensor>(output_rank);
        TensorUtils::allgatherTensor(decomposed_rho->G,
                                     seq_core_rho);
        if (taskId == 0) {
            std::cout << "ouput rank: (" << output_rank[0] << ", " << output_rank[1]
                      << ", " << output_rank[2] << ")" << std::endl;
            std::ofstream fout;
            std::cout << "writing out core tensor of electron density: "
                      << std::endl;
            fout.open("rank_" + std::to_string(current_rank_x) + "_core");
            fout << output_rank[0] << " " << output_rank[1] << " " << output_rank[2]
                 << std::endl;
            int cnt = 0;
            for (int k = 0; k < output_rank[2]; ++k) {
                for (int j = 0; j < output_rank[1]; ++j) {
                    for (int i = 0; i < output_rank[0]; ++i) {
                        fout << std::setprecision(18) << std::scientific
                             << seq_core_rho->data()[cnt] << std::endl;
                        cnt++;
                    }
                }
            }
            fout.close();

            auto output_factor_matrices = [](const FEM &fem,
                                             int rank,
                                             Tucker::Matrix &factor,
                                             std::string filename) {
                std::ofstream printout(filename);
                int number_nodes_fem = fem.getTotalNumberNodes();
                printout << number_nodes_fem << " " << rank << std::endl;
                for (int i = 0; i < number_nodes_fem; ++i) {
                    printout << std::setprecision(18) << std::scientific
                             << fem.getGlobalNodalCoord()[i] << '\t';
                }
                printout << std::endl;
                for (int output_r = 0; output_r < rank; ++output_r) {
                    for (int i = 0; i < number_nodes_fem; ++i) {
                        printout << std::setprecision(18) << std::scientific
                                 << factor.data()[output_r * number_nodes_fem + i]
                                 << '\t';
                    }
                    printout << std::endl;
                }
                printout.clear();
                printout.close();
            };

            std::cout << "writinng x nodal values" << std::endl;
            output_factor_matrices(femX,
                                   output_rank[0],
                                   *(decomposed_rho->U[0]),
                                   "rank_" + std::to_string(current_rank_x) +
                                   "_factor_x");
            std::cout << "writinng y nodal values" << std::endl;
            output_factor_matrices(femY,
                                   output_rank[1],
                                   *(decomposed_rho->U[1]),
                                   "rank_" + std::to_string(current_rank_y) +
                                   "_factor_y");
            std::cout << "writinng z nodal values" << std::endl;
            output_factor_matrices(femZ,
                                   output_rank[2],
                                   *(decomposed_rho->U[2]),
                                   "rank_" + std::to_string(current_rank_z) +
                                   "_factor_z");

            Tucker::MemoryManager::safe_delete(seq_core_rho);
            Tucker::MemoryManager::safe_delete(decomposed_rho);
        }
    }

    void
    computeJacobWeight3DMat(const FEM &femX,
                            const FEM &femY,
                            const FEM &femZ,
                            Tensor3DMPI &jacob3DMat,
                            Tensor3DMPI &weight3DMat) {
        jacob3DMat.setEntriesZero();
        weight3DMat.setEntriesZero();
        const std::vector<double> &jacobX = femX.getJacobQuadPointValues();
        const std::vector<double> &jacobY = femY.getJacobQuadPointValues();
        const std::vector<double> &jacobZ = femZ.getJacobQuadPointValues();
        const std::vector<double> &weightX = femX.getWeightQuadPointValues();
        const std::vector<double> &weightY = femY.getWeightQuadPointValues();
        const std::vector<double> &weightZ = femZ.getWeightQuadPointValues();

        std::array<int, 6> jacob3DMatGlobalIdx;
        std::array<int, 6> weight3DMatGlobalIdx;
        double *jacob3DMatLocal = jacob3DMat.getLocalData(jacob3DMatGlobalIdx);
        double *weight3DMatLocal = weight3DMat.getLocalData(weight3DMatGlobalIdx);

        int cnt = 0;
        for (int k = jacob3DMatGlobalIdx[4]; k < jacob3DMatGlobalIdx[5]; ++k) {
            for (int j = jacob3DMatGlobalIdx[2]; j < jacob3DMatGlobalIdx[3]; ++j) {
                for (int i = jacob3DMatGlobalIdx[0]; i < jacob3DMatGlobalIdx[1];
                     ++i) {
                    jacob3DMatLocal[cnt] = jacobX[i] * jacobY[j] * jacobZ[k];
                    cnt = cnt + 1;
                }
            }
        }

        cnt = 0;
        for (int k = weight3DMatGlobalIdx[4]; k < weight3DMatGlobalIdx[5]; ++k) {
            for (int j = weight3DMatGlobalIdx[2]; j < weight3DMatGlobalIdx[3]; ++j) {
                for (int i = weight3DMatGlobalIdx[0]; i < weight3DMatGlobalIdx[1];
                     ++i) {
                    weight3DMatLocal[cnt] = weightX[i] * weightY[j] * weightZ[k];
                    cnt = cnt + 1;
                }
            }
        }
    }
} // namespace
