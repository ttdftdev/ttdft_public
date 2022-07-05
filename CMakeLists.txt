cmake_minimum_required(VERSION 3.16)
project(ttdft_gpu CXX CUDA)

#set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_CXX_STANDARD_REQUIRED True)
#
#set(CMAKE_CUDA_STANDARD 14)
#set(CMAKE_CUDA_STANDARD_REQUIRED True)

include(cmake/CMakeSources.cmake)

set(PETSC_DIR "$ENV{PETSC_DIR}")
set(PETSC_ARCH "$ENV{PETSC_ARCH}")
set(SLEPC_DIR $ENV{SLEPC_DIR})
set(TUCKER_DIR "$ENV{TUCKER_DIR}")
set(TUCKER_BUILD_DIR "$ENV{TUCKER_BUILD_DIR}")
find_package(BLAS)
find_package(LAPACK)
find_package(MPI REQUIRED)
find_package(Boost REQUIRED)
message("${TUCKER_BUILD_DIR}")

find_package(CUDAToolkit 10 EXACT REQUIRED)

# MPI
include_directories($ENV{MPI_HOME}/include)
link_directories($ENV{MPI_HOME}/lib)

# MKL
if (EXISTS $ENV{CRAY_LIBSCI_DIR})
    message("libsci used for blas/lapack/scalapack")
    include_directories($ENV{CRAY_LIBSCI_PREFIX_DIR}/include)
    link_directories($ENV{CRAY_LIBSCI_PREFIX_DIR}/lib)
    set(LINALG_LIBS sci_intel_mpi_mp)
elseif (EXISTS $ENV{MKLROOT})
    message("MKL Library is used")
    include_directories($ENV{MKL_INCLUDE})
    link_directories($ENV{MKL_LIB})
    set(LINALG_LIBS mkl_intel_thread mkl_core mkl_intel_lp64 iomp5 pthread)
elseif (BLAS_FOUND AND LAPACK_FOUND)
    message("blas/lapack are used")
    set(LINALG_LIBS ${LIBS} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
else (EXISTS $ENV{CRAY_LIBSCI_DIR})
    message(SEND_ERROR "Cannot find MKL! Please set MKLROOT and configure again.")
endif (EXISTS $ENV{CRAY_LIBSCI_DIR})

# boost
if (Boost_FOUND)
    message("Found Boost version ${Boost_VERSION}")
    include_directories($ENV{BOOST_INC})
    link_directories($ENV{BOOST_LIB})
    set(BOOST_LIBS boost_regex boost_filesystem boost_system)
else (Boost_FOUND)
    message(SEND_ERROR "Cannot find boost! Please load module boost or set BOOST_ROOT and configure again.")
endif (Boost_FOUND)

# PETSc
if (PETSC_DIR)
    message("Found PETSc")
    include_directories(${PETSC_DIR}/include)
    include_directories(${PETSC_DIR}/${PETSC_ARCH}/include)
    link_directories(${PETSC_DIR}/${PETSC_ARCH}/lib)
    set(PETSC_LIBS petsc)
    message("PETSc dir: ${PETSC_DIR}")
    message("PETSc arch: ${PETSC_ARCH}")
else (PETSC_DIR)
    message(SEND_ERROR "Cannot find petsc! Please load module petsc or set PETSC_DIR PETSC_ARCH and configure again.")
endif (PETSC_DIR)

# SLEPc
find_path(SLEPC_DIR include/slepc.h)
if (SLEPC_DIR AND EXISTS ${SLEPC_DIR}/include/slepc.h)
    message("Found SLEPc")
    set(HAVE_PKG_slepc 1)
    include_directories(${SLEPC_DIR}/include)
    include_directories(${SLEPC_DIR}/${PETSC_ARCH}/include)
    link_directories(${SLEPC_DIR}/${PETSC_ARCH}/lib)
    set(SLEPC_LIBS slepc)
    message("PETSc dir: ${SLEPC_DIR}")
else (SLEPC_DIR AND EXISTS ${SLEPC_DIR}/include/slepc.h)
    message(SEND_ERROR "Cannot find Slepc! Please set SLEPC_DIR and configure again.")
endif (SLEPC_DIR AND EXISTS ${SLEPC_DIR}/include/slepc.h)

if (TUCKER_DIR AND EXISTS ${TUCKER_BUILD_DIR}/mpi/libmpi_tucker.a AND EXISTS ${TUCKER_BUILD_DIR}/serial/libserial_tucker.a)
    include_directories(${TUCKER_DIR}/src/mpi)
    include_directories(${TUCKER_DIR}/src/serial)
    link_directories(${TUCKER_BUILD_DIR}/mpi)
    link_directories(${TUCKER_BUILD_DIR}/serial)
    set(TUCKER_LIBS mpi_tucker serial_tucker)
else (TUCKER_DIR AND EXISTS ${TUCKER_BUILD_DIR}/mpi/libmpi_tucker.a AND EXISTS ${TUCKER_BUILD_DIR}/serial/libserial_tucker.a)
    message(SEND_ERROR "Cannot find Tucker library")
endif (TUCKER_DIR AND EXISTS ${TUCKER_BUILD_DIR}/mpi/libmpi_tucker.a AND EXISTS ${TUCKER_BUILD_DIR}/serial/libserial_tucker.a)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

#message(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set(LIBS cudart cusparse cublas cusolver ${SLEPC_LIBS} ${PETSC_LIBS} ${TUCKER_LIBS} ${MPI_LIBS} ${LINALG_LIBS}
        ${BOOST_LIBS})

add_library(tuckerFEM ${fem_sources})
add_library(tuckerDFT ${dft_sources})
add_library(tuckerEigen ${eigen_sources})
add_library(tuckerTensor ${tensor_sources})
add_library(tuckerAlglib ${alglib_sources})

add_executable(main main.cc ${other_source})

target_link_libraries(main tuckerFEM tuckerDFT tuckerTensor tuckerEigen tuckerAlglib ${LIBS})