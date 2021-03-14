cmake_minimum_required(VERSION 3.10)
project(SparseProject)
#set(ENV{LD_LIBRARY_PATH} "/home/aromanov/intel/oneapi/mkl/2021.1.1/env/../lib/intel64:/usr/local/lib:/home/aromanov/Programs/AMDAPPSDK-3.0/lib/x86_64")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(BLA_DEFINITIONS)

if ("$ENV{MKLROOT}" STREQUAL "")
    message(FATAL_ERROR "MKLROOT is not set. Please source the Intel MKL mklvars.sh file.")
endif()

option(MKL_USE_parallel "Use MKL parallel" True)
option(MKL_USE_sdl "Single Dynamic Library or static/dynamic" False)
set(MKL_USE_interface "lp64" CACHE STRING "for Intel(R)64 compatible arch: ilp64/lp64 or for ia32 arch: cdecl/stdcall")

if (BLA_VENDOR MATCHES "_seq")
    set(MKL_USE_parallel OFF)
else()
    set(MKL_USE_parallel ON)
endif()

find_package(MKL REQUIRED)

if (MKL_FOUND)
    set(BLA_INCLUDE_DIR ${MKL_INCLUDE_DIR})
    set(BLAS_LIBRARIES ${MKL_LIBRARIES})
    set(BLA_DEFINITIONS USE_MKL)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR})
endif()
message(STATUS "BLAS Libraries: ${BLAS_LIBRARIES}")

find_package(OpenCL REQUIRED)
if (OPENCL_FOUND)
    include_directories( ${OPENCL_INCLUDE_DIR} )
endif()
message(STATUS "OpenCL Libraries: ${OPENCL_LIBRARIES}")

find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
#    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}") ??
    message(STATUS "OpenMP Libraries: ${OpenMP_CXX_LIBRARIES}")
endif()

get_filename_component(DIR_NAME ${CMAKE_CURRENT_LIST_DIR} NAME)
set(MODULE      "${DIR_NAME}")
set(LIBRARY     "lib_${MODULE}")
set(APPLICATION "app_${MODULE}")


add_subdirectory(src)
add_subdirectory(app)
