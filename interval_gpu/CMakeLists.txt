cmake_minimum_required(VERSION 3.18)
project(interval_app LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 89)

option(USE_GAOL "Enable GAOL refinement" ON)

# If you want to force the same host compiler you use for GAOL:
set(CMAKE_CXX_COMPILER g++-9 CACHE FILEPATH "" FORCE)
set(CMAKE_CUDA_HOST_COMPILER g++-9 CACHE FILEPATH "" FORCE)

add_executable(interval_app main.cu)

target_compile_options(interval_app PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_89,code=sm_89>
)

if(USE_GAOL)
    find_library(GAOL_LIB gaol HINTS /usr/local/lib /usr/lib)
    find_library(ULTIM_LIB ultim HINTS /usr/local/lib /usr/lib)
    find_library(GDTOA_LIB gdtoa HINTS /usr/local/lib /usr/lib)
    find_library(GINAC_LIB ginac HINTS /usr/local/lib /usr/lib)
    find_library(CLN_LIB cln HINTS /usr/local/lib /usr/lib)

    if(NOT GAOL_LIB)
        message(FATAL_ERROR "Could not find libgaol")
    endif()

    target_compile_definitions(interval_app PRIVATE USE_GAOL)

    target_include_directories(interval_app PRIVATE /usr/local/include)
    target_link_directories(interval_app PRIVATE /usr/local/lib)

    target_link_libraries(interval_app PRIVATE
        ${GAOL_LIB}
        ${ULTIM_LIB}
        ${GDTOA_LIB}
        ${GINAC_LIB}
        ${CLN_LIB}
        stdc++fs
    )
endif()