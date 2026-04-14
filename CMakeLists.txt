cmake_minimum_required(VERSION 3.14)
project(jpeg_projekt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(eigen)

FetchContent_Declare(
        xsimd
        GIT_REPOSITORY https://github.com/xtensor-stack/xsimd.git
        GIT_TAG 13.2.0
)
FetchContent_MakeAvailable(xsimd)

if(APPLE) 
    FetchContent_Declare(
            bshoshany_thread_pool
            GIT_REPOSITORY https://github.com/bshoshany/thread-pool.git
            GIT_TAG v5.0.0
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
            EXCLUDE_FROM_ALL
            SYSTEM
    )
    FetchContent_MakeAvailable(bshoshany_thread_pool)
    add_library(BS_thread_pool INTERFACE)
    target_include_directories(BS_thread_pool INTERFACE ${bshoshany_thread_pool_SOURCE_DIR}/include)
endif()


# --- Include your own headers ---
include_directories(include)

# --- Gather source files ---
file(GLOB_RECURSE SRC_FILES src/*.cpp)
add_executable(${PROJECT_NAME} main.cpp ${SRC_FILES}
        include/Sampling.h
        include/utils.h
        include/BitStream.h
        include/Segments.h
        include/Huffman.h
        include/DCT.h
        include/Quantization.h
        include/RLE.h)

if(APPLE)
    target_link_libraries(${PROJECT_NAME} PRIVATE BS_thread_pool)
endif()

# --- Add Eigen include path ---
target_include_directories(${PROJECT_NAME} PRIVATE ${eigen_SOURCE_DIR})
target_include_directories(${PROJECT_NAME} PRIVATE ${xsimd_SOURCE_DIR}/include)

# --- OpenMP support (skip on Apple) ---
if (NOT APPLE)
    find_package(OpenMP)
    if (OpenMP_CXX_FOUND)
        target_compile_options(${PROJECT_NAME} PUBLIC -fopenmp)
        target_link_libraries(${PROJECT_NAME} PUBLIC -fopenmp)
    endif ()
endif ()

if (NOT APPLE) # Platform-specific compiler/linker flags (non-Apple toolchains)
    target_compile_options(${PROJECT_NAME} PRIVATE -mavx -mavx2 -mfma -ftree-vectorize -funroll-loops -finline-functions)
    target_link_options(${PROJECT_NAME} PRIVATE -mavx -mavx2 -mfma)
endif ()
# the big fast
if (NOT APPLE)
    target_compile_options(${PROJECT_NAME} PRIVATE -march=native -O3 -ffast-math)
    target_link_options(${PROJECT_NAME} PRIVATE -march=native)
endif ()

