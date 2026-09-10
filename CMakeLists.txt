cmake_minimum_required(VERSION 3.16)
project(dials-metal-spotfinder VERSION 0.1.0 LANGUAGES C CXX)  # C for bitshuffle and lz4

# The version lives in project() above and nowhere else. vcpkg.json has to carry
# its own copy because vcpkg reads it before CMake runs, so the two are checked
# against each other here rather than left to drift -- "which build is this?" is
# the first question asked in an incident, and an answer that is quietly wrong is
# worse than none.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg.json" spotfinder_manifest)
string(REGEX MATCH "\"version-string\"[ \t]*:[ \t]*\"([^\"]+)\""
       spotfinder_manifest_version "${spotfinder_manifest}")
if(NOT CMAKE_MATCH_1 STREQUAL PROJECT_VERSION)
    message(FATAL_ERROR
            "version-string in vcpkg.json is ${CMAKE_MATCH_1} but project() says "
            "${PROJECT_VERSION}; make them agree")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(SPOTFINDER_TESTS "Build the tests" ON)
option(SPOTFINDER_CUDA "Build the CUDA threshold kernels" OFF)
option(SPOTFINDER_METAL "Build the Metal threshold kernels" OFF)

# One backend at a time. gpu::find has one definition, and a build with both
# would either fail at the linker or -- worse -- link whichever the linker saw
# first, so that -gpu ran a device nobody chose.
if(SPOTFINDER_CUDA AND SPOTFINDER_METAL)
    message(FATAL_ERROR
            "SPOTFINDER_CUDA and SPOTFINDER_METAL are both on, and they implement "
            "the same interface. Pick one.")
endif()

# CUDA is enabled as a language only when asked for, so that a machine without a
# toolkit still configures. check_language reports its absence as a plain message
# rather than the hard failure enable_language would give.
if(SPOTFINDER_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR
                "SPOTFINDER_CUDA is on but no CUDA compiler was found. Put nvcc "
                "on PATH, or set CMAKE_CUDA_COMPILER, or configure with "
                "-DSPOTFINDER_CUDA=OFF.")
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # Left to the user, since the right answer is whichever cards this will run
    # on. "native" needs CMake 3.24; before that, name them, e.g. "70;80;90".
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
            set(CMAKE_CUDA_ARCHITECTURES native)
        else()
            set(CMAKE_CUDA_ARCHITECTURES 70)
        endif()
    endif()
    message(STATUS "CUDA: building for architectures ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# Metal, likewise, is off by default and everything it needs is looked for only
# when it is asked for, so that a Linux machine still configures.
#
# metal-cpp is a header-only wrapper Apple distributes as a zip rather than
# through any package manager, so it is pointed at rather than found:
#
#   cmake -S . -B build -DSPOTFINDER_METAL=ON -DMETAL_CPP_DIR=/path/to/metal-cpp
if(SPOTFINDER_METAL)
    if(NOT APPLE)
        message(FATAL_ERROR
                "SPOTFINDER_METAL is on but this is not macOS. The CPU path runs "
                "everywhere; -DSPOTFINDER_CUDA=ON is the other device.")
    endif()

    find_path(METAL_CPP_INCLUDE_DIR
              NAMES Metal/Metal.hpp
              HINTS ${METAL_CPP_DIR} ENV METAL_CPP_DIR
              PATH_SUFFIXES metal-cpp
              DOC "Directory containing Metal/Metal.hpp from Apple's metal-cpp")
    if(NOT METAL_CPP_INCLUDE_DIR)
        message(FATAL_ERROR
                "metal-cpp was not found. Download it from "
                "https://developer.apple.com/metal/cpp/ and configure with "
                "-DMETAL_CPP_DIR=<the unpacked directory>.")
    endif()

    find_program(SPOTFINDER_XCRUN xcrun)
    if(NOT SPOTFINDER_XCRUN)
        message(FATAL_ERROR
                "xcrun was not found, so the shader cannot be compiled. Install "
                "the Xcode command line tools.")
    endif()

    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)

    message(STATUS "Metal: metal-cpp at ${METAL_CPP_INCLUDE_DIR}")

    # The shader is compiled at build time rather than from source at run time.
    # newLibraryWithSource needs the Metal compiler present on the machine that
    # runs the binary, which is a beamline machine, and a shader that only fails
    # to compile at the first frame of an acquisition is a bad trade for the
    # second or so it saves here.
    #
    # -fno-fast-math because the point of that file is that it agrees with
    # dext.cc: fast math is on by default and would reassociate the dispersion
    # test out from under the comparison.
    set(spotfinder_metal_air "${CMAKE_CURRENT_BINARY_DIR}/dext_metal.air")
    set(spotfinder_metallib "${CMAKE_CURRENT_BINARY_DIR}/dext_metal.metallib")
    set(spotfinder_metallib_cc "${CMAKE_CURRENT_BINARY_DIR}/dext_metal_library.cc")

    add_custom_command(
        OUTPUT ${spotfinder_metal_air}
        COMMAND ${SPOTFINDER_XCRUN} -sdk macosx metal
                -std=metal3.0 -fno-fast-math -Wall -Werror
                -c "${CMAKE_CURRENT_SOURCE_DIR}/src/dext_metal.metal"
                -o ${spotfinder_metal_air}
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/dext_metal.metal"
        COMMENT "Compiling src/dext_metal.metal")

    add_custom_command(
        OUTPUT ${spotfinder_metallib}
        COMMAND ${SPOTFINDER_XCRUN} -sdk macosx metallib ${spotfinder_metal_air}
                -o ${spotfinder_metallib}
        DEPENDS ${spotfinder_metal_air}
        COMMENT "Linking dext_metal.metallib")

    add_custom_command(
        OUTPUT ${spotfinder_metallib_cc}
        COMMAND ${CMAKE_COMMAND} -DINPUT=${spotfinder_metallib}
                -DOUTPUT=${spotfinder_metallib_cc}
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_metallib.cmake"
        DEPENDS ${spotfinder_metallib}
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_metallib.cmake"
        COMMENT "Embedding dext_metal.metallib")
endif()

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/bitshuffle/src/bitshuffle.h")
    message(FATAL_ERROR
            "third_party/bitshuffle is missing: run 'git submodule update --init'")
endif()

# Sanitizers, for chasing a fault that only appears under load. Applied with
# add_compile_options rather than CMAKE_CXX_FLAGS so that they reach everything
# built in this tree -- bitshuffle included, which matters: a sanitizer only
# reports an access it instrumented, so a read off the end of a buffer inside
# bitshuffle is invisible unless bitshuffle itself was built this way.
set(SPOTFINDER_SANITIZE "" CACHE STRING
    "Sanitizers to build with: address, undefined, address,undefined, or thread")
set_property(CACHE SPOTFINDER_SANITIZE PROPERTY STRINGS
             "" address undefined "address,undefined" thread)

if(SPOTFINDER_SANITIZE)
    if(MSVC)
        message(FATAL_ERROR "SPOTFINDER_SANITIZE is not supported with MSVC here")
    endif()
    # nvcc does not take -fsanitize itself; it has to be handed to the host
    # compiler, one flag per sanitizer, because nvcc splits -Xcompiler arguments
    # on commas. A device fault needs compute-sanitizer instead.
    string(REPLACE "," ";" spotfinder_sanitizers "${SPOTFINDER_SANITIZE}")
    set(spotfinder_sanitize_host "")
    set(spotfinder_sanitize_cuda "")
    foreach(sanitizer IN LISTS spotfinder_sanitizers)
        list(APPEND spotfinder_sanitize_host "-fsanitize=${sanitizer}")
        list(APPEND spotfinder_sanitize_cuda "-Xcompiler=-fsanitize=${sanitizer}")
    endforeach()

    add_compile_options(
        "$<$<COMPILE_LANGUAGE:C,CXX>:${spotfinder_sanitize_host};-fno-omit-frame-pointer>"
        "$<$<COMPILE_LANGUAGE:CUDA>:${spotfinder_sanitize_cuda};-Xcompiler=-fno-omit-frame-pointer>")
    add_link_options(${spotfinder_sanitize_host})
    message(STATUS "sanitizers: ${SPOTFINDER_SANITIZE}, applied to everything built here")
endif()

# Warning flags as usage requirements, so that every target picks them up by
# linking this and the compiler check lives in one place. An INTERFACE library
# with no location contributes nothing to the link line.
add_library(spotfinder_warnings INTERFACE)
if(NOT MSVC)
    target_compile_options(spotfinder_warnings INTERFACE -Wall -Wextra)
endif()

# bitshuffle has no CMake build of its own -- it is packaged for Python -- so the
# four C files that make up the library are compiled here. bshuf_h5filter.c and
# the plugin sources are deliberately left out: they need HDF5, and nothing here
# goes through HDF5's filter pipeline.
add_library(bitshuffle STATIC
    third_party/bitshuffle/src/bitshuffle.c
    third_party/bitshuffle/src/bitshuffle_core.c
    third_party/bitshuffle/src/iochain.c
    third_party/bitshuffle/lz4/lz4.c)
set_target_properties(bitshuffle PROPERTIES C_STANDARD 99)
target_include_directories(bitshuffle SYSTEM PUBLIC
    third_party/bitshuffle/src third_party/bitshuffle/lz4)

# bitshuffle chooses its SIMD path at compile time, from what the compiler says
# the target has: __SSE2__, __AVX2__, or __ARM_NEON with __aarch64__. Two of those
# need no help, since SSE2 is baseline on x86-64 and NEON is baseline on aarch64
# -- a plain build is already vectorised on both. AVX2 is the tier that has to be
# asked for, and asking for it is only meaningful on x86.
option(SPOTFINDER_AVX2
       "Compile bitshuffle's AVX2 path; the binary then requires an AVX2 CPU" ON)

set(spotfinder_target_arch "${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_OSX_ARCHITECTURES)
    # A universal or cross build overrides the host processor and may name more
    # than one architecture, so "arm64" and "arm64;x86_64" both fall through to
    # the else branch below -- which is what we want, since -mavx2 would fail the
    # arm64 slice.
    set(spotfinder_target_arch "${CMAKE_OSX_ARCHITECTURES}")
endif()

if(NOT SPOTFINDER_AVX2)
    message(STATUS "bitshuffle: AVX2 off by request")
elseif(NOT spotfinder_target_arch MATCHES "^(x86_64|amd64|AMD64|i[3-6]86)$")
    message(STATUS
            "bitshuffle: ${spotfinder_target_arch} is not x86, so AVX2 does not "
            "apply; NEON is baseline on aarch64")
else()
    include(CheckCCompilerFlag)
    if(MSVC)
        set(spotfinder_avx2_flag /arch:AVX2)
    else()
        set(spotfinder_avx2_flag -mavx2)
    endif()
    # Ask the compiler rather than assuming: a clang targeting arm64 rejects
    # -mavx2 outright, and this is the last line of defence if the architecture
    # test above is ever fooled.
    check_c_compiler_flag(${spotfinder_avx2_flag} SPOTFINDER_COMPILER_TAKES_AVX2)
    if(SPOTFINDER_COMPILER_TAKES_AVX2)
        target_compile_options(bitshuffle PRIVATE ${spotfinder_avx2_flag})
        message(STATUS
                "bitshuffle: AVX2 path enabled, so this build will not run on a "
                "CPU without AVX2 -- configure with -DSPOTFINDER_AVX2=OFF for a "
                "portable SSE2 build")
    else()
        message(STATUS
                "bitshuffle: ${spotfinder_avx2_flag} rejected by the C compiler, "
                "falling back to SSE2")
    endif()
endif()

find_package(Threads REQUIRED)

# HDF5, the C library only, and only H5D chunk reads plus the virtual-dataset
# property list calls. Both need 1.10.3 or newer.
find_package(HDF5 REQUIRED COMPONENTS C)

# HDF5 2.0 gave H5Dread_chunk a buffer-size argument, and its versioned-API macros
# point the old name at the new function, so the call has two shapes and no macro
# tells them apart. Ask the compiler which one is in front of it rather than
# guessing from H5_VERSION_GE: vcpkg and the distributions are on different sides
# of this. Compiled as a static library so the probe does not need to link.
include(CheckCXXSourceCompiles)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
set(CMAKE_REQUIRED_INCLUDES ${HDF5_C_INCLUDE_DIRS})
check_cxx_source_compiles("
#include <hdf5.h>
#include <cstddef>
#include <cstdint>
herr_t probe(hid_t dataset, const hsize_t *offset, std::uint32_t *filters,
             void *buffer, std::size_t *size) {
  return H5Dread_chunk(dataset, H5P_DEFAULT, offset, filters, buffer, size);
}
" SPOTFINDER_H5DREAD_CHUNK_TAKES_SIZE)
unset(CMAKE_REQUIRED_INCLUDES)
unset(CMAKE_TRY_COMPILE_TARGET_TYPE)
if(SPOTFINDER_H5DREAD_CHUNK_TAKES_SIZE)
    message(STATUS "HDF5: H5Dread_chunk takes a buffer size (2.0 and later)")
else()
    message(STATUS "HDF5: H5Dread_chunk takes no buffer size (before 2.0)")
endif()

# Where frames come from: an NXmx master file whose virtual dataset is unpacked
# so its chunks can be read directly. HDF5 stays behind this library rather than
# reaching the tool.
add_library(spotfinder_series STATIC src/nxmx.cc)
target_include_directories(spotfinder_series PUBLIC src)
target_include_directories(spotfinder_series SYSTEM PRIVATE ${HDF5_C_INCLUDE_DIRS})
target_link_libraries(spotfinder_series
    PRIVATE ${HDF5_C_LIBRARIES} spotfinder_warnings)
target_compile_definitions(spotfinder_series PRIVATE
    $<$<BOOL:${SPOTFINDER_H5DREAD_CHUNK_TAKES_SIZE}>:SPOTFINDER_H5DREAD_CHUNK_TAKES_SIZE>)

# bitshuffle+LZ4, plain LZ4 and uncompressed chunks.
add_library(spotfinder_decompress STATIC src/decompress.cc)
target_include_directories(spotfinder_decompress PUBLIC src)
target_link_libraries(spotfinder_decompress PRIVATE bitshuffle spotfinder_warnings)

# The extended dispersion threshold, explicitly instantiated for 16 and 32 bit
# pixels.
add_library(spotfinder_dext STATIC src/dext.cc)
target_include_directories(spotfinder_dext PUBLIC src)
target_link_libraries(spotfinder_dext PRIVATE spotfinder_warnings)

# Ordering a device's signal pixels by index. Host code, shared by both device
# backends, and its own library because it is worth testing on a machine with no
# GPU at all -- which is where it was written and where its bug was found.
add_library(spotfinder_signal_order STATIC src/signal_order.cc)
target_include_directories(spotfinder_signal_order PUBLIC src)
target_link_libraries(spotfinder_signal_order PRIVATE spotfinder_warnings)

# The backend-agnostic half of gpu::: which window each stage uses, whether to
# profile, and the last frame's split. Plain C++ with no toolkit in it, so it
# builds whether the backend is CUDA or Metal and is compiled once either way.
add_library(spotfinder_dext_gpu STATIC src/dext_gpu.cc)
target_include_directories(spotfinder_dext_gpu PUBLIC src)
target_link_libraries(spotfinder_dext_gpu PRIVATE spotfinder_warnings)

# Grouping signal pixels six-connected in three dimensions, as DIALS does, with
# its centroids and its filters. No Boost: the streaming flush renumbers
# components as it compacts, and a union-find it owns is both smaller than
# adjacency_list and the reason the whole sweep need not be held in memory.
add_library(spotfinder_dials_spots STATIC src/dials_spots.cc)
target_include_directories(spotfinder_dials_spots PUBLIC src)
target_link_libraries(spotfinder_dials_spots PRIVATE spotfinder_warnings)

# Writing a DIALS reflection table. A .refl is msgpack around raw column dumps,
# so this needs neither DIALS nor a msgpack library; what it does need is the
# column type names to be exactly right, since DIALS refuses a name it does not
# know.
add_library(spotfinder_refl STATIC src/refl.cc)
target_include_directories(spotfinder_refl PUBLIC src)
target_link_libraries(spotfinder_refl
    PUBLIC spotfinder_dials_spots
    PRIVATE spotfinder_warnings)

# Reading the three facts a spot list needs out of what dials.import wrote.
add_library(spotfinder_expt STATIC src/expt.cc)
target_include_directories(spotfinder_expt PUBLIC src)
target_link_libraries(spotfinder_expt PRIVATE spotfinder_warnings)

add_executable(dials-metal-find-spots src/find_spots.cc)
target_compile_definitions(dials-metal-find-spots PRIVATE
                           SPOTFINDER_VERSION="${PROJECT_VERSION}")
target_link_libraries(dials-metal-find-spots
    PRIVATE spotfinder_series spotfinder_decompress spotfinder_dext
            spotfinder_dials_spots spotfinder_refl spotfinder_expt
            Threads::Threads spotfinder_warnings)

# The kernels go into the executable rather than a library of their own, so that
# adding more .cu files is a one-line change here.
if(SPOTFINDER_CUDA)
    target_sources(dials-metal-find-spots PRIVATE src/dext_cuda.cu)
    target_compile_definitions(dials-metal-find-spots PRIVATE
                               SPOTFINDER_CUDA SPOTFINDER_GPU)
    target_link_libraries(dials-metal-find-spots
        PRIVATE spotfinder_signal_order spotfinder_dext_gpu)
    # No separable compilation: there is one .cu file and nothing calls across
    # translation units on the device, so the extra nvcc device-link step buys
    # nothing -- and it would be handed the host linker's flags, which it does
    # not understand.
    set_target_properties(dials-metal-find-spots
                          PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
endif()

# Metal, unlike CUDA, goes into a library of its own rather than sources hung on
# the executable, because tests/test_dext_gpu.cc has to link it too: a backend
# only reachable through the tool cannot be diffed against the CPU without a
# detector and a series.
if(SPOTFINDER_METAL)
    add_library(spotfinder_dext_metal STATIC
                src/dext_metal.cc ${spotfinder_metallib_cc})
    target_include_directories(spotfinder_dext_metal PUBLIC src)
    # SYSTEM: metal-cpp is not clean under -Wall -Wextra and it is not ours to
    # fix. The warnings that matter are the ones in src/.
    target_include_directories(spotfinder_dext_metal SYSTEM PUBLIC
                               ${METAL_CPP_INCLUDE_DIR})
    target_link_libraries(spotfinder_dext_metal
        PUBLIC ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK}
        PRIVATE spotfinder_signal_order spotfinder_dext_gpu spotfinder_warnings)

    target_link_libraries(dials-metal-find-spots PRIVATE spotfinder_dext_metal)
    target_compile_definitions(dials-metal-find-spots PRIVATE
                               SPOTFINDER_METAL SPOTFINDER_GPU)
endif()

install(TARGETS dials-metal-find-spots RUNTIME DESTINATION bin)

if(SPOTFINDER_TESTS)
    enable_testing()

    # Not a regression: it pins the domain the 32-bit window sums are exact over,
    # and fails if a change to the kernel size or to the masking eats the
    # headroom.
    add_executable(test_dext_squares tests/test_dext_squares.cc)
    target_link_libraries(test_dext_squares
        PRIVATE spotfinder_dext spotfinder_warnings)
    add_test(NAME dext_squares COMMAND test_dext_squares)

    # No GPU needed: the ordering is host code, and this is where its heap
    # overflow was caught.
    add_executable(test_signal_order tests/test_signal_order.cc)
    target_link_libraries(test_signal_order
        PRIVATE spotfinder_signal_order spotfinder_warnings)
    add_test(NAME signal_order COMMAND test_signal_order)

    # The DIALS side: the grouping, the file, the experiment list, and the three
    # of them together over planted frames. None of these needs a GPU, a server
    # or a detector, which is deliberate -- they are the tests that say whether
    # what dials.index is handed means what it says.
    add_executable(test_dials_spots tests/test_dials_spots.cc)
    target_link_libraries(test_dials_spots
        PRIVATE spotfinder_dials_spots spotfinder_warnings)
    add_test(NAME dials_spots COMMAND test_dials_spots)

    # Not a ctest: it is a measurement to read, not a pass or a fail. It is
    # here because the grouping looks like the expensive stage -- it is the one
    # that cannot be spread over threads -- and at the density a real frame has
    # it is a tenth of a per cent of one. The numbers are in its header.
    add_executable(bench_dials_spots tests/bench_dials_spots.cc)
    target_link_libraries(bench_dials_spots
        PRIVATE spotfinder_dials_spots spotfinder_warnings)

    add_executable(test_refl tests/test_refl.cc)
    target_link_libraries(test_refl PRIVATE spotfinder_refl spotfinder_warnings)
    add_test(NAME refl COMMAND test_refl)

    add_executable(test_expt tests/test_expt.cc)
    target_link_libraries(test_expt PRIVATE spotfinder_expt spotfinder_warnings)
    add_test(NAME expt COMMAND test_expt)

    add_executable(test_dials_end_to_end tests/test_dials_end_to_end.cc)
    target_link_libraries(test_dials_end_to_end
        PRIVATE spotfinder_dext spotfinder_refl spotfinder_warnings)
    add_test(NAME dials_end_to_end COMMAND test_dials_end_to_end)

    # The CPU and the device must agree exactly, so this is only built when there
    # is a device to disagree with. It skips, rather than fails, when the backend
    # is compiled in but the machine has none. Either backend: the CUDA kernels
    # are compiled into the executable rather than a library, so the test needs
    # the .cu too.
    if(SPOTFINDER_CUDA OR SPOTFINDER_METAL)
        set(spotfinder_gpu_test_libs spotfinder_dext spotfinder_signal_order
                                     spotfinder_dext_gpu spotfinder_warnings)
        if(SPOTFINDER_METAL)
            list(APPEND spotfinder_gpu_test_libs spotfinder_dext_metal)
            set(spotfinder_gpu_test_sources "")
        else()
            set(spotfinder_gpu_test_sources src/dext_cuda.cu)
        endif()

        add_executable(test_dext_gpu tests/test_dext_gpu.cc
                       ${spotfinder_gpu_test_sources})
        target_include_directories(test_dext_gpu PRIVATE tests)
        target_link_libraries(test_dext_gpu PRIVATE ${spotfinder_gpu_test_libs})
        add_test(NAME dext_gpu COMMAND test_dext_gpu)
        set_tests_properties(dext_gpu PROPERTIES SKIP_RETURN_CODE 77)

        # Which window is faster is a question about the hardware, so it is
        # answered by measurement. Not a ctest: it takes minutes and its output
        # is a number to read, not a pass or a fail.
        add_executable(bench_dext_gpu tests/bench_dext_gpu.cc
                       ${spotfinder_gpu_test_sources})
        target_include_directories(bench_dext_gpu PRIVATE tests)
        target_link_libraries(bench_dext_gpu
            PRIVATE ${spotfinder_gpu_test_libs} Threads::Threads)
    endif()
endif()
