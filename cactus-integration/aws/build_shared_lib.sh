#!/bin/bash
# =============================================================================
# Build Cactus as Shared Library for Python FFI
# =============================================================================

set -e

echo "🔨 Building Cactus as shared library..."

cd ~/cactus-integration/cactus/cactus

# Clean previous build
rm -rf build_shared
mkdir -p build_shared
cd build_shared

# Create modified CMakeLists for shared library
cat > CMakeLists_shared.txt << 'EOF'
cmake_minimum_required(VERSION 3.10)
project(CactusShared LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ARM64 optimizations
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a+fp16+simd+dotprod -pthread -Wall -O3 -fPIC")

file(GLOB ENGINE_SOURCES "../engine/*.cpp")
file(GLOB GRAPH_SOURCES "../graph/*.cpp")
file(GLOB KERNEL_SOURCES "../kernel/*.cpp")
file(GLOB FFI_SOURCES "../ffi/*.cpp")
file(GLOB MODEL_SOURCES "../models/*.cpp")

set(COMMON_SOURCES
    ${KERNEL_SOURCES}
    ${GRAPH_SOURCES}
    ${ENGINE_SOURCES}
    ${FFI_SOURCES}
    ${MODEL_SOURCES}
)

# Build as SHARED library for Python ctypes
add_library(cactus_shared SHARED ${COMMON_SOURCES})
target_compile_definitions(cactus_shared PUBLIC PLATFORM_CPU_ONLY=1)

target_include_directories(cactus_shared PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ../engine
    ../graph
    ../kernel
    ../ffi
    ../models
)

set_target_properties(cactus_shared PROPERTIES OUTPUT_NAME "cactus")
EOF

# Build shared library
cmake -f CMakeLists_shared.txt ..
make -j$(nproc)

# Copy to accessible location
mkdir -p ~/cactus-integration/lib
cp libcactus.so ~/cactus-integration/lib/

echo ""
echo "✅ Shared library built!"
echo "   Location: ~/cactus-integration/lib/libcactus.so"
