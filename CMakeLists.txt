cmake_minimum_required(VERSION 3.15)
project(TensorOps LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Point to the Ubuntu-installed MLIR
find_package(MLIR REQUIRED CONFIG PATHS "/usr/lib/llvm-18/lib/cmake/mlir")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_BINARY_DIR})
include_directories(${PROJECT_SOURCE_DIR})

# TableGen generation logic
set(LLVM_TARGET_DEFINITIONS TensorOps.td)
mlir_tablegen(TensorOps.h.inc -gen-op-decls)
mlir_tablegen(TensorOps.cpp.inc -gen-op-defs)
mlir_tablegen(TensorOpsDialect.h.inc -gen-dialect-decls -dialect=tensorops)
mlir_tablegen(TensorOpsDialect.cpp.inc -gen-dialect-defs -dialect=tensorops)
add_public_tablegen_target(TensorOpsIncGen)

add_mlir_dialect_library(MLIRTensorOps
	TensorOpsDialect.cpp
	PARTIAL_SOURCES_INTENDED

	DEPENDS
	TensorOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
)

add_mlir_library(MLIRTensorOpsTransforms
	TensorOpsPasses.cpp
	PARTIAL_SOURCES_INTENDED

	LINK_LIBS PUBLIC
	MLIRIR
	MLIRPass
	MLIRTransforms
	MLIRMemRefDialect
	MLIRSCFDialect
	MLIRFuncDialect
	MLIRArithDialect
	MLIRTensorOps
)

add_llvm_executable(tensorops-opt
	TensorOpsOpt.cpp
	PARTIAL_SOURCES_INTENDED
)

llvm_update_compile_flags(tensorops-opt)
target_link_libraries(tensorops-opt PRIVATE
	MLIROptLib
	MLIRTensorOps
	MLIRTensorOpsTransforms
)
mlir_check_all_link_libraries(tensorops-opt)