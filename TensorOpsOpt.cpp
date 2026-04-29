// TensorOpsOpt.cpp
#include "TensorOpsDialect.h"
#include "TensorOpsPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  registry.insert<
      mlir::arith::ArithDialect,
      mlir::func::FuncDialect,
      mlir::memref::MemRefDialect,
      mlir::scf::SCFDialect,
      mlir::tensorops::TensorOpsDialect>();

  mlir::tensorops::registerTensorOpsPasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(
          argc,
          argv,
          "TensorOps optimizer driver\n",
          registry));
}
