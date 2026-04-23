#include "TensorOpsDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::tensorops;

#include "TensorOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "TensorOps.cpp.inc"

void TensorOpsDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TensorOps.cpp.inc"
      >();
}
