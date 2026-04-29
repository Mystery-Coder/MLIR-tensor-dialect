#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "TensorOpsDialect.h.inc"
#include "mlir/IR/BuiltinAttributes.h"
#define GET_OP_CLASSES
#include "TensorOps.h.inc"
