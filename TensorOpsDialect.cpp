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

// Custom assembly format for SliceOp:
// tensorops.slice %src[%o0,%o1][%s0,%s1][%str0,%str1] : memref<4x4xf32> to memref<2x2xf32>
ParseResult SliceOp::parse(OpAsmParser &parser,
                            OperationState &result) {
  OpAsmParser::UnresolvedOperand input;
  SmallVector<OpAsmParser::UnresolvedOperand> offsets, sizes, strides;
  Type inputType, resultType;

  if (parser.parseOperand(input))
    return failure();

  auto parseList = [&](SmallVector<OpAsmParser::UnresolvedOperand> &list) {
    return parser.parseCommaSeparatedList(
        OpAsmParser::Delimiter::Square,
        [&]() -> ParseResult {
          return parser.parseOperand(list.emplace_back());
        });
  };

  if (parseList(offsets) || parseList(sizes) || parseList(strides))
    return failure();

  if (parser.parseColonType(inputType) ||
      parser.parseKeyword("to") ||
      parser.parseType(resultType))
    return failure();

  auto memrefType = dyn_cast<MemRefType>(inputType);
  if (!memrefType)
    return failure();

  IndexType idxType = IndexType::get(parser.getContext());
  SmallVector<Type> idxTypes(offsets.size(), idxType);

  if (parser.resolveOperand(input, inputType, result.operands) ||
      parser.resolveOperands(offsets, idxTypes, parser.getNameLoc(), result.operands) ||
      parser.resolveOperands(sizes,   idxTypes, parser.getNameLoc(), result.operands) ||
      parser.resolveOperands(strides, idxTypes, parser.getNameLoc(), result.operands))
    return failure();

  result.addTypes(resultType);
  return success();
}

void SliceOp::print(OpAsmPrinter &p) {
  p << " " << getInput();
  auto printList = [&](ValueRange vals) {
    p << "[";
    llvm::interleaveComma(vals, p);
    p << "]";
  };
  printList(getOffsets());
  printList(getSizes());
  printList(getStrides());
  p << " : " << getInput().getType()
    << " to " << getResult().getType();
}
