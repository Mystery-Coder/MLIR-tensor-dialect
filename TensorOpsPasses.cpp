#include "TensorOpsPasses.h"

#include "TensorOpsDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
class LowerTransposePattern : public OpRewritePattern<mlir::tensorops::TransposeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::tensorops::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getInput().getType());
    auto outputType = dyn_cast<RankedTensorType>(op.getOutput().getType());
    if (!inputType || !outputType || inputType.getRank() != 2 || outputType.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "expected ranked 2D input/output tensors");

    Location loc = op.getLoc();
    Value input = op.getInput();

    SmallVector<OpFoldResult> outShape;
    outShape.reserve(outputType.getRank());
    for (int64_t i = 0; i < outputType.getRank(); ++i) {
      int64_t dim = outputType.getDimSize(i);
      if (dim == ShapedType::kDynamic)
        outShape.push_back(rewriter.create<tensor::DimOp>(loc, input, i == 0 ? 1 : 0).getResult());
      else
        outShape.push_back(rewriter.getIndexAttr(dim));
    }

    Value init =
        rewriter.create<tensor::EmptyOp>(loc, outShape, outputType.getElementType());
    SmallVector<int64_t> permutation = {1, 0};
    auto transposeOp =
      rewriter.create<linalg::TransposeOp>(loc, input, init, permutation);
    rewriter.replaceOp(op, transposeOp.getResult().front());
    return success();
  }
};

class LowerTensorOpsTransposePass
    : public PassWrapper<LowerTensorOpsTransposePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTensorOpsTransposePass)

  StringRef getArgument() const final { return "lower-tensorops-transpose"; }
  StringRef getDescription() const final {
    return "Lower tensorops.transpose to linalg.transpose";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() final {
    ConversionTarget target(getContext());
    target.addIllegalDialect<mlir::tensorops::TensorOpsDialect>();
    target.addLegalDialect<arith::ArithDialect, BuiltinDialect, func::FuncDialect,
                           linalg::LinalgDialect, tensor::TensorDialect>();

    RewritePatternSet patterns(&getContext());
    patterns.add<LowerTransposePattern>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

namespace mlir::tensorops {
std::unique_ptr<Pass> createLowerTensorOpsTransposePass() {
  return std::make_unique<LowerTensorOpsTransposePass>();
}

void registerTensorOpsPasses() {
  PassRegistration<LowerTensorOpsTransposePass>();
}
} // namespace mlir::tensorops
