// TensorOpsPasses.cpp
#include "TensorOpsPasses.h"
#include "TensorOpsDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

// ===================================
// Lower tensorops.alloc
// ===================================
class LowerAllocPattern
    : public OpRewritePattern<mlir::tensorops::AllocOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
      mlir::tensorops::AllocOp op,
      PatternRewriter &rewriter) const override {

    auto type = dyn_cast<MemRefType>(op.getResult().getType());
    if (!type)
      return failure();

    Value alloc =
        rewriter.create<memref::AllocOp>(op.getLoc(), type);

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

// ===================================
// Lower tensorops.load
// ===================================
class LowerLoadPattern
    : public OpRewritePattern<mlir::tensorops::LoadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
      mlir::tensorops::LoadOp op,
      PatternRewriter &rewriter) const override {

    Value value =
        rewriter.create<memref::LoadOp>(
            op.getLoc(),
            op.getInput(),
            op.getIndices());

    rewriter.replaceOp(op, value);
    return success();
  }
};

// ===================================
// Lower tensorops.transpose
// ===================================
class LowerTransposePattern
    : public OpRewritePattern<mlir::tensorops::TransposeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
      mlir::tensorops::TransposeOp op,
      PatternRewriter &rewriter) const override {

    auto inputType =
        dyn_cast<MemRefType>(op.getInput().getType());

    auto outputType =
        dyn_cast<MemRefType>(op.getOutput().getType());

    if (!inputType || !outputType)
      return failure();

    if (inputType.getRank() != 2 ||
        outputType.getRank() != 2)
      return failure();

    Location loc = op.getLoc();
    Value input = op.getInput();

    auto getDimValue = [&](int64_t dim) -> Value {
      int64_t staticDim = inputType.getDimSize(dim);

      if (staticDim != ShapedType::kDynamic)
        return rewriter.create<arith::ConstantIndexOp>(
            loc, staticDim);

      return rewriter.create<memref::DimOp>(
          loc, input, dim);
    };

    Value dim0 = getDimValue(0);
    Value dim1 = getDimValue(1);

    SmallVector<Value> dynamicSizes;

    for (int64_t i = 0; i < outputType.getRank(); ++i) {
      if (outputType.getDimSize(i) ==
          ShapedType::kDynamic) {
        dynamicSizes.push_back(i == 0 ? dim1 : dim0);
      }
    }

    Value output =
        rewriter.create<memref::AllocOp>(
            loc, outputType, dynamicSizes);

    Value c0 =
        rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 =
        rewriter.create<arith::ConstantIndexOp>(loc, 1);

    auto outer =
        rewriter.create<scf::ForOp>(
            loc, c0, dim0, c1);

    {
      OpBuilder::InsertionGuard guard(rewriter);

      rewriter.setInsertionPointToStart(
          outer.getBody());

      Value i = outer.getInductionVar();

      auto inner =
          rewriter.create<scf::ForOp>(
              loc, c0, dim1, c1);

      rewriter.setInsertionPointToStart(
          inner.getBody());

      Value j = inner.getInductionVar();

      Value inVal =
          rewriter.create<memref::LoadOp>(
              loc, input,
              ValueRange{i, j});

      rewriter.create<memref::StoreOp>(
          loc, inVal, output,
          ValueRange{j, i});
    }

    rewriter.replaceOp(op, output);
    return success();
  }
};

// ===================================
// Pass
// ===================================
class LowerTensorOpsPass
    : public PassWrapper<
          LowerTensorOpsPass,
          OperationPass<ModuleOp>> {

public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      LowerTensorOpsPass)

  StringRef getArgument() const final {
    return "lower-tensorops";
  }

  StringRef getDescription() const final {
    return "Lower tensorops dialect to memref";
  }

  void getDependentDialects(
      DialectRegistry &registry) const override {

    registry.insert<
        arith::ArithDialect,
        func::FuncDialect,
        memref::MemRefDialect,
        scf::SCFDialect>();
  }

  void runOnOperation() final {
    ConversionTarget target(getContext());

    target.addIllegalDialect<
        mlir::tensorops::TensorOpsDialect>();

    target.addLegalDialect<
        arith::ArithDialect,
        func::FuncDialect,
        memref::MemRefDialect,
        scf::SCFDialect>();

    RewritePatternSet patterns(&getContext());

    patterns.add<
        LowerAllocPattern,
        LowerLoadPattern,
        LowerTransposePattern>(
        &getContext());

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir::tensorops {

std::unique_ptr<Pass> createLowerTensorOpsPass() {
  return std::make_unique<LowerTensorOpsPass>();
}

void registerTensorOpsPasses() {
  PassRegistration<LowerTensorOpsPass>();
}

} // namespace mlir::tensorops
