// TensorOps.td
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// 1. Define the Dialect
def TensorOps_Dialect : Dialect {
  let name = "tensorops";
  let summary = "A custom dialect for multi-dimensional array operations.";
  let description = [{
    This dialect provides high-level operations like transpose, slice, and alloc
  }];
  let cppNamespace = "::mlir::tensorops";
}

// 2. Base class
class TensorOps_Op<string mnemonic, list<Trait> traits = []> :
    Op<TensorOps_Dialect, mnemonic, traits>;

// 3. The Transpose Operation
def TransposeOp : TensorOps_Op<"transpose", [Pure]> {
  let summary = "Transpose a 2D tensor.";
  let arguments = (ins RankedTensorOf<[F32]>:$input);
  let results = (outs RankedTensorOf<[F32]>:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}