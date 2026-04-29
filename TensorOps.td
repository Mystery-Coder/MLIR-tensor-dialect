// TensorOps.td
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Dialect
def TensorOps_Dialect : Dialect {
  let name = "tensorops";
  let summary = "Custom dialect for tensor operations";
  let description = [{
    Provides high-level tensor operations such as alloc, load, transpose.
  }];
  let cppNamespace = "::mlir::tensorops";
}

// Base op
class TensorOps_Op<string mnemonic, list<Trait> traits = []> :
    Op<TensorOps_Dialect, mnemonic, traits>;

// =======================
// tensorops.transpose
// =======================
def TransposeOp : TensorOps_Op<"transpose", [Pure]> {
  let summary = "Transpose a 2D tensor";
  let arguments = (ins MemRefOf<[F32]>:$input);
  let results = (outs MemRefOf<[F32]>:$output);

  let assemblyFormat =
    "$input attr-dict `:` type($input) `->` type($output)";
}

// =======================
// tensorops.alloc
// =======================
def AllocOp : TensorOps_Op<"alloc", [Pure]> {
  let summary = "Allocate memref";
  let results = (outs AnyMemRef:$result);

  let assemblyFormat =
    "attr-dict `:` type($result)";
}

// =======================
// tensorops.load
// =======================
def LoadOp : TensorOps_Op<"load", [Pure]> {
  let summary = "Load from memref";

  let arguments = (
    ins AnyMemRef:$input,
    Variadic<Index>:$indices
  );

  let results = (outs F32:$value);

  let assemblyFormat =
    "$input `[` $indices `]` attr-dict `:` type($input)";
}
