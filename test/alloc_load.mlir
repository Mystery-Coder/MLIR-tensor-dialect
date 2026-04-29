// test/alloc_load.mlir
module {
  func.func @main() -> f32 {
    %0 = tensorops.alloc : memref<4x4xf32>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %1 = tensorops.load %0[%c0, %c1] : memref<4x4xf32>

    return %1 : f32
  }
}
