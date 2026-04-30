// test/slice.mlir
module {
  func.func @main() -> f32 {
    %0 = tensorops.alloc : memref<4x4xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %val = arith.constant 7.0 : f32
    tensorops.store %val, %0[%c1, %c1] : memref<4x4xf32>
    %slice = tensorops.slice %0[%c1, %c1][%c2, %c2][%c1, %c1]
               : memref<4x4xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
    %result = tensorops.load %slice[%c0, %c0]
               : memref<?x?xf32, strided<[?, ?], offset: ?>>
    return %result : f32
  }
}
