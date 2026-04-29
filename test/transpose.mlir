func.func @transpose2d(%arg0: memref<2x3xf32>) -> memref<3x2xf32> {
  %0 = tensorops.transpose %arg0 : memref<2x3xf32> -> memref<3x2xf32>
  return %0 : memref<3x2xf32>
}
