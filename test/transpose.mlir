func.func @transpose2d(%arg0: tensor<2x3xf32>) -> tensor<3x2xf32> {
  %0 = tensorops.transpose %arg0 : tensor<2x3xf32> -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}
