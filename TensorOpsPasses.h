#pragma once

#include <memory>

namespace mlir {
class Pass;

namespace tensorops {
std::unique_ptr<Pass> createLowerTensorOpsTransposePass();
void registerTensorOpsPasses();
} // namespace tensorops
} // namespace mlir
