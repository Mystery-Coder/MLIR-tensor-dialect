
# MLIR Dialect for Tensor Operations

## Description
Define a custom MLIR dialect for multi-dimensional array operations (`allocate`, `load`, `store`, `slice`, `transpose`) and write a lowering pass to convert it to MLIR's standard `memref` dialect.

## Background
MLIR (Multi-Level IR) allows you to define domain-specific intermediate representations with custom types and operations. This assignment mimics how TensorFlow/PyTorch frontends lower to MLIR — you'll define high-level tensor ops, then progressively lower them through MLIR's dialect hierarchy.

The **memref** dialect in MLIR provides operations to allocate, deallocate, and manipulate structured, multi-dimensional memory buffers, acting as a bridge between high-level arrays and low-level pointers. 

The **scf** (Structured Control Flow) dialect defines loop (scf.for, scf.parallel) and conditional (scf.if) constructs that represent structured control flow, acting as a intermediate lowering stage for loops. 



## Objective
Build:
- **Custom dialect** with ops like `tensor.alloc`, `tensor.load`, `tensor.transpose`
- **TableGen definitions** for op syntax and verification
- **Conversion pass** using MLIR's DialectConversion framework to lower your ops to `memref` dialect
- **Simple test program** showing the multi-stage lowering

## Deliverables
- TableGen `.td` file defining your dialect's ops and types
- C++ implementation of `ConversionPattern` for each op
- Test `.mlir` file showing high-level IR → lowered IR transformation
- Report: How does MLIR's progressive lowering differ from LLVM's single-IR model?