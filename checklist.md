# MLIR Tensor Dialect Development Checklist

> Before you write a single line of logic, you need the "Compiler for Compilers" ready.

## Phase 1: Environment & Setup

- [ ] **LLVM/MLIR Build:** Ensure you have a successful build of llvm-project in WSL2. You need `mlir-tblgen` and `mlir-opt` in your path.
- [ ] **Project Skeleton:** Set up an "Out-of-Tree" CMake project.
    - Tip: Copy the standalone example from `llvm-project/mlir/examples/standalone` as your template.
- [ ] **Dialect Namespace:** Decide on a name (e.g., `CustomArray` or `TensorLab`).

---

## Phase 2: The Blueprint (TableGen .td)

> This is where you define the "Shape" of your language.

- [ ] **Dialect.td:** Define your dialect with a unique prefix (e.g., `custom`).
- [ ] **Types.td:** Define your custom array/tensor type (usually wrapping a `RankedTensorType`).
- [ ] **Operation Definitions:**
    - [ ] **AllocOp:** No inputs, returns your custom tensor type.
    - [ ] **LoadOp/StoreOp:** Handles coordinates and values.
    - [ ] **TransposeOp:** Takes a tensor, returns a tensor with swapped dimensions.
    - [ ] **SliceOp:** Takes a tensor and offsets, returns a sub-tensor.
- [ ] **Assembly Formats:** Define how each op looks in a `.mlir` file so you don't have to write a custom parser.

---

## Phase 3: Boilerplate Implementation (C++)

> Turning the TableGen blueprint into actual C++ classes.

- [ ] **Op Definitions:** Include the TableGen-generated `.inc` files in a `Dialect.cpp` file.
- [ ] **Register Dialect:** Write the C++ code to let MLIR know your dialect exists.
- [ ] **Verifiers:** (Optional but recommended) Add C++ logic to ensure transpose only works on 2D arrays.

---

## Phase 4: The Lowering Pass (The Brain)

> This is where you convert custom ops to memref ops.

- [ ] **Conversion Target:** Define a `ConversionTarget` that marks your custom dialect as illegal and the `memref` and `scf` dialects as legal.
- [ ] **Type Converter:** Write a rule to convert your `custom.tensor` type into a standard `memref<...>` type.
- [ ] **Lowering Patterns:**
    - [ ] **AllocLowering:** Map `custom.alloc` → `memref.alloc`.
    - [ ] **TransposeLowering:** This is the big one.
        - Generate `memref.alloc` for the destination.
        - Use `ImplicitLocOpBuilder` to create nested `affine.for` loops.
        - Inside the loops, create `memref.load` and `memref.store` with swapped indices.
- [ ] **Pass Registration:** Wrap these patterns into an MLIR Pass so it can be called via the command line.

---

## Phase 5: Verification & Testing

> Proving it actually works.

- [ ] **Test Case:** Create `test.mlir` using your high-level ops.
- [ ] **The "Opt" Run:** Run your custom opt tool:

    ```sh
    ./bin/my-opt --lower-to-memref test.mlir -o output.mlir
    ```

- [ ] **The "Eye Test":** Open `output.mlir`.
    - Is the `custom.transpose` gone?
    - Are there `affine.for` loops in its place?
    - Are the indices swapped correctly (e.g., `load[%i, %j]` → `store[%j, %i]`)?

---