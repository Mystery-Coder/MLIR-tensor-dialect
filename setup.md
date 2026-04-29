# Environment Setup on WSL Ubuntu

## 1. Update and install dependencies

sudo apt update
sudo apt install -y build-essential clang-18 lld-18 llvm-18 llvm-18-dev llvm-18-tools mlir-18-tools cmake ninja-build git pkg-config

## 2. Verify tool availability

```bash
clang-18 --version
mlir-tblgen-18 --version
mlir-opt-18 --version
llvm-config-18 --version
```

## 3. Configure and build your project

```bash
cd ~/Desktop/CD_LAB/MLIR-tensor-dialect
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
```
## 4. Run your transpose lowering

```bash
./build/tensorops-opt --lower-tensorops test/transpose.mlir
./build/tensorops-opt --lower-tensorops test/alloc_load.mlir

```
