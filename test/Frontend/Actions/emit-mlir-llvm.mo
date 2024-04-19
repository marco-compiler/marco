// RUN: marco -mc1 -emit-mlir-llvm --omc-bypass -o - %s | FileCheck %s

// CHECK: llvm.mlir.global internal constant @modelName

model M
end M;
