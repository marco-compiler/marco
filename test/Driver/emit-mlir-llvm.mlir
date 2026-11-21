// RUN: marco -emit-mlir-llvm -o - %s | FileCheck %s

// CHECK: llvm.mlir.global internal constant @modelName

bmodelica.model @M {

}
