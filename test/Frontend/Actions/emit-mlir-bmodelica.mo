// RUN: marco -mc1 -emit-mlir-bmodelica --omc-bypass -o - %s | FileCheck %s

// CHECK: bmodelica.model

model M
end M;
