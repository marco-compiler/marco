// RUN: marco -mc1 -emit-mlir-modelica --omc-bypass -o - %s | FileCheck %s

// CHECK: bmodelica.model

model M
end M;
