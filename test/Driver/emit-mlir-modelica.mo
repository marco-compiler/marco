// RUN: marco -emit-mlir-modelica --omc-bypass -o - %s | FileCheck %s

// CHECK: bmodelica.model

model M
end M;
