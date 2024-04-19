// RUN: marco -emit-mlir-modelica --omc-bypass -o - %s | FileCheck %s

// CHECK: modelica.model

model M
end M;
