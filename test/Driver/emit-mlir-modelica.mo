// RUN: marco -emit-mlir-modelica -o - %s | FileCheck %s

// CHECK:       modelica.model @M {
// CHECK:           modelica.component @x : @Real

model M
    Real x;
equation
    x = time;
end M;
