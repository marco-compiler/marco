// RUN: marco %s --omc-bypass --emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @foo
// CHECK: modelica.max
// CHECK-SAME: (!modelica.real, !modelica.real) -> !modelica.real

function foo
    input Real x;
    input Real y;
    output Real z;
algorithm
    z := max(x, y);
end foo;