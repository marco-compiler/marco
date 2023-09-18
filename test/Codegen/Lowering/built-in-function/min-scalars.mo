// RUN: marco %s --omc-bypass --emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @foo
// CHECK: modelica.min
// CHECK-SAME: (!modelica.real, !modelica.real) -> !modelica.real

function foo
    input Real x;
    input Real y;
    output Real z;
algorithm
    z := min(x, y);
end foo;
