// RUN: marco %s --omc-bypass --emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @foo
// CHECK: modelica.atan2
// CHECK-SAME: (!modelica.real, !modelica.real) -> !modelica.real

function foo
    input Real y;
    input Real x;
    output Real z;
algorithm
    z := atan2(y, x);
end foo;
