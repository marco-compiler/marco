// RUN: marco %s --omc-bypass --emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @foo
// CHECK: modelica.sign
// CHECK-SAME: !modelica.real -> !modelica.int

function foo
    input Real x;
    output Real y;
algorithm
    y := sign(x);
end foo;