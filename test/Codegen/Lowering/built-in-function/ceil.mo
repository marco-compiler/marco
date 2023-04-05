// RUN: marco %s --omc-bypass --emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @foo
// CHECK: modelica.ceil
// CHECK-SAME: !modelica.real -> !modelica.real

function foo
    input Real x;
    output Real y;
algorithm
    y := ceil(x);
end foo;
