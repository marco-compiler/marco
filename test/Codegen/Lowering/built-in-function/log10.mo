// RUN: marco %s --omc-bypass --emit-mlir -o %t
// RUN: cat %t | FileCheck %s

// CHECK-LABEL: @foo
// CHECK: modelica.log10
// CHECK-SAME: !modelica.real -> !modelica.real

function foo
    input Real x;
    output Real y;
algorithm
    y := log10(x);
end foo;
