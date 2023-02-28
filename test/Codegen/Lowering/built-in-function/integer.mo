// RUN: marco %s --omc-bypass --emit-mlir -o %t
// RUN: cat %t | FileCheck %s

// CHECK-LABEL: @foo
// CHECK: modelica.integer
// CHECK-SAME: !modelica.real -> !modelica.int

function foo
    input Real x;
    output Real y;
algorithm
    y := integer(x);
end foo;
