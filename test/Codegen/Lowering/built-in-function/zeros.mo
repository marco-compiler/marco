// RUN: marco %s --omc-bypass --emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @foo
// CHECK: modelica.zeros
// CHECK-SAME: (!modelica.int, !modelica.int) -> !modelica.array<?x?x!modelica.int>

function foo
    input Integer n1;
    input Integer n2;
    output Real[:] y;
algorithm
    y := zeros(n1, n2);
end foo;
