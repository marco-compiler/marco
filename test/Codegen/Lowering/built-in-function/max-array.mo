// RUN: marco %s --omc-bypass --emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @foo
// CHECK: modelica.max
// CHECK-SAME: !modelica.array<?x?x!modelica.real> -> !modelica.real

function foo
    input Real[:,:] x;
    output Real y;
algorithm
    y := max(x);
end foo;
