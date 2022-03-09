// RUN: marco %s --omc-bypass --emit-modelica-dialect | FileCheck %s

// CHECK-LABEL: @foo
// CHECK-SAME: %arg0 : !modelica.real
// CHECK: modelica.sin %arg0 : !modelica.real -> !modelica.real

function foo
    input Real x;
    output Real y;

algorithm
    y := sin(x);
end foo;
