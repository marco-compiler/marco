// RUN: marco %s --omc-bypass --emit-modelica-dialect | FileCheck %s

// CHECK-LABEL: @foo
// CHECK-SAME: %arg0 : !modelica.real
// CHECK-SAME: %arg1 : !modelica.real
// CHECK: modelica.atan2 %arg0, %arg1 : (!modelica.real, !modelica.real) -> !modelica.real

function foo
    input Real y;
    input Real x;
    output Real z;

algorithm
    z := atan2(y, x);
end foo;
