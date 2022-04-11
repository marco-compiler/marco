// RUN: marco %s --omc-bypass --emit-modelica-dialect | FileCheck %s

// CHECK-LABEL: @foo
// CHECK: modelica.sqrt
// CHECK-SAME: !modelica.real -> !modelica.real

function foo
    input Real x;
    output Real y;

algorithm
    y := sqrt(x);
end foo;
