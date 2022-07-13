// RUN: marco %s --omc-bypass --emit-modelica-dialect | FileCheck %s

// CHECK-LABEL: @foo
// CHECK: modelica.log
// CHECK-SAME: !modelica.real -> !modelica.real

function foo
    input Real x;
    output Real y;
algorithm
    y := log(x);
end foo;
