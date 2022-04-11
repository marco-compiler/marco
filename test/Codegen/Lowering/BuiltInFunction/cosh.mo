// RUN: marco %s --omc-bypass --emit-modelica-dialect | FileCheck %s

// CHECK-LABEL: @foo
// CHECK: modelica.cosh
// CHECK-SAME: !modelica.real -> !modelica.real

function foo
    input Real x;
    output Real y;

algorithm
    y := cosh(x);
end foo;
