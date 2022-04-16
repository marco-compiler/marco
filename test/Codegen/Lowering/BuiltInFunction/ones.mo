// RUN: marco %s --omc-bypass --emit-modelica-dialect | FileCheck %s

// CHECK-LABEL: @foo
// CHECK: modelica.ones
// CHECK-SAME: (!modelica.int, !modelica.int) -> !modelica.array<?x?x!modelica.int>

function foo
    input Integer n1;
    input Integer n2;
    output Real[:] y;

algorithm
    y := ones(n1, n2);
end foo;