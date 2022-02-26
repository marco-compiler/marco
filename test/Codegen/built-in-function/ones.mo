// RUN: marco %s --omc-bypass --emit-modelica-dialect | FileCheck %s

// CHECK-LABEL: @foo
// CHECK-SAME: %arg0 : !modelica.int
// CHECK-SAME: %arg1 : !modelica.int
// CHECK: modelica.ones %arg0, %arg1 : (!modelica.int, !modelica.int) -> !modelica.array<heap, ?x?x!modelica.int>

function foo
    input Integer n1;
    input Integer n2;
    output Real[:] y;

algorithm
    y := ones(n1, n2);
end foo;
