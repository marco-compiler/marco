// RUN: marco %s --omc-bypass --emit-modelica-dialect | FileCheck %s

// CHECK-LABEL: @foo
// CHECK-SAME: %arg0 : !modelica.array<?x!modelica.int>
// CHECK: modelica.product %arg0 : !modelica.array<?x!modelica.int> -> !modelica.int

function foo
    input Integer[:] x;
    output Integer y;

algorithm
    y := product(x);
end foo;