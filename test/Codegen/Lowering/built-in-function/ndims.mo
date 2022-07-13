// RUN: marco %s --omc-bypass --emit-modelica-dialect | FileCheck %s

// CHECK-LABEL: @foo
// CHECK: modelica.ndims
// CHECK-SAME: !modelica.array<?x?x!modelica.int> -> !modelica.int

function foo
    input Integer[:,:] x;
    output Integer y;
algorithm
    y := ndims(x);
end foo;
