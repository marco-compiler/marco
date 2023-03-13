// RUN: marco %s --omc-bypass --emit-mlir -o %t
// RUN: cat %t | FileCheck %s

// CHECK-LABEL: @foo

// CHECK: modelica.variable @x : !modelica.member<?x!modelica.int, input>
// CHECK: modelica.variable @y : !modelica.member<!modelica.int, output>
// CHECK: modelica.variable @z : !modelica.member<2x!modelica.int>

function foo
    input Integer[:] x;
    output Integer y;
protected
    Integer[2] z;
algorithm
end foo;
