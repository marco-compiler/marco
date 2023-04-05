// RUN: marco %s --omc-bypass --emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @foo

// CHECK: modelica.variable @x : !modelica.variable<?x!modelica.int, input>
// CHECK: modelica.variable @y : !modelica.variable<!modelica.int, output>
// CHECK: modelica.variable @z : !modelica.variable<2x!modelica.int>

function foo
    input Integer[:] x;
    output Integer y;
protected
    Integer[2] z;
algorithm
end foo;
