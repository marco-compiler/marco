// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @foo

// CHECK: bmodelica.variable @x : !bmodelica.variable<?x!bmodelica.int, input>
// CHECK: bmodelica.variable @y : !bmodelica.variable<!bmodelica.int, output>
// CHECK: bmodelica.variable @z : !bmodelica.variable<2x!bmodelica.int>

function foo
    input Integer[:] x;
    output Integer y;
protected
    Integer[2] z;
algorithm
end foo;
