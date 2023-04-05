// RUN: marco %s --omc-bypass --emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @foo
// CHECK: modelica.linspace
// CHECK-SAME: (!modelica.int, !modelica.int, !modelica.int) -> !modelica.array<?x!modelica.real>

function foo
    input Integer start;
    input Integer stop;
    input Integer n;
    output Real[:] y;
algorithm
    y := linspace(start, stop, n);
end foo;
