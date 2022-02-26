// RUN: marco %s --omc-bypass --emit-modelica-dialect | FileCheck %s

// CHECK-LABEL: @foo
// CHECK-SAME: %arg0 : !modelica.int
// CHECK-SAME: %arg1 : !modelica.int
// CHECK-SAME: %arg2 : !modelica.int
// CHECK: modelica.linspace %arg0, %arg1, %arg2 : (!modelica.int, !modelica.int, !modelica.int) -> !modelica.array<heap, ?x!modelica.real>

function foo
    input Integer start;
    input Integer stop;
    input Integer n;
    output Real[:] y;

algorithm
    y := linspace(start, stop, n);
end foo;
