// RUN: marco %s --omc-bypass --emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @Test
// CHECK: modelica.variable @n : !modelica.variable<!modelica.int, parameter>

model Test
    parameter Integer n;
end Test;
