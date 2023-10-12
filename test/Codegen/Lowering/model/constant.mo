// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @Test
// CHECK: modelica.variable @n : !modelica.variable<!modelica.int, constant>

model Test
    constant Integer n;
end Test;
