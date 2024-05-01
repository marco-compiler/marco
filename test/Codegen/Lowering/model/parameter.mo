// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @Test
// CHECK: bmodelica.variable @n : !bmodelica.variable<!bmodelica.int, parameter>

model Test
    parameter Integer n;
end Test;
