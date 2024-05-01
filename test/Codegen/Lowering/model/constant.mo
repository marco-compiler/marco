// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @Test
// CHECK: bmodelica.variable @n : !bmodelica.variable<!bmodelica.int, constant>

model Test
    constant Integer n;
end Test;
