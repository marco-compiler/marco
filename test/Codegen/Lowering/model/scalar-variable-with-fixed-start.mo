// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @Test
// CHECK:       bmodelica.start @x {
// CHECK-NEXT:      %[[value:.*]] = bmodelica.constant #bmodelica.int<5>
// CHECK-NEXT:      bmodelica.yield %[[value]] : !bmodelica.int
// CHECK-NEXT:  } {each = false, fixed = true}

model Test
    Real x(start = 5, fixed = true);
end Test;
