// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @Test
// CHECK:       bmodelica.start @x {
// CHECK-NEXT:      %[[value:.*]] = bmodelica.constant #bmodelica.int<5>
// CHECK-NEXT:      bmodelica.yield %[[value]] : !bmodelica.int
// CHECK-NEXT:  } {each = true, fixed = false}

model Test
    Real[3] x(each start = 5);
end Test;
