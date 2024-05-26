// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK:       %[[condition:.*]] = bmodelica.eq
// CHECK:       bmodelica.if (%[[condition]] : !bmodelica.bool) {
// CHECK-NEXT:      %[[value:.*]] = bmodelica.constant #bmodelica<int 1>
// CHECK-NEXT:      bmodelica.variable_set @y, %[[value]]
// CHECK-NEXT:  }

function Test
    input Real x;
    output Real y;
algorithm
    if x == 0 then
        y := 1;
    end if;
end Test;
