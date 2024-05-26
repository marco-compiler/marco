// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK:       %[[condition:.*]] = bmodelica.eq
// CHECK:       bmodelica.if (%[[condition]] : !bmodelica.bool) {
// CHECK-NEXT:      %[[if_value:.*]] = bmodelica.constant #bmodelica<int 1>
// CHECK-NEXT:      bmodelica.variable_set @y, %[[if_value]]
// CHECK-NEXT:  } else {
// CHECK-NEXT:      %[[else_value:.*]] = bmodelica.constant #bmodelica<int 2>
// CHECK-NEXT:      bmodelica.variable_set @y, %[[else_value]]
// CHECK-NEXT:  }

function Test
    input Real x;
    output Real y;
algorithm
    if x == 0 then
        y := 1;
    else
        y := 2;
    end if;
end Test;
