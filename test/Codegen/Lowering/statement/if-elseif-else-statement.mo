// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK:       %[[if_condition:.*]] = bmodelica.eq
// CHECK:       bmodelica.if (%[[if_condition]] : !bmodelica.bool) {
// CHECK-NEXT:      %[[if_value:.*]] = bmodelica.constant #bmodelica<int 1>
// CHECK-NEXT:      bmodelica.variable_set @y, %[[if_value]]
// CHECK-NEXT:  } else {
// CHECK:           %[[elseif_1_condition:.*]] = bmodelica.eq
// CHECK:           bmodelica.if (%[[elseif_1_condition]] : !bmodelica.bool) {
// CHECK-NEXT:          %[[elseif_1_value:.*]] = bmodelica.constant #bmodelica<int 2>
// CHECK-NEXT:          bmodelica.variable_set @y, %[[elseif_1_value]]
// CHECK-NEXT:      } else {
// CHECK:               %[[elseif_2_condition:.*]] = bmodelica.eq
// CHECK:               bmodelica.if (%[[elseif_2_condition]] : !bmodelica.bool) {
// CHECK-NEXT:              %[[elseif_2_value:.*]] = bmodelica.constant #bmodelica<int 3>
// CHECK-NEXT:              bmodelica.variable_set @y, %[[elseif_2_value]]
// CHECK-NEXT:          } else {
// CHECK-NEXT:              %[[else_value:.*]] = bmodelica.constant #bmodelica<int 4>
// CHECK-NEXT:              bmodelica.variable_set @y, %[[else_value]]
// CHECK-NEXT:          }
// CHECK-NEXT:      }
// CHECK-NEXT:  }

function Test
    input Real x;
    output Real y;
algorithm
    if x == 0 then
        y := 1;
    elseif x == 1 then
        y := 2;
    elseif x == 2 then
        y := 3;
    else
        y := 4;
    end if;
end Test;
