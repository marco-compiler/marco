// RUN: marco %s --omc-bypass --emit-mlir -o %t
// RUN: cat %t | FileCheck %s

// CHECK:       %[[condition:.*]] = modelica.eq
// CHECK:       modelica.if (%[[condition]] : !modelica.bool) {
// CHECK-NEXT:      %[[if_value:.*]] = modelica.constant #modelica.int<1>
// CHECK-NEXT:      modelica.variable_set @y, %[[if_value]]
// CHECK-NEXT:  } else {
// CHECK-NEXT:      %[[else_value:.*]] = modelica.constant #modelica.int<2>
// CHECK-NEXT:      modelica.variable_set @y, %[[else_value]]
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
