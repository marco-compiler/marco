// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK:       %[[condition:.*]] = modelica.eq
// CHECK:       modelica.if (%[[condition]] : !modelica.bool) {
// CHECK-NEXT:      %[[value:.*]] = modelica.constant #modelica.int<1>
// CHECK-NEXT:      modelica.variable_set @y, %[[value]]
// CHECK-NEXT:  }

function Test
    input Real x;
    output Real y;
algorithm
    if x == 0 then
        y := 1;
    end if;
end Test;
