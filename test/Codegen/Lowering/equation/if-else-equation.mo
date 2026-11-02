// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @M
// CHECK:       bmodelica.if_equation if {
// CHECK:           %[[if_cond:.*]] = bmodelica.gt
// CHECK-NEXT:      bmodelica.yield %[[if_cond]]
// CHECK-NEXT:  } then {
// CHECK-NEXT:      bmodelica.equation {
// CHECK:               %[[then_rhs:.*]] = bmodelica.constant #bmodelica<int 1>
// CHECK:               bmodelica.equation_sides
// CHECK-NEXT:      }
// CHECK-NEXT:  } else {
// CHECK-NEXT:      bmodelica.equation {
// CHECK:               %[[else_rhs:.*]] = bmodelica.constant #bmodelica<int 0>
// CHECK:               bmodelica.equation_sides
// CHECK-NEXT:      }
// CHECK-NEXT:  }

model M
    Real x(start = 0, fixed = true);
equation
    if x > 1 then
        x = 1;
    else
        x = 0;
    end if;
end M;
