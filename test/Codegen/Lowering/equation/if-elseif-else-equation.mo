// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @M
// CHECK:       %[[if_cond:.*]] = bmodelica.gt
// CHECK:       bmodelica.if_equation if(%[[if_cond]] : !bmodelica.bool) {
// CHECK-NEXT:      bmodelica.equation {
// CHECK:               %[[then_rhs:.*]] = bmodelica.constant #bmodelica<int 1>
// CHECK:               bmodelica.equation_sides
// CHECK-NEXT:      }
// CHECK-NEXT:  } else {
// CHECK:           %[[elseif_cond:.*]] = bmodelica.gt
// CHECK:           bmodelica.if_equation if(%[[elseif_cond]] : !bmodelica.bool) {
// CHECK-NEXT:          bmodelica.equation {
// CHECK:                   %[[elseif_rhs:.*]] = bmodelica.constant #bmodelica<int 2>
// CHECK:                   bmodelica.equation_sides
// CHECK-NEXT:          }
// CHECK-NEXT:      } else {
// CHECK-NEXT:          bmodelica.equation {
// CHECK:                   %[[else_rhs:.*]] = bmodelica.constant #bmodelica<int 3>
// CHECK:                   bmodelica.equation_sides
// CHECK-NEXT:          }
// CHECK-NEXT:      }
// CHECK-NEXT:  }

model M
    Real x(start = 0, fixed = true);
equation
    if x > 1 then
        x = 1;
    elseif x > 2 then
        x = 2;
    else
        x = 3;
    end if;
end M;
