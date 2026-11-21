// RUN: marco-opt %s --modelica-connection-expansion | FileCheck %s

// Unconnected flow variables are set to zero, also in the absence of
// connect-equations.
//
// connector Pin
//   Real v;
//   flow Real i;
// end Pin;
// model TwoPin
//   Pin p, n;
// end TwoPin;
// model Unconnected
//   TwoPin r[3];
// end Unconnected;

// CHECK:       modelica.model @Unconnected
// CHECK:           modelica.equations {
// CHECK-NEXT:          modelica.for_equation %[[k:.*]] : !modelica.int in {
// CHECK:                   modelica.range {{.*}} -> tensor<3x!modelica.int>
// CHECK:               } do {
// CHECK-NEXT:              modelica.equation {
// CHECK-NEXT:                  %[[r:.*]] = modelica.ref @r
// CHECK-NEXT:                  %[[rk:.*]] = modelica.subscript %[[r]][%[[k]]]
// CHECK-NEXT:                  %[[p:.*]] = modelica.member %[[rk]], @p
// CHECK-NEXT:                  %[[i:.*]] = modelica.member %[[p]], @i
// CHECK-NEXT:                  %[[zero:.*]] = modelica.constant #modelica<real 0.000000e+00>
// CHECK-NEXT:                  modelica.equation_sides %[[i]] = %[[zero]]
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:          modelica.for_equation %[[k:.*]] : !modelica.int in {
// CHECK:                   modelica.range {{.*}} -> tensor<3x!modelica.int>
// CHECK:               } do {
// CHECK-NEXT:              modelica.equation {
// CHECK-NEXT:                  %[[r:.*]] = modelica.ref @r
// CHECK-NEXT:                  %[[rk:.*]] = modelica.subscript %[[r]][%[[k]]]
// CHECK-NEXT:                  %[[n:.*]] = modelica.member %[[rk]], @n
// CHECK-NEXT:                  %[[i:.*]] = modelica.member %[[n]], @i
// CHECK-NEXT:                  %[[zero:.*]] = modelica.constant #modelica<real 0.000000e+00>
// CHECK-NEXT:                  modelica.equation_sides %[[i]] = %[[zero]]
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:      }

module {
  modelica.model @Unconnected {
    modelica.instance @r : tensor<3x!modelica.object<@TwoPin>> {
      modelica.instance connector @p : !modelica.object<@Pin> {
        modelica.variable @v : !modelica.real
        modelica.variable flow @i : !modelica.real
      }
      modelica.instance connector @n : !modelica.object<@Pin> {
        modelica.variable @v : !modelica.real
        modelica.variable flow @i : !modelica.real
      }
    }
  }
}
