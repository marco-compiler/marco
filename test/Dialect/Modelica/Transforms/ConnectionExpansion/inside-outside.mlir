// RUN: marco-opt %s --modelica-connection-expansion | FileCheck %s

// The same connectors are connected both as inside and as outside
// connectors. The inside and outside elements of a flow variable belong to
// different connection sets, the outside ones being taken negatively, while
// the potential variables are equated only once.
//
// connector Pin
//   Real v;
//   flow Real i;
// end Pin;
// model Inner
//   Pin p, n;
// equation
//   connect(p, n);
// end Inner;
// model InsideOutside
//   Inner c[3];
// equation
//   for k in 1:2 loop
//     connect(c[k].n, c[k + 1].p);
//   end for;
// end InsideOutside;

// CHECK-NOT:   modelica.connect
// CHECK:           modelica.equations {
// CHECK-NEXT:      }
// CHECK-NEXT:      modelica.equations {

// Potential variables: five equations among the six elements.
// CHECK-NEXT:          modelica.for_equation %[[k:.*]] : !modelica.int in {
// CHECK:                   modelica.range {{.*}} -> tensor<2x!modelica.int>
// CHECK:               } do {
// CHECK-NEXT:              modelica.equation {
// CHECK:                       modelica.member %{{.*}}, @p
// CHECK-NEXT:                  %[[lhs:.*]] = modelica.member %{{.*}}, @v
// CHECK:                       modelica.add %[[k]]
// CHECK:                       modelica.member %{{.*}}, @n
// CHECK-NEXT:                  %[[rhs:.*]] = modelica.member %{{.*}}, @v
// CHECK-NEXT:                  modelica.equation_sides %[[lhs]] = %[[rhs]]
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:          modelica.for_equation %[[k:.*]] : !modelica.int in {
// CHECK:                   modelica.range {{.*}} -> tensor<3x!modelica.int>
// CHECK:               } do {
// CHECK-NEXT:              modelica.equation {
// CHECK:                       modelica.member %{{.*}}, @n
// CHECK-NEXT:                  %[[lhs:.*]] = modelica.member %{{.*}}, @v
// CHECK:                       modelica.member %{{.*}}, @p
// CHECK-NEXT:                  %[[rhs:.*]] = modelica.member %{{.*}}, @v
// CHECK-NEXT:                  modelica.equation_sides %[[lhs]] = %[[rhs]]
// CHECK-NEXT:              }
// CHECK-NEXT:          }

// Flow variables of the inside connectors.
// CHECK-NEXT:          modelica.for_equation %[[k:.*]] : !modelica.int in {
// CHECK:                   modelica.range {{.*}} -> tensor<2x!modelica.int>
// CHECK:               } do {
// CHECK-NEXT:              modelica.equation {
// CHECK:                       modelica.member %{{.*}}, @p
// CHECK-NEXT:                  %[[lhs:.*]] = modelica.member %{{.*}}, @i
// CHECK:                       modelica.member %{{.*}}, @n
// CHECK-NEXT:                  %[[rhs:.*]] = modelica.member %{{.*}}, @i
// CHECK-NEXT:                  modelica.add %[[lhs]], %[[rhs]]
// CHECK:                   }
// CHECK-NEXT:          }
// CHECK-NEXT:          modelica.equation {
// CHECK:                   modelica.member %{{.*}}, @p
// CHECK-NEXT:              modelica.member %{{.*}}, @i
// CHECK-NEXT:              modelica.constant #modelica<real 0.000000e+00>
// CHECK-NEXT:              modelica.equation_sides
// CHECK-NEXT:          }
// CHECK-NEXT:          modelica.equation {
// CHECK:                   modelica.member %{{.*}}, @n
// CHECK-NEXT:              modelica.member %{{.*}}, @i
// CHECK-NEXT:              modelica.constant #modelica<real 0.000000e+00>
// CHECK-NEXT:              modelica.equation_sides
// CHECK-NEXT:          }

// Flow variables of the outside connectors.
// CHECK-NEXT:          modelica.for_equation %[[k:.*]] : !modelica.int in {
// CHECK:                   modelica.range {{.*}} -> tensor<3x!modelica.int>
// CHECK:               } do {
// CHECK-NEXT:              modelica.equation {
// CHECK:                       modelica.member %{{.*}}, @p
// CHECK-NEXT:                  %[[pi:.*]] = modelica.member %{{.*}}, @i
// CHECK-NEXT:                  %[[neg:.*]] = modelica.neg %[[pi]]
// CHECK:                       modelica.member %{{.*}}, @n
// CHECK-NEXT:                  %[[ni:.*]] = modelica.member %{{.*}}, @i
// CHECK-NEXT:                  %[[diff:.*]] = modelica.sub %[[neg]], %[[ni]]
// CHECK-NEXT:                  %[[zero:.*]] = modelica.constant #modelica<real 0.000000e+00>
// CHECK-NEXT:                  modelica.equation_sides %[[diff]] = %[[zero]]
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:      }

module {
  modelica.model @InsideOutside {
    modelica.instance @c : tensor<3x!modelica.object<@Inner>> {
      modelica.instance connector @p : !modelica.object<@Pin> {
        modelica.variable @v : !modelica.real
        modelica.variable flow @i : !modelica.real
      }
      modelica.instance connector @n : !modelica.object<@Pin> {
        modelica.variable @v : !modelica.real
        modelica.variable flow @i : !modelica.real
      }
    }
    modelica.equations {
      modelica.for_equation %arg0 : !modelica.int in {
        %0 = modelica.constant #modelica<int 0> : !modelica.int
        %1 = modelica.constant #modelica<int 1> : !modelica.int
        %2 = modelica.range %0 to %1 : (!modelica.int, !modelica.int) -> tensor<2x!modelica.int>
        modelica.yield %2 : tensor<2x!modelica.int>
      } do {
        modelica.connect {
          %0 = modelica.ref @c : tensor<3x!modelica.object<@Inner>>
          %1 = modelica.subscript %0[%arg0] : (tensor<3x!modelica.object<@Inner>>, !modelica.int) -> !modelica.object<@Inner>
          %2 = modelica.member %1, @n : (!modelica.object<@Inner>) -> !modelica.object<@Pin>
          %3 = modelica.ref @c : tensor<3x!modelica.object<@Inner>>
          %4 = modelica.constant #modelica<int 1> : !modelica.int
          %5 = modelica.add %arg0, %4 : (!modelica.int, !modelica.int) -> !modelica.int
          %6 = modelica.subscript %3[%5] : (tensor<3x!modelica.object<@Inner>>, !modelica.int) -> !modelica.object<@Inner>
          %7 = modelica.member %6, @p : (!modelica.object<@Inner>) -> !modelica.object<@Pin>
          modelica.yield %2, %7 : !modelica.object<@Pin>, !modelica.object<@Pin>
        }
      }
      modelica.for_equation %arg0 : !modelica.int in {
        %0 = modelica.constant #modelica<int 0> : !modelica.int
        %1 = modelica.constant #modelica<int 2> : !modelica.int
        %2 = modelica.range %0 to %1 : (!modelica.int, !modelica.int) -> tensor<3x!modelica.int>
        modelica.yield %2 : tensor<3x!modelica.int>
      } do {
        modelica.connect {
          %0 = modelica.ref @c : tensor<3x!modelica.object<@Inner>>
          %1 = modelica.subscript %0[%arg0] : (tensor<3x!modelica.object<@Inner>>, !modelica.int) -> !modelica.object<@Inner>
          %2 = modelica.member %1, @p : (!modelica.object<@Inner>) -> !modelica.object<@Pin>
          %3 = modelica.ref @c : tensor<3x!modelica.object<@Inner>>
          %4 = modelica.subscript %3[%arg0] : (tensor<3x!modelica.object<@Inner>>, !modelica.int) -> !modelica.object<@Inner>
          %5 = modelica.member %4, @n : (!modelica.object<@Inner>) -> !modelica.object<@Pin>
          modelica.yield %2, %5 : !modelica.object<@Pin>, !modelica.object<@Pin>
        } {lhsOutside, rhsOutside}
      }
    }
  }
}
