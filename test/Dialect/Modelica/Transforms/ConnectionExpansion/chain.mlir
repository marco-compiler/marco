// RUN: marco-opt %s --modelica-connection-expansion | FileCheck %s

// Connections of arrays of instances. The equations of
// the connection sets are grouped into for-equations, and the unconnected
// flow variables are set to zero.
//
// connector Pin
//   Real v;
//   flow Real i;
// end Pin;
// model TwoPin
//   Pin p, n;
// equation
//   p.i + n.i = 0;
// end TwoPin;
// model Chain
//   parameter Integer N = 4;
//   TwoPin c[N];
//   Pin p;
// equation
//   for i in 1:N - 1 loop
//     connect(c[i].n, c[i + 1].p);
//   end for;
//   connect(p, c[1].p);
//   connect(c[N].n, c[1].p);
// end Chain;

// CHECK-NOT:   modelica.connect

// Potential variables.
// CHECK:           modelica.equations {
// CHECK:           modelica.equations {
// CHECK-NEXT:          modelica.for_equation %[[j:.*]] : !modelica.int in {
// CHECK-NEXT:              %[[from:.*]] = modelica.constant #modelica<int 1>
// CHECK-NEXT:              %[[to:.*]] = modelica.constant #modelica<int 3>
// CHECK-NEXT:              %[[range:.*]] = modelica.range %[[from]] to %[[to]] {{.*}} -> tensor<3x!modelica.int>
// CHECK-NEXT:              modelica.yield %[[range]]
// CHECK-NEXT:          } do {
// CHECK-NEXT:              modelica.equation {
// CHECK-NEXT:                  %[[c:.*]] = modelica.ref @c
// CHECK-NEXT:                  %[[cj:.*]] = modelica.subscript %[[c]][%[[j]]]
// CHECK-NEXT:                  %[[p:.*]] = modelica.member %[[cj]], @p
// CHECK-NEXT:                  %[[pv:.*]] = modelica.member %[[p]], @v
// CHECK-NEXT:                  %[[minusOne:.*]] = modelica.constant #modelica<int -1>
// CHECK-NEXT:                  %[[previous:.*]] = modelica.add %[[j]], %[[minusOne]]
// CHECK-NEXT:                  %[[c:.*]] = modelica.ref @c
// CHECK-NEXT:                  %[[cprevious:.*]] = modelica.subscript %[[c]][%[[previous]]]
// CHECK-NEXT:                  %[[n:.*]] = modelica.member %[[cprevious]], @n
// CHECK-NEXT:                  %[[nv:.*]] = modelica.member %[[n]], @v
// CHECK-NEXT:                  modelica.equation_sides %[[pv]] = %[[nv]]
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:          modelica.equation {
// CHECK-NEXT:              %[[first:.*]] = modelica.constant #modelica<int 0>
// CHECK-NEXT:              %[[c:.*]] = modelica.ref @c
// CHECK-NEXT:              %[[cfirst:.*]] = modelica.subscript %[[c]][%[[first]]]
// CHECK-NEXT:              %[[p:.*]] = modelica.member %[[cfirst]], @p
// CHECK-NEXT:              %[[pv:.*]] = modelica.member %[[p]], @v
// CHECK-NEXT:              %[[last:.*]] = modelica.constant #modelica<int 3>
// CHECK-NEXT:              %[[c:.*]] = modelica.ref @c
// CHECK-NEXT:              %[[clast:.*]] = modelica.subscript %[[c]][%[[last]]]
// CHECK-NEXT:              %[[n:.*]] = modelica.member %[[clast]], @n
// CHECK-NEXT:              %[[nv:.*]] = modelica.member %[[n]], @v
// CHECK-NEXT:              modelica.equation_sides %[[pv]] = %[[nv]]
// CHECK-NEXT:          }
// CHECK-NEXT:          modelica.equation {
// CHECK-NEXT:              %[[p:.*]] = modelica.ref @p
// CHECK-NEXT:              %[[pv:.*]] = modelica.member %[[p]], @v
// CHECK-NEXT:              %[[last:.*]] = modelica.constant #modelica<int 3>
// CHECK-NEXT:              %[[c:.*]] = modelica.ref @c
// CHECK-NEXT:              %[[clast:.*]] = modelica.subscript %[[c]][%[[last]]]
// CHECK-NEXT:              %[[n:.*]] = modelica.member %[[clast]], @n
// CHECK-NEXT:              %[[nv:.*]] = modelica.member %[[n]], @v
// CHECK-NEXT:              modelica.equation_sides %[[pv]] = %[[nv]]
// CHECK-NEXT:          }

// Flow variables, with the outside connector taken negatively.
// CHECK-NEXT:          modelica.for_equation %[[j:.*]] : !modelica.int in {
// CHECK:                   modelica.range {{.*}} -> tensor<3x!modelica.int>
// CHECK:               } do {
// CHECK-NEXT:              modelica.equation {
// CHECK:                       %[[pi:.*]] = modelica.member %{{.*}}, @i
// CHECK:                       %[[ni:.*]] = modelica.member %{{.*}}, @i
// CHECK-NEXT:                  %[[sum:.*]] = modelica.add %[[pi]], %[[ni]]
// CHECK-NEXT:                  %[[zero:.*]] = modelica.constant #modelica<real 0.000000e+00>
// CHECK-NEXT:                  modelica.equation_sides %[[sum]] = %[[zero]]
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:          modelica.equation {
// CHECK:                   %[[sum:.*]] = modelica.add
// CHECK-NEXT:              %[[p:.*]] = modelica.ref @p
// CHECK-NEXT:              %[[pi:.*]] = modelica.member %[[p]], @i
// CHECK-NEXT:              %[[diff:.*]] = modelica.sub %[[sum]], %[[pi]]
// CHECK-NEXT:              %[[zero:.*]] = modelica.constant #modelica<real 0.000000e+00>
// CHECK-NEXT:              modelica.equation_sides %[[diff]] = %[[zero]]
// CHECK-NEXT:          }

// Unconnected flow variable of the outside connector, which is also an
// inside connector of the model.
// CHECK-NEXT:          modelica.equation {
// CHECK-NEXT:              %[[p:.*]] = modelica.ref @p
// CHECK-NEXT:              %[[i:.*]] = modelica.member %[[p]], @i
// CHECK-NEXT:              %[[zero:.*]] = modelica.constant #modelica<real 0.000000e+00>
// CHECK-NEXT:              modelica.equation_sides %[[i]] = %[[zero]]
// CHECK-NEXT:          }
// CHECK-NEXT:      }

module {
  modelica.model @Chain {
    modelica.variable parameter @N : !modelica.int {
      %0 = modelica.constant #modelica<int 4> : !modelica.int
      modelica.bind %0 : !modelica.int
    }
    modelica.instance @c : tensor<4x!modelica.object<@TwoPin>> {
      modelica.instance connector @p : !modelica.object<@Pin> {
        modelica.variable @v : !modelica.real
        modelica.variable flow @i : !modelica.real
      }
      modelica.instance connector @n : !modelica.object<@Pin> {
        modelica.variable @v : !modelica.real
        modelica.variable flow @i : !modelica.real
      }
    }
    modelica.instance connector @p : !modelica.object<@Pin> {
      modelica.variable @v : !modelica.real
      modelica.variable flow @i : !modelica.real
    }
    modelica.equations {
      modelica.for_equation %arg0 : !modelica.int in {
        %0 = modelica.constant #modelica<int 0> : !modelica.int
        %1 = modelica.constant #modelica<int 2> : !modelica.int
        %2 = modelica.range %0 to %1 : (!modelica.int, !modelica.int) -> tensor<3x!modelica.int>
        modelica.yield %2 : tensor<3x!modelica.int>
      } do {
        modelica.connect {
          %0 = modelica.ref @c : tensor<4x!modelica.object<@TwoPin>>
          %1 = modelica.subscript %0[%arg0] : (tensor<4x!modelica.object<@TwoPin>>, !modelica.int) -> !modelica.object<@TwoPin>
          %2 = modelica.member %1, @n : (!modelica.object<@TwoPin>) -> !modelica.object<@Pin>
          %3 = modelica.ref @c : tensor<4x!modelica.object<@TwoPin>>
          %4 = modelica.constant #modelica<int 1> : !modelica.int
          %5 = modelica.add %arg0, %4 : (!modelica.int, !modelica.int) -> !modelica.int
          %6 = modelica.subscript %3[%5] : (tensor<4x!modelica.object<@TwoPin>>, !modelica.int) -> !modelica.object<@TwoPin>
          %7 = modelica.member %6, @p : (!modelica.object<@TwoPin>) -> !modelica.object<@Pin>
          modelica.yield %2, %7 : !modelica.object<@Pin>, !modelica.object<@Pin>
        }
      }
      modelica.connect {
        %0 = modelica.ref @p : !modelica.object<@Pin>
        %1 = modelica.ref @c : tensor<4x!modelica.object<@TwoPin>>
        %2 = modelica.constant #modelica<int 0> : !modelica.int
        %3 = modelica.subscript %1[%2] : (tensor<4x!modelica.object<@TwoPin>>, !modelica.int) -> !modelica.object<@TwoPin>
        %4 = modelica.member %3, @p : (!modelica.object<@TwoPin>) -> !modelica.object<@Pin>
        modelica.yield %0, %4 : !modelica.object<@Pin>, !modelica.object<@Pin>
      } {lhsOutside}
      modelica.connect {
        %0 = modelica.ref @c : tensor<4x!modelica.object<@TwoPin>>
        %1 = modelica.constant #modelica<int 3> : !modelica.int
        %2 = modelica.subscript %0[%1] : (tensor<4x!modelica.object<@TwoPin>>, !modelica.int) -> !modelica.object<@TwoPin>
        %3 = modelica.member %2, @n : (!modelica.object<@TwoPin>) -> !modelica.object<@Pin>
        %4 = modelica.ref @c : tensor<4x!modelica.object<@TwoPin>>
        %5 = modelica.constant #modelica<int 0> : !modelica.int
        %6 = modelica.subscript %4[%5] : (tensor<4x!modelica.object<@TwoPin>>, !modelica.int) -> !modelica.object<@TwoPin>
        %7 = modelica.member %6, @p : (!modelica.object<@TwoPin>) -> !modelica.object<@Pin>
        modelica.yield %3, %7 : !modelica.object<@Pin>, !modelica.object<@Pin>
      }
      modelica.for_equation %arg0 : !modelica.int in {
        %0 = modelica.constant #modelica<int 0> : !modelica.int
        %1 = modelica.constant #modelica<int 3> : !modelica.int
        %2 = modelica.range %0 to %1 : (!modelica.int, !modelica.int) -> tensor<4x!modelica.int>
        modelica.yield %2 : tensor<4x!modelica.int>
      } do {
        modelica.equation {
          %0 = modelica.ref @c : tensor<4x!modelica.object<@TwoPin>>
          %1 = modelica.subscript %0[%arg0] : (tensor<4x!modelica.object<@TwoPin>>, !modelica.int) -> !modelica.object<@TwoPin>
          %2 = modelica.member %1, @p : (!modelica.object<@TwoPin>) -> !modelica.object<@Pin>
          %3 = modelica.member %2, @i : (!modelica.object<@Pin>) -> !modelica.real
          %4 = modelica.ref @c : tensor<4x!modelica.object<@TwoPin>>
          %5 = modelica.subscript %4[%arg0] : (tensor<4x!modelica.object<@TwoPin>>, !modelica.int) -> !modelica.object<@TwoPin>
          %6 = modelica.member %5, @n : (!modelica.object<@TwoPin>) -> !modelica.object<@Pin>
          %7 = modelica.member %6, @i : (!modelica.object<@Pin>) -> !modelica.real
          %8 = modelica.add %3, %7 : (!modelica.real, !modelica.real) -> !modelica.real
          %9 = modelica.constant #modelica<int 0> : !modelica.int
          modelica.equation_sides %8 = %9 : !modelica.real = !modelica.int
        }
      }
    }
  }
}
