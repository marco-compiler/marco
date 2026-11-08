// RUN: marco-opt %s --modelica-expand-connections | FileCheck %s

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
// CHECK-NEXT:          modelica.equation {
// CHECK-DAG:               %[[first:.*]] = modelica.constant #modelica<int 0>
// CHECK-DAG:               %[[last:.*]] = modelica.constant #modelica<int 3>
// CHECK:                   %[[pv:.*]] = modelica.ref @c
// CHECK:                   modelica.subscript %[[pv]][%[[first]]]
// CHECK:                   %[[nv:.*]] = modelica.ref @c
// CHECK:                   modelica.subscript %[[nv]][%[[last]]]
// CHECK:                   modelica.equation_sides
// CHECK:               modelica.for_equation %[[j:.*]] : !modelica.int in {
// CHECK-DAG:               %[[from:.*]] = modelica.constant #modelica<int 0>
// CHECK-DAG:               %[[to:.*]] = modelica.constant #modelica<int 2>
// CHECK:                   modelica.range %[[from]] to %[[to]] {{.*}} -> tensor<3x!modelica.int>
// CHECK:               } do {
// CHECK-NEXT:              modelica.equation {
// CHECK-DAG:                   %[[one:.*]] = modelica.constant #modelica<int 1>
// CHECK-DAG:                   %[[lhsIndex:.*]] = modelica.add %[[one]], %[[j]]
// CHECK-DAG:                   %[[zero:.*]] = modelica.constant #modelica<int 0>
// CHECK-DAG:                   %[[rhsIndex:.*]] = modelica.add %[[zero]], %[[j]]
// CHECK:                       %[[lhs:.*]] = modelica.member %{{.*}}, @p
// CHECK:                       %[[lhsv:.*]] = modelica.member %[[lhs]], @v
// CHECK:                       %[[rhs:.*]] = modelica.member %{{.*}}, @n
// CHECK:                       %[[rhsv:.*]] = modelica.member %[[rhs]], @v
// CHECK:                       modelica.equation_sides %[[lhsv]] = %[[rhsv]]
// CHECK:               modelica.equation {
// CHECK:                   modelica.ref @p
// CHECK:                   modelica.member %{{.*}}, @v

// Flow variables, with the outside connector taken negatively.
// CHECK:               modelica.for_equation
// CHECK:                   modelica.add
// CHECK:                   modelica.equation_sides %{{.*}} = %{{.*}} : !modelica.real = !modelica.real
// CHECK:               modelica.equation {
// CHECK:                   %[[sum:.*]] = modelica.add
// CHECK:                   %[[diff:.*]] = modelica.sub %[[sum]], %{{.*}}
// CHECK:                   %[[zero:.*]] = modelica.constant #modelica<real 0.000000e+00>
// CHECK:                   modelica.equation_sides %[[diff]] = %[[zero]]

// Unconnected flow variable of the outside connector, which is also an
// inside connector of the model.
// CHECK:               modelica.equation {
// CHECK-NEXT:              %[[p:.*]] = modelica.ref @p
// CHECK-NEXT:              %[[i:.*]] = modelica.member %[[p]], @i
// CHECK-NEXT:              %[[zero:.*]] = modelica.constant #modelica<real 0.000000e+00>
// CHECK-NEXT:              modelica.equation_sides %[[i]] = %[[zero]]

module {
  modelica.model @Chain {
    modelica.variable parameter @N : !modelica.int {
      %0 = modelica.constant #modelica<int 4> : !modelica.int
      modelica.bind %0 : !modelica.int
    }
    modelica.instance model @c : tensor<4x!modelica.object<@TwoPin>> {
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
