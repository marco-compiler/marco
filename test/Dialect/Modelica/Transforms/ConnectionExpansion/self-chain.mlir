// RUN: marco-opt %s --modelica-connection-expansion | FileCheck %s

// Array connected to itself at shifted indices: all its elements belong to
// the same connection set, whose flow variables are summed by a reduction.
//
// connector Pin
//   Real v;
//   flow Real i;
// end Pin;
// model C
//   Pin x[5];
// end C;
// model SelfChain
//   C c;
// equation
//   for k in 1:4 loop
//     connect(c.x[k], c.x[k + 1]);
//   end for;
// end SelfChain;

// CHECK-NOT:   modelica.connect
// CHECK:           modelica.equations {
// CHECK-NEXT:      }
// CHECK-NEXT:      modelica.equations {

// Potential variables.
// CHECK-NEXT:          modelica.for_equation %[[k:.*]] : !modelica.int in {
// CHECK-DAG:               %[[from:.*]] = modelica.constant #modelica<int 1>
// CHECK-DAG:               %[[to:.*]] = modelica.constant #modelica<int 4>
// CHECK:                   modelica.range %[[from]] to %[[to]] {{.*}} -> tensor<4x!modelica.int>
// CHECK:               } do {
// CHECK-NEXT:              modelica.equation {
// CHECK:                       modelica.member %{{.*}}, @v
// CHECK:                       modelica.member %{{.*}}, @v
// CHECK-NEXT:                  modelica.equation_sides
// CHECK-NEXT:              }
// CHECK-NEXT:          }

// Flow variables.
// CHECK-NEXT:          modelica.equation {
// CHECK-DAG:               %[[from:.*]] = modelica.constant #modelica<int 0>
// CHECK-DAG:               %[[to:.*]] = modelica.constant #modelica<int 4>
// CHECK:                   %[[range:.*]] = modelica.range %[[from]] to %[[to]] {{.*}} -> tensor<5x!modelica.int>
// CHECK-NEXT:              %[[sum:.*]] = modelica.reduction sum (%[[j:.*]] in %[[range]] : tensor<5x!modelica.int>) -> !modelica.real {
// CHECK-NEXT:                  %[[c:.*]] = modelica.ref @c
// CHECK-NEXT:                  %[[x:.*]] = modelica.member %[[c]], @x
// CHECK-NEXT:                  %[[xj:.*]] = modelica.subscript %[[x]][%[[j]]]
// CHECK-NEXT:                  %[[i:.*]] = modelica.member %[[xj]], @i
// CHECK-NEXT:                  modelica.yield %[[i]]
// CHECK-NEXT:              }
// CHECK-NEXT:              %[[zero:.*]] = modelica.constant #modelica<real 0.000000e+00>
// CHECK-NEXT:              modelica.equation_sides %[[sum]] = %[[zero]]
// CHECK-NEXT:          }
// CHECK-NEXT:      }

module {
  modelica.model @SelfChain {
    modelica.instance @c : !modelica.object<@C> {
      modelica.instance connector @x : tensor<5x!modelica.object<@Pin>> {
        modelica.variable @v : !modelica.real
        modelica.variable flow @i : !modelica.real
      }
    }
    modelica.equations {
      modelica.for_equation %arg0 : !modelica.int in {
        %0 = modelica.constant #modelica<int 0> : !modelica.int
        %1 = modelica.constant #modelica<int 3> : !modelica.int
        %2 = modelica.range %0 to %1 : (!modelica.int, !modelica.int) -> tensor<4x!modelica.int>
        modelica.yield %2 : tensor<4x!modelica.int>
      } do {
        modelica.connect {
          %0 = modelica.ref @c : !modelica.object<@C>
          %1 = modelica.member %0, @x : (!modelica.object<@C>) -> tensor<5x!modelica.object<@Pin>>
          %2 = modelica.subscript %1[%arg0] : (tensor<5x!modelica.object<@Pin>>, !modelica.int) -> !modelica.object<@Pin>
          %3 = modelica.ref @c : !modelica.object<@C>
          %4 = modelica.member %3, @x : (!modelica.object<@C>) -> tensor<5x!modelica.object<@Pin>>
          %5 = modelica.constant #modelica<int 1> : !modelica.int
          %6 = modelica.add %arg0, %5 : (!modelica.int, !modelica.int) -> !modelica.int
          %7 = modelica.subscript %4[%6] : (tensor<5x!modelica.object<@Pin>>, !modelica.int) -> !modelica.object<@Pin>
          modelica.yield %2, %7 : !modelica.object<@Pin>, !modelica.object<@Pin>
        }
      }
    }
  }
}
