// RUN: marco-opt %s --modelica-connection-expansion | FileCheck %s

// Long chain of components: the connection sets and their equations are
// computed without enumerating the elements of the arrays, and the equations
// are emitted as for-equations. The only equations outside for-equations are
// the ones of the unconnected flow variables at the two ends of the chain.
// The chain has 100000 components rather than a million, as the assertions
// of the index sets in the debug builds check each of their elements.
//
// model Scaling
//   TwoPin r[100000];
// equation
//   for k in 1:99999 loop
//     connect(r[k].n, r[k + 1].p);
//   end for;
// end Scaling;

// CHECK-NOT:   modelica.connect
// CHECK:           modelica.equations {
// CHECK-NEXT:      }
// CHECK-NEXT:      modelica.equations {

// Potential variables.
// CHECK-NEXT:          modelica.for_equation %{{.*}} : !modelica.int in {
// CHECK:                   modelica.range {{.*}} -> tensor<99999x!modelica.int>
// CHECK:               } do {
// CHECK-NEXT:              modelica.equation {
// CHECK:                       modelica.member %{{.*}}, @v
// CHECK:                       modelica.member %{{.*}}, @v
// CHECK-NEXT:                  modelica.equation_sides
// CHECK-NEXT:              }
// CHECK-NEXT:          }

// Flow variables.
// CHECK-NEXT:          modelica.for_equation %{{.*}} : !modelica.int in {
// CHECK:                   modelica.range {{.*}} -> tensor<99999x!modelica.int>
// CHECK:               } do {
// CHECK-NEXT:              modelica.equation {
// CHECK:                       modelica.add %{{.*}}, %{{.*}} : (!modelica.real, !modelica.real)
// CHECK-NEXT:                  modelica.constant #modelica<real 0.000000e+00>
// CHECK-NEXT:                  modelica.equation_sides
// CHECK-NEXT:              }
// CHECK-NEXT:          }

// Unconnected flow variables at the ends of the chain.
// CHECK-NEXT:          modelica.equation {
// CHECK-NOT:               modelica.add
// CHECK:                   modelica.equation_sides
// CHECK-NEXT:          }
// CHECK-NEXT:          modelica.equation {
// CHECK-NOT:               modelica.add
// CHECK:                   modelica.equation_sides
// CHECK-NEXT:          }
// CHECK-NEXT:      }
// CHECK-NEXT:  }

module {
  modelica.model @Scaling {
    modelica.instance @r : tensor<100000x!modelica.object<@TwoPin>> {
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
        %1 = modelica.constant #modelica<int 99998> : !modelica.int
        %2 = modelica.range %0 to %1 : (!modelica.int, !modelica.int) -> tensor<99999x!modelica.int>
        modelica.yield %2 : tensor<99999x!modelica.int>
      } do {
        modelica.connect {
          %0 = modelica.ref @r : tensor<100000x!modelica.object<@TwoPin>>
          %1 = modelica.subscript %0[%arg0] : (tensor<100000x!modelica.object<@TwoPin>>, !modelica.int) -> !modelica.object<@TwoPin>
          %2 = modelica.member %1, @n : (!modelica.object<@TwoPin>) -> !modelica.object<@Pin>
          %3 = modelica.ref @r : tensor<100000x!modelica.object<@TwoPin>>
          %4 = modelica.constant #modelica<int 1> : !modelica.int
          %5 = modelica.add %arg0, %4 : (!modelica.int, !modelica.int) -> !modelica.int
          %6 = modelica.subscript %3[%5] : (tensor<100000x!modelica.object<@TwoPin>>, !modelica.int) -> !modelica.object<@TwoPin>
          %7 = modelica.member %6, @p : (!modelica.object<@TwoPin>) -> !modelica.object<@Pin>
          modelica.yield %2, %7 : !modelica.object<@Pin>, !modelica.object<@Pin>
        }
      }
    }
  }
}
