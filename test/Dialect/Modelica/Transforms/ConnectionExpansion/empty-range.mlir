// RUN: marco-opt %s --modelica-connection-expansion | FileCheck %s

// Connect-equations within for-equations with empty ranges connect nothing:
// only the unconnected flow variables are set to zero.
//
// model EmptyRange
//   TwoPin r[2];
// equation
//   for k in 1:0 loop
//     connect(r[k].n, r[k + 1].p);
//   end for;
// end EmptyRange;

// CHECK:       modelica.model @EmptyRange
// CHECK-NOT:       modelica.connect
// CHECK:           modelica.equations {
// CHECK-NEXT:      }
// CHECK-NEXT:      modelica.equations {
// CHECK-NEXT:          modelica.for_equation %{{.*}} : !modelica.int in {
// CHECK:                   modelica.range {{.*}} -> tensor<2x!modelica.int>
// CHECK:               } do {
// CHECK-NEXT:              modelica.equation {
// CHECK:                       modelica.member %{{.*}}, @p
// CHECK-NEXT:                  modelica.member %{{.*}}, @i
// CHECK:                   }
// CHECK-NEXT:          }
// CHECK-NEXT:          modelica.for_equation %{{.*}} : !modelica.int in {
// CHECK:                   modelica.range {{.*}} -> tensor<2x!modelica.int>
// CHECK:               } do {
// CHECK-NEXT:              modelica.equation {
// CHECK:                       modelica.member %{{.*}}, @n
// CHECK-NEXT:                  modelica.member %{{.*}}, @i
// CHECK:                   }
// CHECK-NEXT:          }
// CHECK-NEXT:      }
// CHECK-NEXT:  }

module {
  modelica.model @EmptyRange {
    modelica.instance @r : tensor<2x!modelica.object<@TwoPin>> {
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
        %1 = modelica.constant #modelica<int -1> : !modelica.int
        %2 = modelica.range %0 to %1 : (!modelica.int, !modelica.int) -> tensor<0x!modelica.int>
        modelica.yield %2 : tensor<0x!modelica.int>
      } do {
        modelica.connect {
          %0 = modelica.ref @r : tensor<2x!modelica.object<@TwoPin>>
          %1 = modelica.subscript %0[%arg0] : (tensor<2x!modelica.object<@TwoPin>>, !modelica.int) -> !modelica.object<@TwoPin>
          %2 = modelica.member %1, @n : (!modelica.object<@TwoPin>) -> !modelica.object<@Pin>
          %3 = modelica.ref @r : tensor<2x!modelica.object<@TwoPin>>
          %4 = modelica.constant #modelica<int 1> : !modelica.int
          %5 = modelica.add %arg0, %4 : (!modelica.int, !modelica.int) -> !modelica.int
          %6 = modelica.subscript %3[%5] : (tensor<2x!modelica.object<@TwoPin>>, !modelica.int) -> !modelica.object<@TwoPin>
          %7 = modelica.member %6, @p : (!modelica.object<@TwoPin>) -> !modelica.object<@Pin>
          modelica.yield %2, %7 : !modelica.object<@Pin>, !modelica.object<@Pin>
        }
      }
    }
  }
}
