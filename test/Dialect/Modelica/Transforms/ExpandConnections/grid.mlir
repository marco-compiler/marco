// RUN: marco-opt %s --modelica-expand-connections | FileCheck %s

// Connections of multidimensional arrays of instances are grouped into
// nested for-equations.
//
// model Grid
//   TwoPin g[3, 2];
// equation
//   for i in 1:2 loop
//     for j in 1:2 loop
//       connect(g[i, j].n, g[i + 1, j].p);
//     end for;
//   end for;
// end Grid;

// CHECK-NOT:   modelica.connect
// CHECK:           modelica.equations {
// CHECK:           modelica.equations {
// CHECK-NEXT:          modelica.for_equation %[[i:.*]] : !modelica.int in {
// CHECK:                   tensor<2x!modelica.int>
// CHECK:               } do {
// CHECK-NEXT:              modelica.for_equation %[[j:.*]] : !modelica.int in {
// CHECK:                       tensor<2x!modelica.int>
// CHECK:                   } do {
// CHECK-NEXT:                  modelica.equation {
// CHECK:                           %[[g:.*]] = modelica.ref @g : tensor<3x2x!modelica.object<@TwoPin>>
// CHECK-NEXT:                      modelica.subscript %[[g]][%{{.*}}, %{{.*}}]

// Unconnected flow variables of the first and last rows.
// CHECK:               modelica.for_equation
// CHECK:                   modelica.member %{{.*}}, @p
// CHECK:                   modelica.member %{{.*}}, @i
// CHECK:               modelica.for_equation
// CHECK:                   modelica.member %{{.*}}, @n
// CHECK:                   modelica.member %{{.*}}, @i

module {
  modelica.model @Grid {
    modelica.instance model @g : tensor<3x2x!modelica.object<@TwoPin>> {
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
        modelica.for_equation %arg1 : !modelica.int in {
          %0 = modelica.constant #modelica<int 0> : !modelica.int
          %1 = modelica.constant #modelica<int 1> : !modelica.int
          %2 = modelica.range %0 to %1 : (!modelica.int, !modelica.int) -> tensor<2x!modelica.int>
          modelica.yield %2 : tensor<2x!modelica.int>
        } do {
          modelica.connect {
            %0 = modelica.ref @g : tensor<3x2x!modelica.object<@TwoPin>>
            %1 = modelica.subscript %0[%arg0, %arg1] : (tensor<3x2x!modelica.object<@TwoPin>>, !modelica.int, !modelica.int) -> !modelica.object<@TwoPin>
            %2 = modelica.member %1, @n : (!modelica.object<@TwoPin>) -> !modelica.object<@Pin>
            %3 = modelica.ref @g : tensor<3x2x!modelica.object<@TwoPin>>
            %4 = modelica.constant #modelica<int 1> : !modelica.int
            %5 = modelica.add %arg0, %4 : (!modelica.int, !modelica.int) -> !modelica.int
            %6 = modelica.subscript %3[%5, %arg1] : (tensor<3x2x!modelica.object<@TwoPin>>, !modelica.int, !modelica.int) -> !modelica.object<@TwoPin>
            %7 = modelica.member %6, @p : (!modelica.object<@TwoPin>) -> !modelica.object<@Pin>
            modelica.yield %2, %7 : !modelica.object<@Pin>, !modelica.object<@Pin>
          }
        }
      }
      modelica.for_equation %arg0 : !modelica.int in {
        %0 = modelica.constant #modelica<int 0> : !modelica.int
        %1 = modelica.constant #modelica<int 2> : !modelica.int
        %2 = modelica.range %0 to %1 : (!modelica.int, !modelica.int) -> tensor<3x!modelica.int>
        modelica.yield %2 : tensor<3x!modelica.int>
      } do {
        modelica.for_equation %arg1 : !modelica.int in {
          %0 = modelica.constant #modelica<int 0> : !modelica.int
          %1 = modelica.constant #modelica<int 1> : !modelica.int
          %2 = modelica.range %0 to %1 : (!modelica.int, !modelica.int) -> tensor<2x!modelica.int>
          modelica.yield %2 : tensor<2x!modelica.int>
        } do {
          modelica.equation {
            %0 = modelica.ref @g : tensor<3x2x!modelica.object<@TwoPin>>
            %1 = modelica.subscript %0[%arg0, %arg1] : (tensor<3x2x!modelica.object<@TwoPin>>, !modelica.int, !modelica.int) -> !modelica.object<@TwoPin>
            %2 = modelica.member %1, @p : (!modelica.object<@TwoPin>) -> !modelica.object<@Pin>
            %3 = modelica.member %2, @i : (!modelica.object<@Pin>) -> !modelica.real
            %4 = modelica.ref @g : tensor<3x2x!modelica.object<@TwoPin>>
            %5 = modelica.subscript %4[%arg0, %arg1] : (tensor<3x2x!modelica.object<@TwoPin>>, !modelica.int, !modelica.int) -> !modelica.object<@TwoPin>
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
}
