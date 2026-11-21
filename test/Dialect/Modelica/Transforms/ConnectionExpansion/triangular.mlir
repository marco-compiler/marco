// RUN: marco-opt %s --modelica-connection-expansion | FileCheck %s

// For-equations whose range depends on the iterators of the enclosing ones.
// Only the elements of the lower triangle are connected.
//
// model Triangular
//   C a[3, 3], b[3, 3];
// equation
//   for i in 1:3 loop
//     for j in 1:i loop
//       connect(a[i, j].p, b[i, j].p);
//     end for;
//   end for;
// end Triangular;

// CHECK-NOT:   modelica.connect
// CHECK:           modelica.equations {
// CHECK-NEXT:      }
// CHECK-NEXT:      modelica.equations {

// Potential variables of the lower triangle, one family per row.
// CHECK-NEXT:          modelica.equation {
// CHECK-NEXT:              %[[zero:.*]] = modelica.constant #modelica<int 0>
// CHECK-NEXT:              %[[zero2:.*]] = modelica.constant #modelica<int 0>
// CHECK-NEXT:              %[[b:.*]] = modelica.ref @b
// CHECK-NEXT:              modelica.subscript %[[b]][%[[zero]], %[[zero2]]]
// CHECK:                   modelica.equation_sides
// CHECK-NEXT:          }
// CHECK-NEXT:          modelica.for_equation %[[j:.*]] : !modelica.int in {
// CHECK:                   modelica.range {{.*}} -> tensor<2x!modelica.int>
// CHECK:               } do {
// CHECK-NEXT:              modelica.equation {
// CHECK-NEXT:                  %[[one:.*]] = modelica.constant #modelica<int 1>
// CHECK-NEXT:                  %[[b:.*]] = modelica.ref @b
// CHECK-NEXT:                  modelica.subscript %[[b]][%[[one]], %[[j]]]
// CHECK:                       modelica.equation_sides
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:          modelica.for_equation %[[j:.*]] : !modelica.int in {
// CHECK:                   modelica.range {{.*}} -> tensor<3x!modelica.int>
// CHECK:               } do {
// CHECK-NEXT:              modelica.equation {
// CHECK-NEXT:                  %[[two:.*]] = modelica.constant #modelica<int 2>
// CHECK-NEXT:                  %[[b:.*]] = modelica.ref @b
// CHECK-NEXT:                  modelica.subscript %[[b]][%[[two]], %[[j]]]
// CHECK:                       modelica.equation_sides
// CHECK-NEXT:              }
// CHECK-NEXT:          }

// Flow variables of the lower triangle: the rows spanning multiple elements,
// then a[1, 2:3] and a[2, 3] (unconnected), then the first row.
// CHECK-NEXT:          modelica.for_equation
// CHECK:                   modelica.range {{.*}} -> tensor<2x!modelica.int>
// CHECK:                       modelica.add
// CHECK:                       modelica.equation_sides
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:          modelica.for_equation
// CHECK:                   modelica.range {{.*}} -> tensor<3x!modelica.int>
// CHECK:                       modelica.add
// CHECK:                       modelica.equation_sides
// CHECK-NEXT:              }
// CHECK-NEXT:          }

// Unconnected flow variables of the upper triangle: a[1, 2:3] and a[2, 3],
// then the first row of the lower triangle, then the same unconnected ones
// for b.
// CHECK-NEXT:          modelica.for_equation %[[j:.*]] : !modelica.int in {
// CHECK-NEXT:              %[[from:.*]] = modelica.constant #modelica<int 1>
// CHECK-NEXT:              %[[to:.*]] = modelica.constant #modelica<int 2>
// CHECK-NEXT:              %[[range:.*]] = modelica.range %[[from]] to %[[to]] {{.*}} -> tensor<2x!modelica.int>
// CHECK-NEXT:              modelica.yield %[[range]]
// CHECK-NEXT:          } do {
// CHECK-NEXT:              modelica.equation {
// CHECK-NEXT:                  %[[zero:.*]] = modelica.constant #modelica<int 0>
// CHECK-NEXT:                  %[[x:.*]] = modelica.ref @a
// CHECK-NEXT:                  %[[xj:.*]] = modelica.subscript %[[x]][%[[zero]], %[[j]]]
// CHECK-NEXT:                  %[[p:.*]] = modelica.member %[[xj]], @p
// CHECK-NEXT:                  %[[i:.*]] = modelica.member %[[p]], @i
// CHECK-NEXT:                  %[[zero:.*]] = modelica.constant #modelica<real 0.000000e+00>
// CHECK-NEXT:                  modelica.equation_sides %[[i]] = %[[zero]]
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:          modelica.equation {
// CHECK-NEXT:              %[[one:.*]] = modelica.constant #modelica<int 1>
// CHECK-NEXT:              %[[two:.*]] = modelica.constant #modelica<int 2>
// CHECK-NEXT:              %[[x:.*]] = modelica.ref @a
// CHECK-NEXT:              %[[x12:.*]] = modelica.subscript %[[x]][%[[one]], %[[two]]]
// CHECK-NEXT:              %[[p:.*]] = modelica.member %[[x12]], @p
// CHECK-NEXT:              %[[i:.*]] = modelica.member %[[p]], @i
// CHECK-NEXT:              %[[zero:.*]] = modelica.constant #modelica<real 0.000000e+00>
// CHECK-NEXT:              modelica.equation_sides %[[i]] = %[[zero]]
// CHECK-NEXT:          }
// CHECK-NEXT:          modelica.equation {
// CHECK:                   modelica.add
// CHECK:                   modelica.equation_sides
// CHECK-NEXT:          }
// CHECK-NEXT:          modelica.for_equation %[[j:.*]] : !modelica.int in {
// CHECK-NEXT:              %[[from:.*]] = modelica.constant #modelica<int 1>
// CHECK-NEXT:              %[[to:.*]] = modelica.constant #modelica<int 2>
// CHECK-NEXT:              %[[range:.*]] = modelica.range %[[from]] to %[[to]] {{.*}} -> tensor<2x!modelica.int>
// CHECK-NEXT:              modelica.yield %[[range]]
// CHECK-NEXT:          } do {
// CHECK-NEXT:              modelica.equation {
// CHECK-NEXT:                  %[[zero:.*]] = modelica.constant #modelica<int 0>
// CHECK-NEXT:                  %[[x:.*]] = modelica.ref @b
// CHECK-NEXT:                  %[[xj:.*]] = modelica.subscript %[[x]][%[[zero]], %[[j]]]
// CHECK-NEXT:                  %[[p:.*]] = modelica.member %[[xj]], @p
// CHECK-NEXT:                  %[[i:.*]] = modelica.member %[[p]], @i
// CHECK-NEXT:                  %[[zero:.*]] = modelica.constant #modelica<real 0.000000e+00>
// CHECK-NEXT:                  modelica.equation_sides %[[i]] = %[[zero]]
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:          modelica.equation {
// CHECK-NEXT:              %[[one:.*]] = modelica.constant #modelica<int 1>
// CHECK-NEXT:              %[[two:.*]] = modelica.constant #modelica<int 2>
// CHECK-NEXT:              %[[x:.*]] = modelica.ref @b
// CHECK-NEXT:              %[[x12:.*]] = modelica.subscript %[[x]][%[[one]], %[[two]]]
// CHECK-NEXT:              %[[p:.*]] = modelica.member %[[x12]], @p
// CHECK-NEXT:              %[[i:.*]] = modelica.member %[[p]], @i
// CHECK-NEXT:              %[[zero:.*]] = modelica.constant #modelica<real 0.000000e+00>
// CHECK-NEXT:              modelica.equation_sides %[[i]] = %[[zero]]
// CHECK-NEXT:          }
// CHECK-NEXT:      }

module {
  modelica.model @Triangular {
    modelica.instance @a : tensor<3x3x!modelica.object<@C>> {
      modelica.instance connector @p : !modelica.object<@Pin> {
        modelica.variable @v : !modelica.real
        modelica.variable flow @i : !modelica.real
      }
    }
    modelica.instance @b : tensor<3x3x!modelica.object<@C>> {
      modelica.instance connector @p : !modelica.object<@Pin> {
        modelica.variable @v : !modelica.real
        modelica.variable flow @i : !modelica.real
      }
    }
    modelica.equations {
      modelica.for_equation %arg0 : !modelica.int in {
        %0 = modelica.constant #modelica<int 0> : !modelica.int
        %1 = modelica.constant #modelica<int 2> : !modelica.int
        %2 = modelica.range %0 to %1 : (!modelica.int, !modelica.int) -> tensor<3x!modelica.int>
        modelica.yield %2 : tensor<3x!modelica.int>
      } do {
        modelica.for_equation %arg1 : !modelica.int in {
          %0 = modelica.constant #modelica<int 0> : !modelica.int
          %1 = modelica.range %0 to %arg0 : (!modelica.int, !modelica.int) -> tensor<?x!modelica.int>
          modelica.yield %1 : tensor<?x!modelica.int>
        } do {
          modelica.connect {
            %0 = modelica.ref @a : tensor<3x3x!modelica.object<@C>>
            %1 = modelica.subscript %0[%arg0, %arg1] : (tensor<3x3x!modelica.object<@C>>, !modelica.int, !modelica.int) -> !modelica.object<@C>
            %2 = modelica.member %1, @p : (!modelica.object<@C>) -> !modelica.object<@Pin>
            %3 = modelica.ref @b : tensor<3x3x!modelica.object<@C>>
            %4 = modelica.subscript %3[%arg0, %arg1] : (tensor<3x3x!modelica.object<@C>>, !modelica.int, !modelica.int) -> !modelica.object<@C>
            %5 = modelica.member %4, @p : (!modelica.object<@C>) -> !modelica.object<@Pin>
            modelica.yield %2, %5 : !modelica.object<@Pin>, !modelica.object<@Pin>
          }
        }
      }
    }
  }
}
