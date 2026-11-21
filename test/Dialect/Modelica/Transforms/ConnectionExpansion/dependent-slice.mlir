// RUN: marco-opt %s --modelica-connection-expansion | FileCheck %s

// Slices whose bounds depend on the iterators of the enclosing
// for-equations. Only the elements of the lower triangle are connected.
//
// model DependentSlice
//   Pin a[3, 3], b[3, 4];
// equation
//   for i in 1:3 loop
//     connect(a[i, 1:i], b[i, 2:i + 1]);
//   end for;
// end DependentSlice;

// CHECK:       modelica.model @DependentSlice
// CHECK-NOT:       modelica.connect
// CHECK:           modelica.equations {
// CHECK-NEXT:      }
// CHECK-NEXT:      modelica.equations {

// b[1, 2] = a[1, 1]
// CHECK-NEXT:          modelica.equation {
// CHECK-NEXT:              %[[zero:.*]] = modelica.constant #modelica<int 0>
// CHECK-NEXT:              %[[one:.*]] = modelica.constant #modelica<int 1>
// CHECK-NEXT:              %[[b:.*]] = modelica.ref @b
// CHECK-NEXT:              %[[b01:.*]] = modelica.subscript %[[b]][%[[zero]], %[[one]]]
// CHECK-NEXT:              %[[lhs:.*]] = modelica.member %[[b01]], @v
// CHECK-NEXT:              %[[zero:.*]] = modelica.constant #modelica<int 0>
// CHECK-NEXT:              %[[zero2:.*]] = modelica.constant #modelica<int 0>
// CHECK-NEXT:              %[[a:.*]] = modelica.ref @a
// CHECK-NEXT:              %[[a00:.*]] = modelica.subscript %[[a]][%[[zero]], %[[zero2]]]
// CHECK-NEXT:              %[[rhs:.*]] = modelica.member %[[a00]], @v
// CHECK-NEXT:              modelica.equation_sides %[[lhs]] = %[[rhs]]
// CHECK-NEXT:          }

// b[2, j + 1] = a[2, j] for j in 1:2
// CHECK-NEXT:          modelica.for_equation %[[j:.*]] : !modelica.int in {
// CHECK-NEXT:              %[[from:.*]] = modelica.constant #modelica<int 1>
// CHECK-NEXT:              %[[to:.*]] = modelica.constant #modelica<int 2>
// CHECK-NEXT:              %[[range:.*]] = modelica.range %[[from]] to %[[to]] {{.*}} -> tensor<2x!modelica.int>
// CHECK-NEXT:              modelica.yield %[[range]]
// CHECK-NEXT:          } do {
// CHECK-NEXT:              modelica.equation {
// CHECK-NEXT:                  %[[one:.*]] = modelica.constant #modelica<int 1>
// CHECK-NEXT:                  %[[b:.*]] = modelica.ref @b
// CHECK-NEXT:                  %[[b1j:.*]] = modelica.subscript %[[b]][%[[one]], %[[j]]]
// CHECK-NEXT:                  %[[lhs:.*]] = modelica.member %[[b1j]], @v
// CHECK-NEXT:                  %[[one:.*]] = modelica.constant #modelica<int 1>
// CHECK-NEXT:                  %[[minusOne:.*]] = modelica.constant #modelica<int -1>
// CHECK-NEXT:                  %[[jm1:.*]] = modelica.add %[[j]], %[[minusOne]]
// CHECK-NEXT:                  %[[a:.*]] = modelica.ref @a
// CHECK-NEXT:                  %[[a1j:.*]] = modelica.subscript %[[a]][%[[one]], %[[jm1]]]
// CHECK-NEXT:                  %[[rhs:.*]] = modelica.member %[[a1j]], @v
// CHECK-NEXT:                  modelica.equation_sides %[[lhs]] = %[[rhs]]
// CHECK-NEXT:              }
// CHECK-NEXT:          }

// b[3, j + 1] = a[3, j] for j in 1:3
// CHECK-NEXT:          modelica.for_equation %[[j:.*]] : !modelica.int in {
// CHECK-NEXT:              %[[from:.*]] = modelica.constant #modelica<int 1>
// CHECK-NEXT:              %[[to:.*]] = modelica.constant #modelica<int 3>
// CHECK-NEXT:              %[[range:.*]] = modelica.range %[[from]] to %[[to]] {{.*}} -> tensor<3x!modelica.int>
// CHECK-NEXT:              modelica.yield %[[range]]
// CHECK-NEXT:          } do {
// CHECK-NEXT:              modelica.equation {
// CHECK-NEXT:                  %[[two:.*]] = modelica.constant #modelica<int 2>
// CHECK-NEXT:                  %[[b:.*]] = modelica.ref @b
// CHECK-NEXT:                  %[[b2j:.*]] = modelica.subscript %[[b]][%[[two]], %[[j]]]
// CHECK-NEXT:                  %[[lhs:.*]] = modelica.member %[[b2j]], @v
// CHECK-NEXT:                  %[[two:.*]] = modelica.constant #modelica<int 2>
// CHECK-NEXT:                  %[[minusOne:.*]] = modelica.constant #modelica<int -1>
// CHECK-NEXT:                  %[[jm1:.*]] = modelica.add %[[j]], %[[minusOne]]
// CHECK-NEXT:                  %[[a:.*]] = modelica.ref @a
// CHECK-NEXT:                  %[[a2j:.*]] = modelica.subscript %[[a]][%[[two]], %[[jm1]]]
// CHECK-NEXT:                  %[[rhs:.*]] = modelica.member %[[a2j]], @v
// CHECK-NEXT:                  modelica.equation_sides %[[lhs]] = %[[rhs]]
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:      }
// CHECK-NEXT:  }

module {
  modelica.model @DependentSlice {
    modelica.instance connector @a : tensor<3x3x!modelica.object<@Pin>> {
      modelica.variable @v : !modelica.real
    }
    modelica.instance connector @b : tensor<3x4x!modelica.object<@Pin>> {
      modelica.variable @v : !modelica.real
    }
    modelica.equations {
      modelica.for_equation %arg0 : !modelica.int in {
        %0 = modelica.constant #modelica<int 0> : !modelica.int
        %1 = modelica.constant #modelica<int 2> : !modelica.int
        %2 = modelica.range %0 to %1 : (!modelica.int, !modelica.int) -> tensor<3x!modelica.int>
        modelica.yield %2 : tensor<3x!modelica.int>
      } do {
        modelica.connect {
          %0 = modelica.ref @a : tensor<3x3x!modelica.object<@Pin>>
          %1 = modelica.constant #modelica<int 0> : !modelica.int
          %2 = modelica.range %1 to %arg0 : (!modelica.int, !modelica.int) -> tensor<?x!modelica.int>
          %3 = modelica.subscript %0[%arg0, %2] : (tensor<3x3x!modelica.object<@Pin>>, !modelica.int, tensor<?x!modelica.int>) -> tensor<?x!modelica.object<@Pin>>
          %4 = modelica.ref @b : tensor<3x4x!modelica.object<@Pin>>
          %5 = modelica.constant #modelica<int 1> : !modelica.int
          %6 = modelica.add %arg0, %5 : (!modelica.int, !modelica.int) -> !modelica.int
          %7 = modelica.range %5 to %6 : (!modelica.int, !modelica.int) -> tensor<?x!modelica.int>
          %8 = modelica.subscript %4[%arg0, %7] : (tensor<3x4x!modelica.object<@Pin>>, !modelica.int, tensor<?x!modelica.int>) -> tensor<?x!modelica.object<@Pin>>
          modelica.yield %3, %8 : tensor<?x!modelica.object<@Pin>>, tensor<?x!modelica.object<@Pin>>
        }
      }
    }
  }
}
