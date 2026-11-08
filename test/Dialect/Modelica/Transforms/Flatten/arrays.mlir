// RUN: marco-opt %s --modelica-flatten | FileCheck %s

// The variables of arrays of instances get the dimensions of the arrays,
// and their modifications are extended to all the elements: the values
// depending on the element become array constructors, and the other ones
// are replicated.

// CHECK:       modelica.model @M {
// CHECK-NEXT:      modelica.variable parameter @rs.r : tensor<3x!modelica.real> {
// CHECK:               %[[range:.*]] = modelica.range %{{.*}} to %{{.*}} : (!modelica.int, !modelica.int) -> tensor<3x!modelica.int>
// CHECK-NEXT:          %[[values:.*]] = modelica.array_for (%[[i:.*]] in %[[range]] : tensor<3x!modelica.int>) -> tensor<3x!modelica.int> {
// CHECK:                   %[[value:.*]] = modelica.subscript %{{.*}}[%[[i]]]
// CHECK-NEXT:              modelica.yield %[[value]]
// CHECK-NEXT:          }
// CHECK-NEXT:          modelica.bind %[[values]]
// CHECK-NEXT:      }
// CHECK-NEXT:      modelica.variable parameter @rs.n : tensor<3x!modelica.int> {
// CHECK-NEXT:          %[[n:.*]] = modelica.constant #modelica<int 4>
// CHECK-NEXT:          %[[size:.*]] = modelica.constant #modelica<int 3>
// CHECK-NEXT:          %[[fill:.*]] = modelica.fill %[[n]], %[[size]]
// CHECK-NEXT:          modelica.bind %[[fill]]
// CHECK-NEXT:      }
// CHECK-NEXT:      modelica.variable @rs.x : tensor<3x4x!modelica.real>
// CHECK-NOT:       modelica.instance
// CHECK:           modelica.equations {
// CHECK:               modelica.equation {
// CHECK-NEXT:              %[[y:.*]] = modelica.ref @y : tensor<3x!modelica.real>
// CHECK-NEXT:              %[[yi:.*]] = modelica.subscript %[[y]][%[[j:.*]]]
// CHECK-NEXT:              %[[x:.*]] = modelica.ref @rs.x : tensor<3x4x!modelica.real>
// CHECK-NEXT:              %[[xi:.*]] = modelica.subscript %[[x]][%[[j]]] : (tensor<3x4x!modelica.real>, !modelica.int) -> tensor<4x!modelica.real>
// CHECK-NEXT:              %[[one:.*]] = modelica.constant #modelica<int 1>
// CHECK-NEXT:              %[[xij:.*]] = modelica.subscript %[[xi]][%[[one]]]
// CHECK-NEXT:              modelica.equation_sides %[[yi]] = %[[xij]]

module {
  modelica.model @M {
    modelica.variable parameter @N : !modelica.int {
      %0 = modelica.constant #modelica<int 3> : !modelica.int
      modelica.bind %0 : !modelica.int
    }
    modelica.instance model @rs : tensor<3x!modelica.object<@R>> {
      modelica.variable parameter @r : !modelica.real {
      ^bb0(%arg0: !modelica.int):
        %0 = modelica.constant #modelica<int 1> : !modelica.int
        %1 = modelica.constant #modelica<int 2> : !modelica.int
        %2 = modelica.constant #modelica<int 3> : !modelica.int
        %3 = modelica.array %0, %1, %2 : (!modelica.int, !modelica.int, !modelica.int) -> tensor<3x!modelica.int>
        %4 = modelica.subscript %3[%arg0] : (tensor<3x!modelica.int>, !modelica.int) -> !modelica.int
        modelica.bind %4 : !modelica.int
      }
      modelica.variable parameter @n : !modelica.int {
      ^bb0(%arg0: !modelica.int):
        %0 = modelica.constant #modelica<int 4> : !modelica.int
        modelica.bind %0 : !modelica.int
      }
      modelica.variable @x : tensor<4x!modelica.real>
    }
    modelica.variable @y : tensor<3x!modelica.real>
    modelica.equations {
      modelica.for_equation %arg0 : !modelica.int in {
        %0 = modelica.constant #modelica<int 1> : !modelica.int
        %1 = modelica.constant #modelica<int 3> : !modelica.int
        %2 = modelica.range %0 to %1 : (!modelica.int, !modelica.int) -> tensor<3x!modelica.int>
        modelica.yield %2 : tensor<3x!modelica.int>
      } do {
        modelica.equation {
          %0 = modelica.ref @y : tensor<3x!modelica.real>
          %1 = modelica.subscript %0[%arg0] : (tensor<3x!modelica.real>, !modelica.int) -> !modelica.real
          %2 = modelica.ref @rs : tensor<3x!modelica.object<@R>>
          %3 = modelica.subscript %2[%arg0] : (tensor<3x!modelica.object<@R>>, !modelica.int) -> !modelica.object<@R>
          %4 = modelica.member %3, @x : (!modelica.object<@R>) -> tensor<4x!modelica.real>
          %5 = modelica.constant #modelica<int 1> : !modelica.int
          %6 = modelica.subscript %4[%5] : (tensor<4x!modelica.real>, !modelica.int) -> !modelica.real
          modelica.equation_sides %1 = %6 : !modelica.real = !modelica.real
        }
      }
      modelica.for_equation %arg0 : !modelica.int in {
        %0 = modelica.constant #modelica<int 1> : !modelica.int
        %1 = modelica.constant #modelica<int 3> : !modelica.int
        %2 = modelica.range %0 to %1 : (!modelica.int, !modelica.int) -> tensor<3x!modelica.int>
        modelica.yield %2 : tensor<3x!modelica.int>
      } do {
        modelica.equation {
          %0 = modelica.ref @rs : tensor<3x!modelica.object<@R>>
          %1 = modelica.subscript %0[%arg0] : (tensor<3x!modelica.object<@R>>, !modelica.int) -> !modelica.object<@R>
          %2 = modelica.member %1, @x : (!modelica.object<@R>) -> tensor<4x!modelica.real>
          %3 = modelica.ref @rs : tensor<3x!modelica.object<@R>>
          %4 = modelica.subscript %3[%arg0] : (tensor<3x!modelica.object<@R>>, !modelica.int) -> !modelica.object<@R>
          %5 = modelica.member %4, @r : (!modelica.object<@R>) -> !modelica.real
          %6 = modelica.ref @rs : tensor<3x!modelica.object<@R>>
          %7 = modelica.subscript %6[%arg0] : (tensor<3x!modelica.object<@R>>, !modelica.int) -> !modelica.object<@R>
          %8 = modelica.member %7, @n : (!modelica.object<@R>) -> !modelica.int
          %9 = modelica.fill %5, %8 : (!modelica.real, !modelica.int) -> tensor<?x!modelica.real>
          modelica.equation_sides %2 = %9 : tensor<4x!modelica.real> = tensor<?x!modelica.real>
        }
      }
    }
  }
}

