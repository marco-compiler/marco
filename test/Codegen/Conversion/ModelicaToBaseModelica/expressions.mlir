// RUN: marco-opt %s --convert-modelica-to-bmodelica | FileCheck %s

// Slices become views, fill-expressions become broadcasts
// with the dimensions known at compile time, if-expressions become selections
// of the values of both branches, reductions keep their iteration
// ranges, and array constructors with iterators are unrolled. The named
// arguments of calls are reordered as the inputs of the callee, and the
// missing arguments preceding them get their default values. The promotion
// of arrays rebuilds them with the new shape, and semiLinear becomes a
// selection.

// CHECK-LABEL: bmodelica.function @f
// CHECK:           bmodelica.default @b
// CHECK:           bmodelica.default @c

// CHECK-LABEL: bmodelica.model @M

// CHECK:       bmodelica.equation {
// CHECK:           %[[x:.*]] = bmodelica.variable.get @x
// CHECK-DAG:       %[[begin:.*]] = bmodelica.constant 1 : index
// CHECK-DAG:       %[[end:.*]] = bmodelica.constant 2 : index
// CHECK:           %[[range:.*]] = bmodelica.range.bounded %[[begin]], %[[end]], %{{.*}} : index, index, index
// CHECK-NEXT:      %[[colon:.*]] = bmodelica.range.unbounded
// CHECK-NEXT:      %[[view:.*]] = bmodelica.tensor.view %[[x]][%[[range]], %[[colon]]]
// CHECK-NEXT:      %[[y:.*]] = bmodelica.variable.get @y
// CHECK:           %[[fill:.*]] = bmodelica.tensor.broadcast %[[y]] : !bmodelica.real -> tensor<2x4x!bmodelica.real>
// CHECK:           %[[lhs:.*]] = bmodelica.equation_side %[[view]]
// CHECK-NEXT:      %[[rhs:.*]] = bmodelica.equation_side %[[fill]]

// CHECK:       bmodelica.equation {
// CHECK:           %[[condition:.*]] = bmodelica.gt
// CHECK:           %[[range:.*]] = bmodelica.range.bounded %{{.*}}, %{{.*}}, %{{.*}} : !bmodelica.int, !bmodelica.int, !bmodelica.int
// CHECK-NEXT:      %[[sum:.*]] = bmodelica.reduction "add", iterables = [%[[range]]], inductions = [%[[i:.*]]: !bmodelica.int] {
// CHECK:               %[[i_index:.*]] = bmodelica.cast %[[i]] : !bmodelica.int -> index
// CHECK-NEXT:          %[[one:.*]] = bmodelica.constant 1 : index
// CHECK-NEXT:          %[[index:.*]] = bmodelica.sub %[[i_index]], %[[one]]
// CHECK:               %[[extract:.*]] = bmodelica.tensor.extract %{{.*}}[%[[index]], %{{.*}}]
// CHECK-NEXT:          bmodelica.yield %[[extract]]
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[c:.*]] = bmodelica.constant #bmodelica<int 3>
// CHECK-NEXT:      %[[b:.*]] = bmodelica.constant #bmodelica<int 1>
// CHECK-NEXT:      %[[call:.*]] = bmodelica.call @f(%{{.*}}, %[[b]], %[[c]])
// CHECK-NEXT:      %[[select:.*]] = bmodelica.select (%[[condition]] : !bmodelica.bool), (%[[sum]] : !bmodelica.real), (%[[call]] : !bmodelica.real)

// CHECK:       bmodelica.equation {
// CHECK:           %[[first:.*]] = bmodelica.mul
// CHECK:           %[[second:.*]] = bmodelica.mul
// CHECK-NEXT:      %[[array:.*]] = bmodelica.tensor.from_elements %[[first]], %[[second]] : !bmodelica.real, !bmodelica.real -> tensor<2x!bmodelica.real>

// CHECK:       bmodelica.equation {
// CHECK:           %[[v:.*]] = bmodelica.variable.get @v
// CHECK-DAG:       %[[first:.*]] = bmodelica.tensor.extract %[[v]][%{{.*}}]
// CHECK-DAG:       %[[second:.*]] = bmodelica.tensor.extract %[[v]][%{{.*}}]
// CHECK:           %[[matrix:.*]] = bmodelica.tensor.from_elements %[[first]], %[[second]] : !bmodelica.real, !bmodelica.real -> tensor<2x1x!bmodelica.real>

// CHECK:       bmodelica.equation {
// CHECK:           %[[y:.*]] = bmodelica.variable.get @y
// CHECK:           %[[zero:.*]] = bmodelica.constant #bmodelica<int 0>
// CHECK-NEXT:      %[[condition:.*]] = bmodelica.gte %[[y]], %[[zero]]
// CHECK-NEXT:      %[[positive:.*]] = bmodelica.mul %{{.*}}, %[[y]]
// CHECK-NEXT:      %[[negative:.*]] = bmodelica.mul %{{.*}}, %[[y]]
// CHECK-NEXT:      %{{.*}} = bmodelica.select (%[[condition]] : !bmodelica.bool), (%[[positive]] : !bmodelica.real), (%[[negative]] : !bmodelica.real)

module {
  modelica.function @f {
    modelica.variable input @a : !modelica.real
    modelica.variable input @b : !modelica.real {
      %0 = modelica.constant #modelica<int 1> : !modelica.int
      modelica.bind %0 : !modelica.int
    }
    modelica.variable input @c : !modelica.real {
      %0 = modelica.constant #modelica<int 2> : !modelica.int
      modelica.bind %0 : !modelica.int
    }
    modelica.variable output @y : !modelica.real
  }
  modelica.model @M {
    modelica.variable @x : tensor<3x4x!modelica.real>
    modelica.variable @y : !modelica.real
    modelica.variable @v : tensor<2x!modelica.real>
    modelica.variable @w : tensor<2x1x!modelica.real>
    modelica.variable @z : !modelica.real
    modelica.equations {
      // x[2:3, :] = fill(y, 2, size(x, 2))
      modelica.equation {
        %x = modelica.ref @x : tensor<3x4x!modelica.real>
        %0 = modelica.constant #modelica<int 1> : !modelica.int
        %1 = modelica.constant #modelica<int 2> : !modelica.int
        %2 = modelica.range %0 to %1 : (!modelica.int, !modelica.int) -> tensor<2x!modelica.int>
        %3 = modelica.subscript %x[%2, :] : (tensor<3x4x!modelica.real>, tensor<2x!modelica.int>) -> tensor<2x4x!modelica.real>
        %y = modelica.ref @y : !modelica.real
        %x2 = modelica.ref @x : tensor<3x4x!modelica.real>
        %4 = modelica.size %x2, %0 : (tensor<3x4x!modelica.real>, !modelica.int) -> !modelica.int
        %5 = modelica.fill %y, %1, %4 : (!modelica.real, !modelica.int, !modelica.int) -> tensor<2x?x!modelica.real>
        modelica.equation_sides %3 = %5 : tensor<2x4x!modelica.real> = tensor<2x?x!modelica.real>
      }
      // y = if y > 0 then sum(x[i, 1] for i in 1:3) else f(y, c = 3)
      modelica.equation {
        %y = modelica.ref @y : !modelica.real
        %0 = modelica.constant #modelica<int 0> : !modelica.int
        %1 = modelica.gt %y, %0 : (!modelica.real, !modelica.int) -> !modelica.bool
        %2 = modelica.if_expr %1 : !modelica.real {
          %3 = modelica.constant #modelica<int 1> : !modelica.int
          %4 = modelica.constant #modelica<int 3> : !modelica.int
          %5 = modelica.range %3 to %4 : (!modelica.int, !modelica.int) -> tensor<3x!modelica.int>
          %6 = modelica.reduction sum (%i in %5 : tensor<3x!modelica.int>) -> !modelica.real {
            %x = modelica.ref @x : tensor<3x4x!modelica.real>
            %one = modelica.constant #modelica<int 1> : !modelica.int
            %index = modelica.sub %i, %one : (!modelica.int, !modelica.int) -> !modelica.int
            %7 = modelica.constant #modelica<int 0> : !modelica.int
            %8 = modelica.subscript %x[%index, %7] : (tensor<3x4x!modelica.real>, !modelica.int, !modelica.int) -> !modelica.real
            modelica.yield %8 : !modelica.real
          }
          modelica.yield %6 : !modelica.real
        } else {
          %3 = modelica.constant #modelica<int 3> : !modelica.int
          %4 = modelica.call @f(%y, @c = %3) : (!modelica.real, !modelica.int) -> !modelica.real
          modelica.yield %4 : !modelica.real
        }
        modelica.equation_sides %y = %2 : !modelica.real = !modelica.real
      }
      // v = {y * i for i in 1:2}
      modelica.equation {
        %v = modelica.ref @v : tensor<2x!modelica.real>
        %0 = modelica.constant #modelica<int 1> : !modelica.int
        %1 = modelica.constant #modelica<int 2> : !modelica.int
        %2 = modelica.range %0 to %1 : (!modelica.int, !modelica.int) -> tensor<2x!modelica.int>
        %3 = modelica.array_for (%i in %2 : tensor<2x!modelica.int>) -> tensor<2x!modelica.real> {
          %y = modelica.ref @y : !modelica.real
          %4 = modelica.mul %y, %i : (!modelica.real, !modelica.int) -> !modelica.real
          modelica.yield %4 : !modelica.real
        }
        modelica.equation_sides %v = %3 : tensor<2x!modelica.real> = tensor<2x!modelica.real>
      }
      // w = promote(v, 2)
      modelica.equation {
        %w = modelica.ref @w : tensor<2x1x!modelica.real>
        %v = modelica.ref @v : tensor<2x!modelica.real>
        %0 = modelica.promote %v, 2 : (tensor<2x!modelica.real>) -> tensor<2x1x!modelica.real>
        modelica.equation_sides %w = %0 : tensor<2x1x!modelica.real> = tensor<2x1x!modelica.real>
      }
      // z = semiLinear(y, 2, 3)
      modelica.equation {
        %z = modelica.ref @z : !modelica.real
        %y = modelica.ref @y : !modelica.real
        %0 = modelica.constant #modelica<int 2> : !modelica.int
        %1 = modelica.constant #modelica<int 3> : !modelica.int
        %2 = modelica.semi_linear %y, %0, %1 : (!modelica.real, !modelica.int, !modelica.int) -> !modelica.real
        modelica.equation_sides %z = %2 : !modelica.real = !modelica.real
      }
    }
  }
}
