// RUN: marco-opt %s --modelica-flatten | FileCheck %s

// References through instances become references to the flattened
// variables, with the subscripts of the instances followed by the ones of
// the variables. The inputs and outputs of the model are the top-level
// ones, and the ones of the top-level connectors. Scalar attributes keep
// applying to each element.

// CHECK:       modelica.model @M {
// CHECK-NEXT:      modelica.variable input @u.x : !modelica.real
// CHECK-NEXT:      modelica.variable @c.u : tensor<2x!modelica.real> {
// CHECK-NEXT:          modelica.modify each @start {
// CHECK-NEXT:              %[[start:.*]] = modelica.constant #modelica<real 1.000000e+00>
// CHECK-NEXT:              modelica.bind %[[start]]
// CHECK-NEXT:          }
// CHECK-NEXT:      }
// CHECK-NEXT:      modelica.variable @c.v : tensor<2x3x!modelica.real>
// CHECK-NEXT:      modelica.variable output @y : !modelica.real
// CHECK-NEXT:      modelica.equations {
// CHECK-NEXT:          modelica.equation {
// CHECK-NEXT:              %[[two:.*]] = modelica.constant #modelica<int 2>
// CHECK-NEXT:              %[[v:.*]] = modelica.ref @c.v : tensor<2x3x!modelica.real>
// CHECK-NEXT:              %[[vi:.*]] = modelica.subscript %[[v]][%[[two]]] : (tensor<2x3x!modelica.real>, !modelica.int) -> tensor<3x!modelica.real>
// CHECK-NEXT:              %[[three:.*]] = modelica.constant #modelica<int 3>
// CHECK-NEXT:              %[[vij:.*]] = modelica.subscript %[[vi]][%[[three]]] : (tensor<3x!modelica.real>, !modelica.int) -> !modelica.real
// CHECK-NEXT:              %[[u:.*]] = modelica.ref @c.u : tensor<2x!modelica.real>
// CHECK-NEXT:              %[[sum:.*]] = modelica.array_reduce sum %[[u]]
// CHECK-NEXT:              modelica.equation_sides %[[vij]] = %[[sum]]
// CHECK-NEXT:          }
// CHECK-NEXT:          modelica.equation {
// CHECK-NEXT:              %[[v:.*]] = modelica.ref @c.v : tensor<2x3x!modelica.real>
// CHECK-NEXT:              %[[one:.*]] = modelica.constant #modelica<int 1>
// CHECK-NEXT:              %[[col:.*]] = modelica.subscript %[[v]][:, %[[one]]] : (tensor<2x3x!modelica.real>, !modelica.int) -> tensor<2x!modelica.real>
// CHECK-NEXT:              %[[u:.*]] = modelica.ref @c.u : tensor<2x!modelica.real>
// CHECK-NEXT:              modelica.equation_sides %[[col]] = %[[u]]
// CHECK-NEXT:          }
// CHECK-NEXT:          modelica.equation {
// CHECK-NEXT:              %[[y:.*]] = modelica.ref @y : !modelica.real
// CHECK-NEXT:              %[[x:.*]] = modelica.ref @u.x : !modelica.real
// CHECK-NEXT:              modelica.equation_sides %[[y]] = %[[x]]

modelica.model @M {
    modelica.instance connector @u : !modelica.object<@In> {
        modelica.variable input @x : !modelica.real
    }
    modelica.instance model @c : tensor<2x!modelica.object<@C>> {
        modelica.variable @u : !modelica.real {
        ^bb0(%i: !modelica.int):
            modelica.modify @start {
                %0 = modelica.constant #modelica<real 1.0> : !modelica.real
                modelica.bind %0 : !modelica.real
            }
        }
        modelica.variable @v : tensor<3x!modelica.real>
    }
    modelica.variable output @y : !modelica.real
    modelica.equations {
        modelica.equation {
            %0 = modelica.ref @c : tensor<2x!modelica.object<@C>>
            %1 = modelica.constant #modelica<int 2> : !modelica.int
            %2 = modelica.subscript %0[%1] : (tensor<2x!modelica.object<@C>>, !modelica.int) -> !modelica.object<@C>
            %3 = modelica.member %2, @v : (!modelica.object<@C>) -> tensor<3x!modelica.real>
            %4 = modelica.constant #modelica<int 3> : !modelica.int
            %5 = modelica.subscript %3[%4] : (tensor<3x!modelica.real>, !modelica.int) -> !modelica.real
            %6 = modelica.ref @c : tensor<2x!modelica.object<@C>>
            %7 = modelica.member %6, @u : (tensor<2x!modelica.object<@C>>) -> tensor<2x!modelica.real>
            %8 = modelica.array_reduce sum %7 : (tensor<2x!modelica.real>) -> !modelica.real
            modelica.equation_sides %5 = %8 : !modelica.real = !modelica.real
        }
        modelica.equation {
            %0 = modelica.ref @c : tensor<2x!modelica.object<@C>>
            %1 = modelica.member %0, @v : (tensor<2x!modelica.object<@C>>) -> tensor<2x3x!modelica.real>
            %2 = modelica.constant #modelica<int 1> : !modelica.int
            %3 = modelica.subscript %1[:, %2] : (tensor<2x3x!modelica.real>, !modelica.int) -> tensor<2x!modelica.real>
            %4 = modelica.ref @c : tensor<2x!modelica.object<@C>>
            %5 = modelica.member %4, @u : (tensor<2x!modelica.object<@C>>) -> tensor<2x!modelica.real>
            modelica.equation_sides %3 = %5 : tensor<2x!modelica.real> = tensor<2x!modelica.real>
        }
        modelica.equation {
            %0 = modelica.ref @y : !modelica.real
            %1 = modelica.ref @u : !modelica.object<@In>
            %2 = modelica.member %1, @x : (!modelica.object<@In>) -> !modelica.real
            modelica.equation_sides %0 = %2 : !modelica.real = !modelica.real
        }
    }
}
