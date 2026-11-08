// RUN: marco-opt %s | marco-opt | FileCheck %s

// CHECK-LABEL: modelica.model @M
// CHECK-SAME:      attributes {annotation = {experiment = {StopTime = 1.000000e+00 : f64}}}
// CHECK-NEXT:      modelica.variable parameter @k : !modelica.real {
// CHECK-NEXT:          %[[k:.*]] = modelica.constant
// CHECK-NEXT:          modelica.bind %[[k]] : !modelica.real
// CHECK-NEXT:      }
// CHECK-NEXT:      modelica.instance protected model @r : tensor<2x!modelica.object<@R>> {
// CHECK-NEXT:          modelica.instance connector @p : !modelica.object<@Pin> {
// CHECK-NEXT:              modelica.variable @v : !modelica.real
// CHECK-NEXT:              modelica.variable flow @i : !modelica.real
// CHECK-NEXT:          }
// CHECK-NEXT:          modelica.variable discrete output @x : tensor<3x!modelica.real> {
// CHECK-NEXT:          ^bb0(%[[index:.*]]: !modelica.int):
// CHECK-NEXT:              modelica.modify each @start {
// CHECK:                   %[[value:.*]] = modelica.subscript %{{.*}}[%[[index]]]
// CHECK:                   %[[fill:.*]] = modelica.fill %[[value]]
// CHECK-NEXT:              modelica.bind %[[fill]]
// CHECK-NEXT:          } attributes {description = "x"}
// CHECK-NEXT:      }
// CHECK-NEXT:      modelica.equations {
// CHECK-NEXT:          modelica.connect {
// CHECK:               } {lhsOutside}

modelica.model @M attributes {annotation = {experiment = {StopTime = 1.0}}} {
    modelica.variable parameter @k : !modelica.real {
        %0 = modelica.constant #modelica<real 2.0> : !modelica.real
        modelica.bind %0 : !modelica.real
    }
    modelica.instance protected model @r : tensor<2x!modelica.object<@R>> {
        modelica.instance connector @p : !modelica.object<@Pin> {
            modelica.variable @v : !modelica.real
            modelica.variable flow @i : !modelica.real
        }
        modelica.variable discrete output @x : tensor<3x!modelica.real> {
        ^bb0(%i: !modelica.int):
            modelica.modify each @start {
                %0 = modelica.constant #modelica<real 1.0> : !modelica.real
                modelica.bind %0 : !modelica.real
            }
            %0 = modelica.constant #modelica<array [1.0, 2.0]> : tensor<2x!modelica.real>
            %1 = modelica.subscript %0[%i] : (tensor<2x!modelica.real>, !modelica.int) -> !modelica.real
            %2 = modelica.constant #modelica<int 3> : !modelica.int
            %3 = modelica.fill %1, %2 : (!modelica.real, !modelica.int) -> tensor<3x!modelica.real>
            modelica.bind %3 : tensor<3x!modelica.real>
        } attributes {description = "x"}
    }
    modelica.equations {
        modelica.connect {
            %0 = modelica.ref @r : tensor<2x!modelica.object<@R>>
            %1 = modelica.member %0, @p : (tensor<2x!modelica.object<@R>>) -> tensor<2x!modelica.object<@Pin>>
            %2 = modelica.ref @r : tensor<2x!modelica.object<@R>>
            %3 = modelica.member %2, @p : (tensor<2x!modelica.object<@R>>) -> tensor<2x!modelica.object<@Pin>>
            modelica.yield %1, %3 : tensor<2x!modelica.object<@Pin>>, tensor<2x!modelica.object<@Pin>>
        } {lhsOutside}
    }
}

// CHECK-LABEL: modelica.function impure @Pkg.f
// CHECK-NEXT:      modelica.variable input @x : tensor<?x!modelica.real>
// CHECK-NEXT:      modelica.variable output @y : tensor<?x!modelica.real> dims {
// CHECK:               modelica.yield %{{.*}} : !modelica.int
// CHECK-NEXT:      }
// CHECK-NEXT:      modelica.algorithm {
// CHECK:           modelica.external "C"

modelica.function impure @"Pkg.f" {
    modelica.variable input @x : tensor<?x!modelica.real>
    modelica.variable output @y : tensor<?x!modelica.real> dims {
        %0 = modelica.ref @x : tensor<?x!modelica.real>
        %1 = modelica.constant #modelica<int 0> : !modelica.int
        %2 = modelica.size %0, %1 : (tensor<?x!modelica.real>, !modelica.int) -> !modelica.int
        modelica.yield %2 : !modelica.int
    }
    modelica.algorithm {
        modelica.assignment {
            %0 = modelica.ref @y : tensor<?x!modelica.real>
            %1 = modelica.ref @x : tensor<?x!modelica.real>
            modelica.assignment_sides %0 = %1 : tensor<?x!modelica.real> = tensor<?x!modelica.real>
        }
    }
    modelica.external "C"
}
