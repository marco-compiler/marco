// RUN: marco-opt %s --convert-modelica-to-bmodelica | FileCheck %s

// Variables keep their variability and causality, while their bindings and
// start values become binding equations and start operations. The 'fixed'
// attribute defaults to false for variables, and 'each' is set for scalar
// start values of arrays. The start values of parameters with a binding are
// dropped, as their value is given by the binding. Enumerations are
// represented by their ordinals.

// CHECK:       bmodelica.model @M
// CHECK-NEXT:      bmodelica.variable @n : !bmodelica.variable<!bmodelica.int, parameter>
// CHECK-NEXT:      bmodelica.variable @k : !bmodelica.variable<!bmodelica.real, constant>
// CHECK-NEXT:      bmodelica.variable @d : !bmodelica.variable<!bmodelica.int, discrete>
// CHECK-NEXT:      bmodelica.variable @u : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:      bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, output>
// CHECK-NEXT:      bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
// CHECK-NEXT:      bmodelica.variable @v : !bmodelica.variable<3x!bmodelica.real>
// CHECK-NEXT:      bmodelica.variable @s : !bmodelica.variable<!bmodelica.int>
// CHECK-NEXT:      bmodelica.binding_equation @n {
// CHECK-NEXT:          %[[n:.*]] = bmodelica.constant #bmodelica<int 3>
// CHECK-NEXT:          bmodelica.yield %[[n]]
// CHECK-NEXT:      }
// CHECK-NEXT:      bmodelica.binding_equation @k {
// CHECK-NEXT:          %[[k:.*]] = bmodelica.constant #bmodelica<real 2.500000e+00>
// CHECK-NEXT:          bmodelica.yield %[[k]]
// CHECK-NEXT:      }
// CHECK-NOT:       bmodelica.start @n
// CHECK-NEXT:      bmodelica.start @x {
// CHECK-NEXT:          %[[x:.*]] = bmodelica.constant #bmodelica<int 1>
// CHECK-NEXT:          bmodelica.yield %[[x]]
// CHECK-NEXT:      } {each = false, fixed = true}
// CHECK-NEXT:      bmodelica.start @v {
// CHECK-NEXT:          %[[v:.*]] = bmodelica.constant #bmodelica<real 2.000000e+00>
// CHECK-NEXT:          bmodelica.yield %[[v]]
// CHECK-NEXT:      } {each = true, fixed = false}
// CHECK-NEXT:      bmodelica.binding_equation @s {
// CHECK-NEXT:          %[[s:.*]] = bmodelica.constant #bmodelica<int 2>
// CHECK-NEXT:          bmodelica.yield %[[s]]
// CHECK-NEXT:      }
// CHECK-NEXT:  }

module {
  modelica.model @M {
    modelica.variable parameter @n : !modelica.int {
      modelica.modify @start {
        %0 = modelica.constant #modelica<int 2> : !modelica.int
        modelica.bind %0 : !modelica.int
      }
      %0 = modelica.constant #modelica<int 3> : !modelica.int
      modelica.bind %0 : !modelica.int
    }
    modelica.variable constant @k : !modelica.real {
      %0 = modelica.constant #modelica<real 2.5> : !modelica.real
      modelica.bind %0 : !modelica.real
    }
    modelica.variable discrete @d : !modelica.int
    modelica.variable input @u : !modelica.real
    modelica.variable output @y : !modelica.real
    modelica.variable @x : !modelica.real {
      modelica.modify @start {
        %0 = modelica.constant #modelica<int 1> : !modelica.int
        modelica.bind %0 : !modelica.int
      }
      modelica.modify @fixed {
        %0 = modelica.constant #modelica<bool true> : !modelica.bool
        modelica.bind %0 : !modelica.bool
      }
      modelica.modify @nominal {
        %0 = modelica.constant #modelica<real 10.0> : !modelica.real
        modelica.bind %0 : !modelica.real
      }
    }
    modelica.variable @v : tensor<3x!modelica.real> {
      modelica.modify each @start {
        %0 = modelica.constant #modelica<real 2.0> : !modelica.real
        modelica.bind %0 : !modelica.real
      }
    }
    modelica.variable @s : !modelica.enum<@E> {
      %0 = modelica.constant #modelica<enum "b" = 2> : !modelica.enum<@E>
      modelica.bind %0 : !modelica.enum<@E>
    }
  }
}
