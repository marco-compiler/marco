// RUN: marco-opt %s --convert-modelica-to-bmodelica | FileCheck %s

// The dynamic dimensions of the variables of functions are either free or
// computed by the dimensions region. For-statements iterate on an induction
// variable stored in memory, assignments to elements set the variable at the
// given indices, and assignments to slices insert the value into the
// current one. External calls assign their result to the destination. Only
// the functions called by the models are converted.

// CHECK-LABEL: bmodelica.function @g
// CHECK-NEXT:      bmodelica.variable @u : !bmodelica.variable<?x!bmodelica.real, input> [free]
// CHECK-NEXT:      bmodelica.variable @y : !bmodelica.variable<?x!bmodelica.real, output> [fixed] {
// CHECK:               %[[size:.*]] = bmodelica.size
// CHECK-NEXT:          %[[cast:.*]] = bmodelica.cast %[[size]] : !bmodelica.int -> index
// CHECK-NEXT:          bmodelica.yield %[[cast]]
// CHECK-NEXT:      }
// CHECK-NEXT:      bmodelica.variable @k : !bmodelica.variable<!bmodelica.int>
// CHECK-NEXT:      bmodelica.algorithm {
// CHECK:               %[[induction:.*]] = bmodelica.array.alloca
// CHECK-NEXT:          bmodelica.array.store %[[induction]][]
// CHECK-NEXT:          bmodelica.for condition {
// CHECK:               } body {
// CHECK-NEXT:              %[[i:.*]] = bmodelica.array.load %[[induction]][]
// CHECK:                   %[[ui:.*]] = bmodelica.tensor.extract
// CHECK-NEXT:              %[[one:.*]] = bmodelica.constant 1 : index
// CHECK-NEXT:              %[[index:.*]] = bmodelica.sub %[[i]], %[[one]]
// CHECK-NEXT:              bmodelica.variable.set @y[%[[index]] : index], %[[ui]]
// CHECK-NEXT:              bmodelica.yield
// CHECK-NEXT:          } step {
// CHECK:               bmodelica.variable.set @k
// CHECK-NEXT:          bmodelica.while {
// CHECK:                   bmodelica.condition
// CHECK-NEXT:          } do {
// CHECK:                   bmodelica.if
// CHECK-NEXT:                  bmodelica.break
// CHECK-NEXT:              }
// CHECK:                   bmodelica.variable.set @k
// CHECK-NEXT:              bmodelica.yield
// CHECK-NEXT:          }
// CHECK:               %[[values:.*]] = bmodelica.constant #bmodelica.dense_real<[3.000000e+00, 4.000000e+00]>
// CHECK:               %[[range:.*]] = bmodelica.range.bounded
// CHECK-NEXT:          %[[y:.*]] = bmodelica.variable.get @y
// CHECK-NEXT:          %[[updated:.*]] = bmodelica.tensor.insert_slice %[[values]], %[[y]][%[[range]]]
// CHECK-NEXT:          bmodelica.variable.set @y, %[[updated]]
// CHECK-NEXT:          bmodelica.assert attributes {level = 1 : i32, message = "negative"} {
// CHECK:                   %[[condition:.*]] = bmodelica.gte
// CHECK-NEXT:              bmodelica.yield %[[condition]]
// CHECK-NEXT:          }
// CHECK-NEXT:      }
// CHECK-NEXT:  }

// CHECK-NOT:   @unused

// CHECK-LABEL: bmodelica.function @h
// CHECK:           bmodelica.algorithm {
// CHECK-NEXT:          %[[x:.*]] = bmodelica.variable.get @x
// CHECK-NEXT:          %[[result:.*]] = bmodelica.external_call @h_impl(%[[x]])
// CHECK-NEXT:          bmodelica.variable.set @y, %[[result]]
// CHECK-NEXT:      }

module {
  modelica.function @g {
    modelica.variable input @u : tensor<?x!modelica.real>
    modelica.variable output @y : tensor<?x!modelica.real> dims {
      %u = modelica.ref @u : tensor<?x!modelica.real>
      %0 = modelica.constant #modelica<int 0> : !modelica.int
      %1 = modelica.size %u, %0 : (tensor<?x!modelica.real>, !modelica.int) -> !modelica.int
      modelica.yield %1 : !modelica.int
    }
    modelica.variable @k : !modelica.int
    modelica.algorithm {
      modelica.for %i : !modelica.int in {
        %0 = modelica.constant #modelica<int 1> : !modelica.int
        %u = modelica.ref @u : tensor<?x!modelica.real>
        %dim = modelica.constant #modelica<int 0> : !modelica.int
        %1 = modelica.size %u, %dim : (tensor<?x!modelica.real>, !modelica.int) -> !modelica.int
        %2 = modelica.range %0 to %1 : (!modelica.int, !modelica.int) -> tensor<?x!modelica.int>
        modelica.yield %2 : tensor<?x!modelica.int>
      } do {
        modelica.assignment {
          %y = modelica.ref @y : tensor<?x!modelica.real>
          %one = modelica.constant #modelica<int 1> : !modelica.int
          %index = modelica.sub %i, %one : (!modelica.int, !modelica.int) -> !modelica.int
          %0 = modelica.subscript %y[%index] : (tensor<?x!modelica.real>, !modelica.int) -> !modelica.real
          %u = modelica.ref @u : tensor<?x!modelica.real>
          %1 = modelica.subscript %u[%index] : (tensor<?x!modelica.real>, !modelica.int) -> !modelica.real
          modelica.assignment_sides %0 = %1 : !modelica.real = !modelica.real
        }
      }
      modelica.assignment {
        %k = modelica.ref @k : !modelica.int
        %0 = modelica.constant #modelica<int 0> : !modelica.int
        modelica.assignment_sides %k = %0 : !modelica.int = !modelica.int
      }
      modelica.while {
        %k = modelica.ref @k : !modelica.int
        %0 = modelica.constant #modelica<int 2> : !modelica.int
        %1 = modelica.lt %k, %0 : (!modelica.int, !modelica.int) -> !modelica.bool
        modelica.yield %1 : !modelica.bool
      } do {
        modelica.if {
          %k = modelica.ref @k : !modelica.int
          %0 = modelica.constant #modelica<int 1> : !modelica.int
          %1 = modelica.eq %k, %0 : (!modelica.int, !modelica.int) -> !modelica.bool
          modelica.yield %1 : !modelica.bool
        } then {
          modelica.break
        }
        modelica.assignment {
          %k = modelica.ref @k : !modelica.int
          %k2 = modelica.ref @k : !modelica.int
          %0 = modelica.constant #modelica<int 1> : !modelica.int
          %1 = modelica.add %k2, %0 : (!modelica.int, !modelica.int) -> !modelica.int
          modelica.assignment_sides %k = %1 : !modelica.int = !modelica.int
        }
      }
      modelica.assignment {
        %y = modelica.ref @y : tensor<?x!modelica.real>
        %0 = modelica.constant #modelica<int 1> : !modelica.int
        %1 = modelica.constant #modelica<int 2> : !modelica.int
        %2 = modelica.range %0 to %1 : (!modelica.int, !modelica.int) -> tensor<2x!modelica.int>
        %3 = modelica.subscript %y[%2] : (tensor<?x!modelica.real>, tensor<2x!modelica.int>) -> tensor<2x!modelica.real>
        %4 = modelica.constant #modelica<array [3.0, 4.0]> : tensor<2x!modelica.real>
        modelica.assignment_sides %3 = %4 : tensor<2x!modelica.real> = tensor<2x!modelica.real>
      }
      modelica.assert {
        %k = modelica.ref @k : !modelica.int
        %0 = modelica.constant #modelica<int 0> : !modelica.int
        %1 = modelica.ge %k, %0 : (!modelica.int, !modelica.int) -> !modelica.bool
        %2 = modelica.constant #modelica<string "negative"> : !modelica.string
        modelica.yield %1, %2 : !modelica.bool, !modelica.string
      }
    }
  }
  modelica.function @h {
    modelica.variable input @x : !modelica.real
    modelica.variable output @y : !modelica.real
    modelica.external "C" "h_impl" output {
      %y = modelica.ref @y : !modelica.real
      %x = modelica.ref @x : !modelica.real
      modelica.yield %y, %x : !modelica.real, !modelica.real
    }
  }
  modelica.function @unused {
    modelica.variable input @message : !modelica.string
  }
  modelica.model @M {
    modelica.variable @x : tensor<2x!modelica.real>
    modelica.variable @y : !modelica.real
    modelica.equations {
      modelica.equation {
        %x = modelica.ref @x : tensor<2x!modelica.real>
        %0 = modelica.call @g(%x) : (tensor<2x!modelica.real>) -> tensor<2x!modelica.real>
        modelica.equation_sides %x = %0 : tensor<2x!modelica.real> = tensor<2x!modelica.real>
      }
      modelica.equation {
        %y = modelica.ref @y : !modelica.real
        %0 = modelica.time : !modelica.real
        %1 = modelica.call @h(%0) : (!modelica.real) -> !modelica.real
        modelica.equation_sides %y = %1 : !modelica.real = !modelica.real
      }
    }
  }
}
