// RUN: marco-opt %s --convert-modelica-to-bmodelica | FileCheck %s

// For-equations over arithmetic progressions are kept, while the other ones
// are unrolled, as well as the ones containing if-equations whose condition
// depends on the induction variable. If-equations with conditions known at
// compile time are replaced by the selected branch, while the equations of
// the branches of the other ones are merged into equations selecting their
// sides. The arithmetic on the indices is performed on values of type
// 'index', so that the accesses remain affine.

// CHECK:       bmodelica.dynamic {
// CHECK-NEXT:      bmodelica.for_equation %[[i:.*]] = 1 to 3 step 2 {
// CHECK-NEXT:          bmodelica.equation {
// CHECK-NEXT:              %[[x:.*]] = bmodelica.variable.get @x
// CHECK-NEXT:              %[[one:.*]] = bmodelica.constant 1 : index
// CHECK-NEXT:              %[[index:.*]] = bmodelica.sub %[[i]], %[[one]] : index, index -> index
// CHECK-NEXT:              %[[xi:.*]] = bmodelica.tensor.extract %[[x]][%[[index]]]
// CHECK-NEXT:              %[[lhs:.*]] = bmodelica.equation_side %[[xi]]
// CHECK-NEXT:              %[[rhs:.*]] = bmodelica.equation_side %[[i]]
// CHECK-NEXT:              bmodelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:          }
// CHECK-NEXT:      }

// CHECK-NEXT:      bmodelica.equation {
// CHECK-NEXT:          %[[x:.*]] = bmodelica.variable.get @x
// CHECK-NEXT:          %[[index:.*]] = bmodelica.constant 1 : index
// CHECK-NEXT:          %[[xi:.*]] = bmodelica.tensor.extract %[[x]][%[[index]]]
// CHECK-NEXT:          %[[value:.*]] = bmodelica.constant #bmodelica<int 0>
// CHECK:           }
// CHECK-NEXT:      bmodelica.equation {
// CHECK-NEXT:          %[[x:.*]] = bmodelica.variable.get @x
// CHECK-NEXT:          %[[index:.*]] = bmodelica.constant 3 : index
// CHECK-NEXT:          %[[xi:.*]] = bmodelica.tensor.extract %[[x]][%[[index]]]
// CHECK-NEXT:          %[[value:.*]] = bmodelica.constant #bmodelica<int 1>
// CHECK:           }

// CHECK-NEXT:      bmodelica.equation {
// CHECK-NEXT:          %[[y:.*]] = bmodelica.variable.get @y
// CHECK-NEXT:          %[[value:.*]] = bmodelica.constant #bmodelica<int 1>
// CHECK:           }

// CHECK-NEXT:      bmodelica.equation {
// CHECK-NEXT:          %[[time:.*]] = bmodelica.time
// CHECK-NEXT:          %[[one:.*]] = bmodelica.constant #bmodelica<int 1>
// CHECK-NEXT:          %[[condition:.*]] = bmodelica.gt %[[time]], %[[one]]
// CHECK-NEXT:          %[[z1:.*]] = bmodelica.variable.get @z
// CHECK-NEXT:          %[[y1:.*]] = bmodelica.variable.get @y
// CHECK-NEXT:          %[[z2:.*]] = bmodelica.variable.get @z
// CHECK-NEXT:          %[[two:.*]] = bmodelica.constant #bmodelica<int 2>
// CHECK-NEXT:          %[[y2:.*]] = bmodelica.variable.get @y
// CHECK-NEXT:          %[[mul:.*]] = bmodelica.mul %[[two]], %[[y2]]
// CHECK-NEXT:          %[[lhs:.*]] = bmodelica.select (%[[condition]] : !bmodelica.bool), (%[[z1]] : !bmodelica.real), (%[[z2]] : !bmodelica.real)
// CHECK-NEXT:          %[[rhs:.*]] = bmodelica.select (%[[condition]] : !bmodelica.bool), (%[[y1]] : !bmodelica.real), (%[[mul]] : !bmodelica.real)
// CHECK-NEXT:          %[[lhs_side:.*]] = bmodelica.equation_side %[[lhs]]
// CHECK-NEXT:          %[[rhs_side:.*]] = bmodelica.equation_side %[[rhs]]
// CHECK-NEXT:          bmodelica.equation_sides %[[lhs_side]], %[[rhs_side]]
// CHECK-NEXT:      }
// CHECK-NEXT:  }

// CHECK-NEXT:  bmodelica.initial {
// CHECK-NEXT:      bmodelica.equation {
// CHECK-NEXT:          %[[y:.*]] = bmodelica.variable.get @y

module {
  modelica.model @M {
    modelica.variable parameter @useA : !modelica.bool {
      %0 = modelica.constant #modelica<bool true> : !modelica.bool
      modelica.bind %0 : !modelica.bool
    }
    modelica.variable @x : tensor<4x!modelica.real>
    modelica.variable @y : !modelica.real
    modelica.variable @z : !modelica.real
    modelica.equations {
      // for i in 1:2:3 loop x[i] = i; end for;
      modelica.for_equation %i : !modelica.int in {
        %0 = modelica.constant #modelica<int 1> : !modelica.int
        %1 = modelica.constant #modelica<int 2> : !modelica.int
        %2 = modelica.constant #modelica<int 3> : !modelica.int
        %3 = modelica.range %0 to %2 step %1 : (!modelica.int, !modelica.int, !modelica.int) -> tensor<2x!modelica.int>
        modelica.yield %3 : tensor<2x!modelica.int>
      } do {
        modelica.equation {
          %0 = modelica.ref @x : tensor<4x!modelica.real>
          %one = modelica.constant #modelica<int 1> : !modelica.int
          %index = modelica.sub %i, %one : (!modelica.int, !modelica.int) -> !modelica.int
          %1 = modelica.subscript %0[%index] : (tensor<4x!modelica.real>, !modelica.int) -> !modelica.real
          modelica.equation_sides %1 = %i : !modelica.real = !modelica.int
        }
      }
      // for i in {2, 4} loop if i == 2 then x[i] = 0; else x[i] = 1; end if; end for;
      modelica.for_equation %i : !modelica.int in {
        %0 = modelica.constant #modelica<array [2, 4]> : tensor<2x!modelica.int>
        modelica.yield %0 : tensor<2x!modelica.int>
      } do {
        modelica.if_equation {
          %0 = modelica.constant #modelica<int 2> : !modelica.int
          %1 = modelica.eq %i, %0 : (!modelica.int, !modelica.int) -> !modelica.bool
          modelica.yield %1 : !modelica.bool
        } then {
          modelica.equation {
            %0 = modelica.ref @x : tensor<4x!modelica.real>
            %one = modelica.constant #modelica<int 1> : !modelica.int
            %index = modelica.sub %i, %one : (!modelica.int, !modelica.int) -> !modelica.int
            %1 = modelica.subscript %0[%index] : (tensor<4x!modelica.real>, !modelica.int) -> !modelica.real
            %2 = modelica.constant #modelica<int 0> : !modelica.int
            modelica.equation_sides %1 = %2 : !modelica.real = !modelica.int
          }
        } else {
          modelica.equation {
            %0 = modelica.ref @x : tensor<4x!modelica.real>
            %one = modelica.constant #modelica<int 1> : !modelica.int
            %index = modelica.sub %i, %one : (!modelica.int, !modelica.int) -> !modelica.int
            %1 = modelica.subscript %0[%index] : (tensor<4x!modelica.real>, !modelica.int) -> !modelica.real
            %2 = modelica.constant #modelica<int 1> : !modelica.int
            modelica.equation_sides %1 = %2 : !modelica.real = !modelica.int
          }
        }
      }
      // if useA then y = 1; else y = 2; end if;
      modelica.if_equation {
        %0 = modelica.ref @useA : !modelica.bool
        modelica.yield %0 : !modelica.bool
      } then {
        modelica.equation {
          %0 = modelica.ref @y : !modelica.real
          %1 = modelica.constant #modelica<int 1> : !modelica.int
          modelica.equation_sides %0 = %1 : !modelica.real = !modelica.int
        }
      } else {
        modelica.equation {
          %0 = modelica.ref @y : !modelica.real
          %1 = modelica.constant #modelica<int 2> : !modelica.int
          modelica.equation_sides %0 = %1 : !modelica.real = !modelica.int
        }
      }
      // if time > 1 then z = y; else z = 2 * y; end if;
      modelica.if_equation {
        %0 = modelica.time : !modelica.real
        %1 = modelica.constant #modelica<int 1> : !modelica.int
        %2 = modelica.gt %0, %1 : (!modelica.real, !modelica.int) -> !modelica.bool
        modelica.yield %2 : !modelica.bool
      } then {
        modelica.equation {
          %0 = modelica.ref @z : !modelica.real
          %1 = modelica.ref @y : !modelica.real
          modelica.equation_sides %0 = %1 : !modelica.real = !modelica.real
        }
      } else {
        modelica.equation {
          %0 = modelica.ref @z : !modelica.real
          %1 = modelica.constant #modelica<int 2> : !modelica.int
          %2 = modelica.ref @y : !modelica.real
          %3 = modelica.mul %1, %2 : (!modelica.int, !modelica.real) -> !modelica.real
          modelica.equation_sides %0 = %3 : !modelica.real = !modelica.real
        }
      }
    }
    modelica.equations initial {
      modelica.equation {
        %0 = modelica.ref @y : !modelica.real
        %1 = modelica.constant #modelica<int 0> : !modelica.int
        modelica.equation_sides %0 = %1 : !modelica.real = !modelica.int
      }
    }
  }
}
