// RUN: marco-opt %s --split-input-file --ida="variables-per-function=2 equations-per-function=2" | FileCheck %s

// COM: The variables and equations are registered by multiple functions, each
// COM: of them handling at most two variables or two equations. The IDA
// COM: variables are shared among such functions by means of a buffer.

// CHECK:       runtime.dynamic_model_begin {
// CHECK-NEXT:      ida.create @ida_main
// CHECK-NEXT:      %[[buffer:.*]] = memref.alloc() : memref<3x!ida.variable>
// CHECK-NEXT:      func.call @ida_main_add_variables_0(%[[buffer]])
// CHECK-NEXT:      func.call @ida_main_add_variables_1(%[[buffer]])
// CHECK-NEXT:      func.call @ida_main_add_equations_0(%[[buffer]])
// CHECK-NEXT:      func.call @ida_main_add_equations_1(%[[buffer]])
// CHECK-NEXT:  }

// CHECK-LABEL: func.func private @ida_main_add_variables_0
// CHECK-SAME:      (%[[arg:.*]]: memref<3x!ida.variable>)
// CHECK:           %[[x:.*]] = ida.add_state_variable
// CHECK:           memref.store %[[x]], %[[arg]][%{{.*}}]
// CHECK:           %[[y:.*]] = ida.add_state_variable
// CHECK:           memref.store %[[y]], %[[arg]][%{{.*}}]
// CHECK-NOT:       ida.add_state_variable
// CHECK:           return

// CHECK-LABEL: func.func private @ida_main_add_variables_1
// CHECK-SAME:      (%[[arg:.*]]: memref<3x!ida.variable>)
// CHECK:           %[[z:.*]] = ida.add_state_variable
// CHECK:           memref.store %[[z]], %[[arg]][%{{.*}}]
// CHECK-NOT:       ida.add_state_variable
// CHECK:           return

// CHECK-LABEL: func.func private @ida_main_add_equations_0
// CHECK-SAME:      (%[[arg:.*]]: memref<3x!ida.variable>)
// CHECK:           %[[eq0:.*]] = ida.add_equation
// CHECK:           %[[var0:.*]] = memref.load %[[arg]][%{{.*}}]
// CHECK:           ida.add_variable_access @ida_main, %[[eq0]], %[[var0]]
// CHECK:           ida.set_residual @ida_main, %[[eq0]]
// CHECK:           ida.add_jacobian @ida_main, %[[eq0]]
// CHECK:           %[[eq1:.*]] = ida.add_equation
// CHECK:           ida.set_residual @ida_main, %[[eq1]]
// CHECK-NOT:       ida.add_equation
// CHECK:           return

// CHECK-LABEL: func.func private @ida_main_add_equations_1
// CHECK-SAME:      (%[[arg:.*]]: memref<3x!ida.variable>)
// CHECK:           %[[eq2:.*]] = ida.add_equation
// CHECK:           %[[var2:.*]] = memref.load %[[arg]][%{{.*}}]
// CHECK:           ida.add_variable_access @ida_main, %[[eq2]], %[[var2]]
// CHECK:           ida.set_residual @ida_main, %[[eq2]]
// CHECK-NOT:       ida.add_equation
// CHECK:           return

bmodelica.model @chunks der = [<@x, @der_x>, <@y, @der_y>, <@z, @der_z>] {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @z : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @der_x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @der_y : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @der_z : !bmodelica.variable<!bmodelica.real>

    // COM: der(x) = y
    %t0 = bmodelica.equation_template inductions = [] {
        %0 = bmodelica.variable.get @der_x : !bmodelica.real
        %1 = bmodelica.variable.get @y : !bmodelica.real
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // COM: der(y) = z
    %t1 = bmodelica.equation_template inductions = [] {
        %0 = bmodelica.variable.get @der_y : !bmodelica.real
        %1 = bmodelica.variable.get @z : !bmodelica.real
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // COM: der(z) = x
    %t2 = bmodelica.equation_template inductions = [] {
        %0 = bmodelica.variable.get @der_z : !bmodelica.real
        %1 = bmodelica.variable.get @x : !bmodelica.real
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    bmodelica.dynamic {
        bmodelica.scc {
            bmodelica.equation_instance %t0, match = @der_x
        }
        bmodelica.scc {
            bmodelica.equation_instance %t1, match = @der_y
        }
        bmodelica.scc {
            bmodelica.equation_instance %t2, match = @der_z
        }
    }
}
