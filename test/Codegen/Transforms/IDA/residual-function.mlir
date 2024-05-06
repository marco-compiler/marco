// RUN: modelica-opt %s --split-input-file --ida | FileCheck %s

// Scalar equation.

// CHECK:       ida.residual_function @ida_main_residualFunction_0(%[[time:.*]]: f64) -> f64 {
// CHECK-DAG:       %[[x:.*]] = bmodelica.qualified_variable_get @Test::@x : !bmodelica.real
// CHECK-DAG:       %[[der_x:.*]] = bmodelica.qualified_variable_get @Test::@der_x : !bmodelica.real
// CHECK-DAG:       %[[result:.*]] = bmodelica.sub %[[der_x]], %[[x]]
// CHECK-DAG:       ida.return %[[result]] : f64
// CHECK-NEXT:  }

module {
    bmodelica.model @Test attributes {derivatives_map = [#bmodelica<var_derivative @x, @der_x>]} {
        bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
        bmodelica.variable @der_x : !bmodelica.variable<!bmodelica.real>

        // x = der(x)
        %t0 = bmodelica.equation_template inductions = [] {
            %0 = bmodelica.variable_get @x : !bmodelica.real
            %1 = bmodelica.variable_get @der_x : !bmodelica.real
            %2 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
            %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
            bmodelica.equation_sides %2, %3 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
        }

        bmodelica.dynamic {
            bmodelica.scc {
                bmodelica.matched_equation_instance %t0 {path = #bmodelica<equation_path [R, 0]>} : !bmodelica.equation
            }
        }
    }
}

// -----

// Vectorized equation with explicit indices.

// CHECK:       ida.residual_function @ida_main_residualFunction_0(%[[time:.*]]: f64, %[[index:.*]]: index) -> f64 {
// CHECK-DAG:       %[[x:.*]] = bmodelica.qualified_variable_get @Test::@x : !bmodelica.array<2x!bmodelica.real>
// CHECK-DAG:       %[[der_x:.*]] = bmodelica.qualified_variable_get @Test::@der_x : !bmodelica.array<2x!bmodelica.real>
// CHECK-DAG:       %[[x_load:.*]] = bmodelica.load %[[x]][%[[index]]]
// CHECK-DAG:       %[[der_x_load:.*]] = bmodelica.load %[[der_x]][%[[index]]]
// CHECK-DAG:       %[[result:.*]] = bmodelica.sub %[[der_x_load]], %[[x_load]]
// CHECK-DAG:       ida.return %[[result]] : f64
// CHECK-NEXT:  }

module {
    bmodelica.model @Test attributes {derivatives_map = [#bmodelica<var_derivative @x, @der_x>]} {
        bmodelica.variable @x : !bmodelica.variable<2x!bmodelica.real>
        bmodelica.variable @der_x : !bmodelica.variable<2x!bmodelica.real>

        // x[i] = der(x[i])
        %t0 = bmodelica.equation_template inductions = [%i0] {
            %0 = bmodelica.variable_get @x : !bmodelica.array<2x!bmodelica.real>
            %1 = bmodelica.load %0[%i0] : !bmodelica.array<2x!bmodelica.real>
            %2 = bmodelica.variable_get @der_x : !bmodelica.array<2x!bmodelica.real>
            %3 = bmodelica.load %2[%i0] : !bmodelica.array<2x!bmodelica.real>
            %4 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
            %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
            bmodelica.equation_sides %4, %5 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
        }

        bmodelica.dynamic {
            bmodelica.scc {
                bmodelica.matched_equation_instance %t0 {indices = #modeling<multidim_range [0,1]>, path = #bmodelica<equation_path [R, 0]>} : !bmodelica.equation
            }
        }
    }
}
