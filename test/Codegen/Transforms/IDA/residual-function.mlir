// RUN: modelica-opt %s --split-input-file --ida | FileCheck %s

// Scalar equation.

// CHECK:       ida.residual_function @ida_main_residualFunction_0(%[[time:.*]]: f64) -> f64 {
// CHECK-DAG:       %[[x:.*]] = modelica.simulation_variable_get @x : !modelica.real
// CHECK-DAG:       %[[der_x:.*]] = modelica.simulation_variable_get @der_x : !modelica.real
// CHECK-DAG:       %[[result:.*]] = modelica.sub %[[der_x]], %[[x]]
// CHECK-DAG:       ida.return %[[result]] : f64
// CHECK-NEXT:  }

module {
    modelica.simulation_variable @x : !modelica.variable<!modelica.real>
    modelica.simulation_variable @der_x : !modelica.variable<!modelica.real>

    modelica.model @Test attributes {derivatives_map = [#modelica<var_derivative @x, @der_x>]} {
        modelica.variable @x : !modelica.variable<!modelica.real>
        modelica.variable @der_x : !modelica.variable<!modelica.real>

        // x = der(x)
        %t0 = modelica.equation_template inductions = [] {
            %0 = modelica.variable_get @x : !modelica.real
            %1 = modelica.variable_get @der_x : !modelica.real
            %2 = modelica.equation_side %0 : tuple<!modelica.real>
            %3 = modelica.equation_side %1 : tuple<!modelica.real>
            modelica.equation_sides %2, %3 : tuple<!modelica.real>, tuple<!modelica.real>
        }

        modelica.main_model {
            modelica.scc {
                modelica.matched_equation_instance %t0 {path = #modelica<equation_path [R, 0]>} : !modelica.equation
            }
        }
    }
}

// -----

// Vectorized equation with explicit indices.

// CHECK:       ida.residual_function @ida_main_residualFunction_0(%[[time:.*]]: f64, %[[index:.*]]: index) -> f64 {
// CHECK-DAG:       %[[x:.*]] = modelica.simulation_variable_get @x : !modelica.array<2x!modelica.real>
// CHECK-DAG:       %[[der_x:.*]] = modelica.simulation_variable_get @der_x : !modelica.array<2x!modelica.real>
// CHECK-DAG:       %[[x_load:.*]] = modelica.load %[[x]][%[[index]]]
// CHECK-DAG:       %[[der_x_load:.*]] = modelica.load %[[der_x]][%[[index]]]
// CHECK-DAG:       %[[result:.*]] = modelica.sub %[[der_x_load]], %[[x_load]]
// CHECK-DAG:       ida.return %[[result]] : f64
// CHECK-NEXT:  }

module {
    modelica.simulation_variable @x : !modelica.variable<2x!modelica.real>
    modelica.simulation_variable @der_x : !modelica.variable<2x!modelica.real>

    modelica.model @Test attributes {derivatives_map = [#modelica<var_derivative @x, @der_x>]} {
        modelica.variable @x : !modelica.variable<2x!modelica.real>
        modelica.variable @der_x : !modelica.variable<2x!modelica.real>

        // x[i] = der(x[i])
        %t0 = modelica.equation_template inductions = [%i0] {
            %0 = modelica.variable_get @x : !modelica.array<2x!modelica.real>
            %1 = modelica.load %0[%i0] : !modelica.array<2x!modelica.real>
            %2 = modelica.variable_get @der_x : !modelica.array<2x!modelica.real>
            %3 = modelica.load %2[%i0] : !modelica.array<2x!modelica.real>
            %4 = modelica.equation_side %1 : tuple<!modelica.real>
            %5 = modelica.equation_side %3 : tuple<!modelica.real>
            modelica.equation_sides %4, %5 : tuple<!modelica.real>, tuple<!modelica.real>
        }

        modelica.main_model {
            modelica.scc {
                modelica.matched_equation_instance %t0 {indices = #modeling<multidim_range [0,1]>, path = #modelica<equation_path [R, 0]>} : !modelica.equation
            }
        }
    }
}
