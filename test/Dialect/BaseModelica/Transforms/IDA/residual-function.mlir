// RUN: modelica-opt %s --split-input-file --ida | FileCheck %s

// CHECK:       ida.residual_function @ida_main_residualFunction_0(%[[time:.*]]: f64) -> f64 {
// CHECK-DAG:       %[[x:.*]] = bmodelica.qualified_variable_get @scalarEquation::@x : !bmodelica.real
// CHECK-DAG:       %[[der_x:.*]] = bmodelica.qualified_variable_get @scalarEquation::@der_x : !bmodelica.real
// CHECK-DAG:       %[[result:.*]] = bmodelica.sub %[[der_x]], %[[x]]
// CHECK-DAG:       ida.return %[[result]] : f64
// CHECK-NEXT:  }

module {
    bmodelica.model @scalarEquation der = [<@x, @der_x>] {
        bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
        bmodelica.variable @der_x : !bmodelica.variable<!bmodelica.real>

        // COM: x = der(x)
        %t0 = bmodelica.equation_template inductions = [] {
            %0 = bmodelica.variable_get @x : !bmodelica.real
            %1 = bmodelica.variable_get @der_x : !bmodelica.real
            %2 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
            %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
            bmodelica.equation_sides %2, %3 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
        }

        bmodelica.dynamic {
            bmodelica.scc {
                bmodelica.matched_equation_instance %t0, match = @der_x
            }
        }
    }
}

// -----

// CHECK:       ida.residual_function @ida_main_residualFunction_0(%[[time:.*]]: f64, %[[index:.*]]: index) -> f64 {
// CHECK-DAG:       %[[x:.*]] = bmodelica.qualified_variable_get @arrayEquationExplicitAccess::@x : tensor<2x!bmodelica.real>
// CHECK-DAG:       %[[der_x:.*]] = bmodelica.qualified_variable_get @arrayEquationExplicitAccess::@der_x : tensor<2x!bmodelica.real>
// CHECK-DAG:       %[[x_extract:.*]] = bmodelica.tensor_extract %[[x]][%[[index]]]
// CHECK-DAG:       %[[der_x_extract:.*]] = bmodelica.tensor_extract %[[der_x]][%[[index]]]
// CHECK-DAG:       %[[result:.*]] = bmodelica.sub %[[der_x_extract]], %[[x_extract]]
// CHECK-DAG:       ida.return %[[result]] : f64
// CHECK-NEXT:  }

module {
    bmodelica.model @arrayEquationExplicitAccess der = [<@x, @der_x>] {
        bmodelica.variable @x : !bmodelica.variable<2x!bmodelica.real>
        bmodelica.variable @der_x : !bmodelica.variable<2x!bmodelica.real>

        // COM: x[i] = der(x[i])
        %t0 = bmodelica.equation_template inductions = [%i0] {
            %0 = bmodelica.variable_get @x : tensor<2x!bmodelica.real>
            %1 = bmodelica.tensor_extract %0[%i0] : tensor<2x!bmodelica.real>
            %2 = bmodelica.variable_get @der_x : tensor<2x!bmodelica.real>
            %3 = bmodelica.tensor_extract %2[%i0] : tensor<2x!bmodelica.real>
            %4 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
            %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
            bmodelica.equation_sides %4, %5 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
        }

        bmodelica.dynamic {
            bmodelica.scc {
                bmodelica.matched_equation_instance %t0, match = <@der_x, {[0,1]}> {indices = #modeling<multidim_range [0,1]>}
            }
        }
    }
}
