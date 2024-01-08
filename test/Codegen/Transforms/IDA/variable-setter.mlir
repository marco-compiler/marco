// RUN: modelica-opt %s --split-input-file --ida | FileCheck %s

// Scalar variables.

// CHECK:       sundials.variable_setter @ida_main_setter_0(%[[value:.*]]: f64) {
// CHECK:           %[[cast:.*]] = modelica.cast %[[value]] : f64 -> !modelica.real
// CHECK:           modelica.simulation_variable_set @x, %[[cast]]
// CHECK:           sundials.return
// CHECK-NEXT:  }

// CHECK:       sundials.variable_setter @ida_main_setter_1(%[[value:.*]]: f64) {
// CHECK:           %[[cast:.*]] = modelica.cast %[[value]] : f64 -> !modelica.real
// CHECK:           modelica.simulation_variable_set @der_x, %[[cast]]
// CHECK:           sundials.return
// CHECK-NEXT:  }

module {
    modelica.simulation_variable @x : !modelica.variable<!modelica.real>
    modelica.simulation_variable @der_x : !modelica.variable<!modelica.real>

    modelica.model @Test attributes {derivatives_map = [#modelica<var_derivative @x, @der_x>]} {
        modelica.variable @x : !modelica.variable<!modelica.real>
        modelica.variable @der_x : !modelica.variable<!modelica.real>
    }
}

// -----

// 1-D array variables.

// CHECK:       sundials.variable_setter @ida_main_setter_0(%[[value:.*]]: f64, %[[index:.*]]: index) {
// CHECK-DAG:       %[[get:.*]] = modelica.simulation_variable_get @x : !modelica.array<2x!modelica.real>
// CHECK-DAG:       %[[cast:.*]] = modelica.cast %[[value]] : f64 -> !modelica.real
// CHECK:           modelica.store %[[get]][%[[index]]], %[[cast]]
// CHECK:           sundials.return
// CHECK-NEXT:  }

// CHECK:       sundials.variable_setter @ida_main_setter_1(%[[value:.*]]: f64, %[[index:.*]]: index) {
// CHECK-DAG:       %[[get:.*]] = modelica.simulation_variable_get @der_x : !modelica.array<2x!modelica.real>
// CHECK-DAG:       %[[cast:.*]] = modelica.cast %[[value]] : f64 -> !modelica.real
// CHECK:           modelica.store %[[get]][%[[index]]], %[[cast]]
// CHECK:           sundials.return
// CHECK-NEXT:  }

module {
    modelica.simulation_variable @x : !modelica.variable<2x!modelica.real>
    modelica.simulation_variable @der_x : !modelica.variable<2x!modelica.real>

    modelica.model @Test attributes {derivatives_map = [#modelica<var_derivative @x, @der_x>]} {
        modelica.variable @x : !modelica.variable<2x!modelica.real>
        modelica.variable @der_x : !modelica.variable<2x!modelica.real>
    }
}

// -----

// 2-D array variables.

// CHECK:       sundials.variable_setter @ida_main_setter_0(%[[value:.*]]: f64, %[[index_0:.*]]: index, %[[index_1:.*]]: index) {
// CHECK-DAG:       %[[array:.*]] = modelica.simulation_variable_get @x : !modelica.array<2x3x!modelica.real>
// CHECK-DAG:       %[[cast:.*]] = modelica.cast %[[value]] : f64 -> !modelica.real
// CHECK:           modelica.store %[[get]][%[[index_0]], %[[index_1]]], %[[cast]]
// CHECK:           sundials.return
// CHECK-NEXT:  }

// CHECK:       sundials.variable_setter @ida_main_setter_1(%[[value:.*]]: f64, %[[index_0:.*]]: index, %[[index_1:.*]]: index) {
// CHECK-DAG:       %[[array:.*]] = modelica.simulation_variable_get @der_x : !modelica.array<2x3x!modelica.real>
// CHECK-DAG:       %[[cast:.*]] = modelica.cast %[[value]] : f64 -> !modelica.real
// CHECK:           modelica.store %[[get]][%[[index_0]], %[[index_1]]], %[[cast]]
// CHECK:           sundials.return
// CHECK-NEXT:  }

module {
    modelica.simulation_variable @x : !modelica.variable<2x3x!modelica.real>
    modelica.simulation_variable @der_x : !modelica.variable<2x3x!modelica.real>

    modelica.model @Test attributes {derivatives_map = [#modelica<var_derivative @x, @der_x>]} {
        modelica.variable @x : !modelica.variable<2x3x!modelica.real>
        modelica.variable @der_x : !modelica.variable<2x3x!modelica.real>
    }
}
