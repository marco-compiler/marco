// RUN: modelica-opt %s --split-input-file --ida | FileCheck %s

// Scalar variables.

// CHECK:       sundials.variable_getter @ida_main_getter_0() -> f64 {
// CHECK:           %[[get:.*]] = modelica.simulation_variable_get @x : !modelica.real
// CHECK:           %[[cast:.*]] = modelica.cast %[[get]] : !modelica.real -> f64
// CHECK:           sundials.return %[[cast]]
// CHECK-NEXT:  }

// CHECK:       sundials.variable_getter @ida_main_getter_1() -> f64 {
// CHECK:           %[[get:.*]] = modelica.simulation_variable_get @der_x : !modelica.real
// CHECK:           %[[cast:.*]] = modelica.cast %[[get]] : !modelica.real -> f64
// CHECK:           sundials.return %[[cast]]
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

// CHECK:       sundials.variable_getter @ida_main_getter_0(%[[index:.*]]: index) -> f64 {
// CHECK:           %[[get:.*]] = modelica.simulation_variable_get @x : !modelica.array<2x!modelica.real>
// CHECK:           %[[load:.*]] = modelica.load %[[get]][%[[index]]]
// CHECK:           %[[cast:.*]] = modelica.cast %[[load]] : !modelica.real -> f64
// CHECK:           sundials.return %[[cast]]
// CHECK-NEXT:  }

// CHECK:       sundials.variable_getter @ida_main_getter_1(%[[index:.*]]: index) -> f64 {
// CHECK:           %[[get:.*]] = modelica.simulation_variable_get @der_x : !modelica.array<2x!modelica.real>
// CHECK:           %[[load:.*]] = modelica.load %[[get]][%[[index]]]
// CHECK:           %[[cast:.*]] = modelica.cast %[[load]] : !modelica.real -> f64
// CHECK:           sundials.return %[[cast]]
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

// CHECK:       sundials.variable_getter @ida_main_getter_0(%[[index_0:.*]]: index, %[[index_1:.*]]: index) -> f64 {
// CHECK:           %[[get:.*]] = modelica.simulation_variable_get @x : !modelica.array<2x3x!modelica.real>
// CHECK:           %[[load:.*]] = modelica.load %[[get]][%[[index_0]], %[[index_1]]]
// CHECK:           %[[cast:.*]] = modelica.cast %[[load]] : !modelica.real -> f64
// CHECK:           sundials.return %[[cast]]
// CHECK-NEXT:  }

// CHECK:       sundials.variable_getter @ida_main_getter_1(%[[index_0:.*]]: index, %[[index_1:.*]]: index) -> f64 {
// CHECK:           %[[get:.*]] = modelica.simulation_variable_get @der_x : !modelica.array<2x3x!modelica.real>
// CHECK:           %[[load:.*]] = modelica.load %[[get]][%[[index_0]], %[[index_1]]]
// CHECK:           %[[cast:.*]] = modelica.cast %[[load]] : !modelica.real -> f64
// CHECK:           sundials.return %[[cast]]
// CHECK-NEXT:  }

module {
    modelica.simulation_variable @x : !modelica.variable<2x3x!modelica.real>
    modelica.simulation_variable @der_x : !modelica.variable<2x3x!modelica.real>

    modelica.model @Test attributes {derivatives_map = [#modelica<var_derivative @x, @der_x>]} {
        modelica.variable @x : !modelica.variable<2x3x!modelica.real>
        modelica.variable @der_x : !modelica.variable<2x3x!modelica.real>
    }
}
