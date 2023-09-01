// RUN: modelica-opt %s --split-input-file --ida | FileCheck %s

// Scalar variables.

// CHECK:       ida.variable_getter @ida_main_getter_0() -> f64 {
// CHECK:           %[[global_get:.*]] = modelica.global_variable_get @var_0 : !modelica.array<!modelica.real>
// CHECK:           %[[load:.*]] = modelica.load %[[global_get]][]
// CHECK:           %[[cast:.*]] = modelica.cast %[[load]] : !modelica.real -> f64
// CHECK:           ida.return %[[cast]]
// CHECK-NEXT:  }

// CHECK:       ida.variable_getter @ida_main_getter_1() -> f64 {
// CHECK:           %[[global_get:.*]] = modelica.global_variable_get @var_1 : !modelica.array<!modelica.real>
// CHECK:           %[[load:.*]] = modelica.load %[[global_get]][]
// CHECK:           %[[cast:.*]] = modelica.cast %[[load]] : !modelica.real -> f64
// CHECK:           ida.return %[[cast]]
// CHECK-NEXT:  }

modelica.model @Test attributes {derivatives_map = [#modelica<var_derivative @x, @der_x>]} {
    modelica.variable @x : !modelica.variable<!modelica.real>
    modelica.variable @der_x : !modelica.variable<!modelica.real>
}

// -----

// 1-D array variables.

// CHECK:       ida.variable_getter @ida_main_getter_0(%[[index:.*]]: index) -> f64 {
// CHECK:           %[[global_get:.*]] = modelica.global_variable_get @var_0 : !modelica.array<2x!modelica.real>
// CHECK:           %[[load:.*]] = modelica.load %[[global_get]][%[[index]]]
// CHECK:           %[[cast:.*]] = modelica.cast %[[load]] : !modelica.real -> f64
// CHECK:           ida.return %[[cast]]
// CHECK-NEXT:  }

// CHECK:       ida.variable_getter @ida_main_getter_1(%[[index:.*]]: index) -> f64 {
// CHECK:           %[[global_get:.*]] = modelica.global_variable_get @var_1 : !modelica.array<2x!modelica.real>
// CHECK:           %[[load:.*]] = modelica.load %[[global_get]][%[[index]]]
// CHECK:           %[[cast:.*]] = modelica.cast %[[load]] : !modelica.real -> f64
// CHECK:           ida.return %[[cast]]
// CHECK-NEXT:  }

modelica.model @Test attributes {derivatives_map = [#modelica<var_derivative @x, @der_x>]} {
    modelica.variable @x : !modelica.variable<2x!modelica.real>
    modelica.variable @der_x : !modelica.variable<2x!modelica.real>
}

// -----

// 2-D array variables.

// CHECK:       ida.variable_getter @ida_main_getter_0(%[[index_0:.*]]: index, %[[index_1:.*]]: index) -> f64 {
// CHECK:           %[[global_get:.*]] = modelica.global_variable_get @var_0 : !modelica.array<2x3x!modelica.real>
// CHECK:           %[[load:.*]] = modelica.load %[[global_get]][%[[index_0]], %[[index_1]]]
// CHECK:           %[[cast:.*]] = modelica.cast %[[load]] : !modelica.real -> f64
// CHECK:           ida.return %[[cast]]
// CHECK-NEXT:  }

// CHECK:       ida.variable_getter @ida_main_getter_1(%[[index_0:.*]]: index, %[[index_1:.*]]: index) -> f64 {
// CHECK:           %[[global_get:.*]] = modelica.global_variable_get @var_1 : !modelica.array<2x3x!modelica.real>
// CHECK:           %[[load:.*]] = modelica.load %[[global_get]][%[[index_0]], %[[index_1]]]
// CHECK:           %[[cast:.*]] = modelica.cast %[[load]] : !modelica.real -> f64
// CHECK:           ida.return %[[cast]]
// CHECK-NEXT:  }

modelica.model @Test attributes {derivatives_map = [#modelica<var_derivative @x, @der_x>]} {
    modelica.variable @x : !modelica.variable<2x3x!modelica.real>
    modelica.variable @der_x : !modelica.variable<2x3x!modelica.real>
}
