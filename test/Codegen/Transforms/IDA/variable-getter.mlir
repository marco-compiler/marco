// RUN: modelica-opt %s --split-input-file --ida | FileCheck %s

// Scalar variables.

// CHECK:       sundials.variable_getter @ida_main_getter_0() -> f64 {
// CHECK:           %[[get:.*]] = bmodelica.qualified_variable_get @Test::@x : !bmodelica.real
// CHECK:           %[[cast:.*]] = bmodelica.cast %[[get]] : !bmodelica.real -> f64
// CHECK:           sundials.return %[[cast]]
// CHECK-NEXT:  }

// CHECK:       sundials.variable_getter @ida_main_getter_1() -> f64 {
// CHECK:           %[[get:.*]] = bmodelica.qualified_variable_get @Test::@der_x : !bmodelica.real
// CHECK:           %[[cast:.*]] = bmodelica.cast %[[get]] : !bmodelica.real -> f64
// CHECK:           sundials.return %[[cast]]
// CHECK-NEXT:  }

bmodelica.model @Test attributes {derivatives_map = [#bmodelica<var_derivative @x, @der_x>]} {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @der_x : !bmodelica.variable<!bmodelica.real>
}

// -----

// 1-D array variables.

// CHECK:       sundials.variable_getter @ida_main_getter_0(%[[index:.*]]: index) -> f64 {
// CHECK:           %[[get:.*]] = bmodelica.qualified_variable_get @Test::@x : !bmodelica.array<2x!bmodelica.real>
// CHECK:           %[[load:.*]] = bmodelica.load %[[get]][%[[index]]]
// CHECK:           %[[cast:.*]] = bmodelica.cast %[[load]] : !bmodelica.real -> f64
// CHECK:           sundials.return %[[cast]]
// CHECK-NEXT:  }

// CHECK:       sundials.variable_getter @ida_main_getter_1(%[[index:.*]]: index) -> f64 {
// CHECK:           %[[get:.*]] = bmodelica.qualified_variable_get @Test::@der_x : !bmodelica.array<2x!bmodelica.real>
// CHECK:           %[[load:.*]] = bmodelica.load %[[get]][%[[index]]]
// CHECK:           %[[cast:.*]] = bmodelica.cast %[[load]] : !bmodelica.real -> f64
// CHECK:           sundials.return %[[cast]]
// CHECK-NEXT:  }

bmodelica.model @Test attributes {derivatives_map = [#bmodelica<var_derivative @x, @der_x>]} {
    bmodelica.variable @x : !bmodelica.variable<2x!bmodelica.real>
    bmodelica.variable @der_x : !bmodelica.variable<2x!bmodelica.real>
}

// -----

// 2-D array variables.

// CHECK:       sundials.variable_getter @ida_main_getter_0(%[[index_0:.*]]: index, %[[index_1:.*]]: index) -> f64 {
// CHECK:           %[[get:.*]] = bmodelica.qualified_variable_get @Test::@x : !bmodelica.array<2x3x!bmodelica.real>
// CHECK:           %[[load:.*]] = bmodelica.load %[[get]][%[[index_0]], %[[index_1]]]
// CHECK:           %[[cast:.*]] = bmodelica.cast %[[load]] : !bmodelica.real -> f64
// CHECK:           sundials.return %[[cast]]
// CHECK-NEXT:  }

// CHECK:       sundials.variable_getter @ida_main_getter_1(%[[index_0:.*]]: index, %[[index_1:.*]]: index) -> f64 {
// CHECK:           %[[get:.*]] = bmodelica.qualified_variable_get @Test::@der_x : !bmodelica.array<2x3x!bmodelica.real>
// CHECK:           %[[load:.*]] = bmodelica.load %[[get]][%[[index_0]], %[[index_1]]]
// CHECK:           %[[cast:.*]] = bmodelica.cast %[[load]] : !bmodelica.real -> f64
// CHECK:           sundials.return %[[cast]]
// CHECK-NEXT:  }

bmodelica.model @Test attributes {derivatives_map = [#bmodelica<var_derivative @x, @der_x>]} {
    bmodelica.variable @x : !bmodelica.variable<2x3x!bmodelica.real>
    bmodelica.variable @der_x : !bmodelica.variable<2x3x!bmodelica.real>
}
