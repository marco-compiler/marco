// RUN: modelica-opt %s --split-input-file --ida | FileCheck %s

// Scalar variables.

// CHECK:       sundials.variable_setter @ida_main_setter_0(%[[value:.*]]: f64) {
// CHECK:           %[[cast:.*]] = bmodelica.cast %[[value]] : f64 -> !bmodelica.real
// CHECK:           bmodelica.qualified_variable_set @Test::@x, %[[cast]]
// CHECK:           sundials.return
// CHECK-NEXT:  }

// CHECK:       sundials.variable_setter @ida_main_setter_1(%[[value:.*]]: f64) {
// CHECK:           %[[cast:.*]] = bmodelica.cast %[[value]] : f64 -> !bmodelica.real
// CHECK:           bmodelica.qualified_variable_set @Test::@der_x, %[[cast]]
// CHECK:           sundials.return
// CHECK-NEXT:  }

bmodelica.model @Test attributes {derivatives_map = [#bmodelica<var_derivative @x, @der_x>]} {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @der_x : !bmodelica.variable<!bmodelica.real>
}

// -----

// 1-D array variables.

// CHECK:       sundials.variable_setter @ida_main_setter_0(%[[value:.*]]: f64, %[[index:.*]]: index) {
// CHECK:           %[[cast:.*]] = bmodelica.cast %[[value]] : f64 -> !bmodelica.real
// CHECK:           bmodelica.qualified_variable_set @Test::@x[%[[index]]], %[[cast]]
// CHECK:           sundials.return
// CHECK-NEXT:  }

// CHECK:       sundials.variable_setter @ida_main_setter_1(%[[value:.*]]: f64, %[[index:.*]]: index) {
// CHECK:           %[[cast:.*]] = bmodelica.cast %[[value]] : f64 -> !bmodelica.real
// CHECK:           bmodelica.qualified_variable_set @Test::@der_x[%[[index]]], %[[cast]]
// CHECK:           sundials.return
// CHECK-NEXT:  }

bmodelica.model @Test attributes {derivatives_map = [#bmodelica<var_derivative @x, @der_x>]} {
    bmodelica.variable @x : !bmodelica.variable<2x!bmodelica.real>
    bmodelica.variable @der_x : !bmodelica.variable<2x!bmodelica.real>
}

// -----

// 2-D array variables.

// CHECK:       sundials.variable_setter @ida_main_setter_0(%[[value:.*]]: f64, %[[index_0:.*]]: index, %[[index_1:.*]]: index) {
// CHECK:           %[[cast:.*]] = bmodelica.cast %[[value]] : f64 -> !bmodelica.real
// CHECK:           bmodelica.qualified_variable_set @Test::@x[%[[index_0]], %[[index_1]]], %[[cast]]
// CHECK:           sundials.return
// CHECK-NEXT:  }

// CHECK:       sundials.variable_setter @ida_main_setter_1(%[[value:.*]]: f64, %[[index_0:.*]]: index, %[[index_1:.*]]: index) {
// CHECK:           %[[cast:.*]] = bmodelica.cast %[[value]] : f64 -> !bmodelica.real
// CHECK:           bmodelica.qualified_variable_set @Test::@der_x[%[[index_0]], %[[index_1]]], %[[cast]]
// CHECK:           sundials.return
// CHECK-NEXT:  }

bmodelica.model @Test attributes {derivatives_map = [#bmodelica<var_derivative @x, @der_x>]} {
    bmodelica.variable @x : !bmodelica.variable<2x3x!bmodelica.real>
    bmodelica.variable @der_x : !bmodelica.variable<2x3x!bmodelica.real>
}
