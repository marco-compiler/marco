// RUN: modelica-opt %s --split-input-file --ida | FileCheck %s

// CHECK:       sundials.variable_setter @ida_main_setter_0(%[[value:.*]]: f64) {
// CHECK:           %[[cast:.*]] = bmodelica.cast %[[value]] : f64 -> !bmodelica.real
// CHECK:           bmodelica.qualified_variable_set @scalarVariables::@x, %[[cast]]
// CHECK:           sundials.return
// CHECK-NEXT:  }

// CHECK:       sundials.variable_setter @ida_main_setter_1(%[[value:.*]]: f64) {
// CHECK:           %[[cast:.*]] = bmodelica.cast %[[value]] : f64 -> !bmodelica.real
// CHECK:           bmodelica.qualified_variable_set @scalarVariables::@der_x, %[[cast]]
// CHECK:           sundials.return
// CHECK-NEXT:  }

bmodelica.model @scalarVariables der = [<@x, @der_x>] {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @der_x : !bmodelica.variable<!bmodelica.real>
}

// -----

// CHECK:       sundials.variable_setter @ida_main_setter_0(%[[value:.*]]: f64, %[[index:.*]]: index) {
// CHECK:           %[[cast:.*]] = bmodelica.cast %[[value]] : f64 -> !bmodelica.real
// CHECK:           bmodelica.qualified_variable_set @arrayVariables1D::@x[%[[index]]], %[[cast]]
// CHECK:           sundials.return
// CHECK-NEXT:  }

// CHECK:       sundials.variable_setter @ida_main_setter_1(%[[value:.*]]: f64, %[[index:.*]]: index) {
// CHECK:           %[[cast:.*]] = bmodelica.cast %[[value]] : f64 -> !bmodelica.real
// CHECK:           bmodelica.qualified_variable_set @arrayVariables1D::@der_x[%[[index]]], %[[cast]]
// CHECK:           sundials.return
// CHECK-NEXT:  }

bmodelica.model @arrayVariables1D der = [<@x, @der_x>] {
    bmodelica.variable @x : !bmodelica.variable<2x!bmodelica.real>
    bmodelica.variable @der_x : !bmodelica.variable<2x!bmodelica.real>
}

// -----

// CHECK:       sundials.variable_setter @ida_main_setter_0(%[[value:.*]]: f64, %[[index_0:.*]]: index, %[[index_1:.*]]: index) {
// CHECK:           %[[cast:.*]] = bmodelica.cast %[[value]] : f64 -> !bmodelica.real
// CHECK:           bmodelica.qualified_variable_set @arrayVariables2D::@x[%[[index_0]], %[[index_1]]], %[[cast]]
// CHECK:           sundials.return
// CHECK-NEXT:  }

// CHECK:       sundials.variable_setter @ida_main_setter_1(%[[value:.*]]: f64, %[[index_0:.*]]: index, %[[index_1:.*]]: index) {
// CHECK:           %[[cast:.*]] = bmodelica.cast %[[value]] : f64 -> !bmodelica.real
// CHECK:           bmodelica.qualified_variable_set @arrayVariables2D::@der_x[%[[index_0]], %[[index_1]]], %[[cast]]
// CHECK:           sundials.return
// CHECK-NEXT:  }

bmodelica.model @arrayVariables2D der = [<@x, @der_x>] {
    bmodelica.variable @x : !bmodelica.variable<2x3x!bmodelica.real>
    bmodelica.variable @der_x : !bmodelica.variable<2x3x!bmodelica.real>
}
