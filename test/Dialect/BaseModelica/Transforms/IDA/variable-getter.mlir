// RUN: modelica-opt %s --split-input-file --ida | FileCheck %s

// CHECK:       sundials.variable_getter @ida_main_getter_0() -> f64 {
// CHECK:           %[[get:.*]] = bmodelica.qualified_variable_get @scalarVariables::@x : !bmodelica.real
// CHECK:           %[[cast:.*]] = bmodelica.cast %[[get]] : !bmodelica.real -> f64
// CHECK:           sundials.return %[[cast]]
// CHECK-NEXT:  }

// CHECK:       sundials.variable_getter @ida_main_getter_1() -> f64 {
// CHECK:           %[[get:.*]] = bmodelica.qualified_variable_get @scalarVariables::@der_x : !bmodelica.real
// CHECK:           %[[cast:.*]] = bmodelica.cast %[[get]] : !bmodelica.real -> f64
// CHECK:           sundials.return %[[cast]]
// CHECK-NEXT:  }

bmodelica.model @scalarVariables der = [<@x, @der_x>] {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @der_x : !bmodelica.variable<!bmodelica.real>
}

// -----

// CHECK:       sundials.variable_getter @ida_main_getter_0(%[[index:.*]]: index) -> f64 {
// CHECK:           %[[get:.*]] = bmodelica.qualified_variable_get @arrayVariables1D::@x : tensor<2x!bmodelica.real>
// CHECK:           %[[load:.*]] = bmodelica.tensor_extract %[[get]][%[[index]]]
// CHECK:           %[[cast:.*]] = bmodelica.cast %[[load]] : !bmodelica.real -> f64
// CHECK:           sundials.return %[[cast]]
// CHECK-NEXT:  }

// CHECK:       sundials.variable_getter @ida_main_getter_1(%[[index:.*]]: index) -> f64 {
// CHECK:           %[[get:.*]] = bmodelica.qualified_variable_get @arrayVariables1D::@der_x : tensor<2x!bmodelica.real>
// CHECK:           %[[load:.*]] = bmodelica.tensor_extract %[[get]][%[[index]]]
// CHECK:           %[[cast:.*]] = bmodelica.cast %[[load]] : !bmodelica.real -> f64
// CHECK:           sundials.return %[[cast]]
// CHECK-NEXT:  }

bmodelica.model @arrayVariables1D der = [<@x, @der_x>] {
    bmodelica.variable @x : !bmodelica.variable<2x!bmodelica.real>
    bmodelica.variable @der_x : !bmodelica.variable<2x!bmodelica.real>
}

// -----

// CHECK:       sundials.variable_getter @ida_main_getter_0(%[[index_0:.*]]: index, %[[index_1:.*]]: index) -> f64 {
// CHECK:           %[[get:.*]] = bmodelica.qualified_variable_get @arrayVariables2D::@x : tensor<2x3x!bmodelica.real>
// CHECK:           %[[load:.*]] = bmodelica.tensor_extract %[[get]][%[[index_0]], %[[index_1]]]
// CHECK:           %[[cast:.*]] = bmodelica.cast %[[load]] : !bmodelica.real -> f64
// CHECK:           sundials.return %[[cast]]
// CHECK-NEXT:  }

// CHECK:       sundials.variable_getter @ida_main_getter_1(%[[index_0:.*]]: index, %[[index_1:.*]]: index) -> f64 {
// CHECK:           %[[get:.*]] = bmodelica.qualified_variable_get @arrayVariables2D::@der_x : tensor<2x3x!bmodelica.real>
// CHECK:           %[[load:.*]] = bmodelica.tensor_extract %[[get]][%[[index_0]], %[[index_1]]]
// CHECK:           %[[cast:.*]] = bmodelica.cast %[[load]] : !bmodelica.real -> f64
// CHECK:           sundials.return %[[cast]]
// CHECK-NEXT:  }

bmodelica.model @arrayVariables2D der = [<@x, @der_x>] {
    bmodelica.variable @x : !bmodelica.variable<2x3x!bmodelica.real>
    bmodelica.variable @der_x : !bmodelica.variable<2x3x!bmodelica.real>
}
