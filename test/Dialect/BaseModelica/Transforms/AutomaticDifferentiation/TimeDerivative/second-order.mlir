// RUN: modelica-opt %s --split-input-file --auto-diff | FileCheck %s

// CHECK-LABEL: @add_der_2
// CHECK-NEXT:  bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @z : !bmodelica.variable<!bmodelica.real>
// CHECK-NEXT:  bmodelica.variable @der_x : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @der_y : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @der_z : !bmodelica.variable<!bmodelica.real>
// CHECK-NEXT:  bmodelica.variable @der_2_x : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @der_2_y : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @der_2_z : !bmodelica.variable<!bmodelica.real, output>
// CHECK-NEXT:  bmodelica.algorithm {
// CHECK-NEXT:      %[[der_x:.*]] = bmodelica.variable_get @der_x
// CHECK-NEXT:      %[[der_2_x:.*]] = bmodelica.variable_get @der_2_x
// CHECK-NEXT:      %[[der_y:.*]] = bmodelica.variable_get @der_y
// CHECK-NEXT:      %[[der_2_y:.*]] = bmodelica.variable_get @der_2_y
// CHECK-NEXT:      %[[add_der_x_der_y:.*]] = bmodelica.add %[[der_x]], %[[der_y]]
// CHECK-NEXT:      %[[add_der_2_x_der_2_y:.*]] = bmodelica.add %[[der_2_x]], %[[der_2_y]]
// CHECK-NEXT:      bmodelica.variable_set @der_z, %[[add_der_x_der_y]]
// CHECK-NEXT:      bmodelica.variable_set @der_2_z, %[[add_der_2_x_der_2_y]]
// CHECK-NEXT:  }

bmodelica.function @add_der attributes {derivative = #bmodelica<func_der "add_der_2", 2>, timeDerivativeOrder = 1 : i64} {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, input>
    bmodelica.variable @z : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @der_x : !bmodelica.variable<!bmodelica.real, input>
    bmodelica.variable @der_y : !bmodelica.variable<!bmodelica.real, input>
    bmodelica.variable @der_z : !bmodelica.variable<!bmodelica.real, output>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @der_x : !bmodelica.real
        %1 = bmodelica.variable_get @der_y : !bmodelica.real
        %2 = bmodelica.add %0, %1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        bmodelica.variable_set @der_z, %2 : !bmodelica.real
    }
}
