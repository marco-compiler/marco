// RUN: modelica-opt %s --split-input-file --der-chain-rule | FileCheck %s

// CHECK-LABEL: @cos_der
// CHECK:       bmodelica.algorithm {
// CHECK-NEXT:      %[[x:.*]] = bmodelica.variable_get @x
// CHECK-NEXT:      %[[der_x:.*]] = bmodelica.variable_get @der_x
// CHECK-NEXT:      %[[sin:.*]] = bmodelica.sin %[[x]]
// CHECK-NEXT:      %[[neg:.*]] = bmodelica.neg %[[sin]]
// CHECK-NEXT:      %[[mul:.*]] = bmodelica.mul_ew %[[neg]], %[[der_x]]
// CHECK-NEXT:      bmodelica.variable_set @y, %[[mul]]
// CHECK-NEXT:  }

bmodelica.function @cos_der {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
    bmodelica.variable @der_x : !bmodelica.variable<!bmodelica.real, input>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, output>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x : !bmodelica.real
        %1 = bmodelica.cos %0 : !bmodelica.real -> !bmodelica.real
        %2 = bmodelica.der %1 : !bmodelica.real -> !bmodelica.real
        bmodelica.variable_set @y, %2 : !bmodelica.real
    }
}
