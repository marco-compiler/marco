// RUN: modelica-opt %s --split-input-file --auto-diff | FileCheck %s

// CHECK-LABEL: @sin_der
// CHECK:   bmodelica.algorithm {
// CHECK:       %[[der_x:.*]] = bmodelica.variable_get @der_x
// CHECK:       %[[x:.*]] = bmodelica.variable_get @x
// CHECK:       %[[cos:.*]] = bmodelica.cos %[[x]]
// CHECK:       %[[mul:.*]] = bmodelica.mul_ew %[[cos]], %[[der_x]]
// CHECK:       bmodelica.variable_set @y, %[[mul]]
// CHECK:   }

bmodelica.function @sin_der {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
    bmodelica.variable @der_x : !bmodelica.variable<!bmodelica.real, input>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, output>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x : !bmodelica.real
        %1 = bmodelica.sin %0 : !bmodelica.real -> !bmodelica.real
        %2 = bmodelica.der %1 : !bmodelica.real -> !bmodelica.real
        bmodelica.variable_set @y, %2 : !bmodelica.real
    }
}

// -----

// CHECK-LABEL: @cos_der
// CHECK:  bmodelica.algorithm {
// CHECK:       %[[der_x:.*]] = bmodelica.variable_get @der_x
// CHECK:       %[[x:.*]] = bmodelica.variable_get @x
// CHECK:       %[[sin:.*]] = bmodelica.sin %[[x]]
// CHECK:       %[[neg:.*]] = bmodelica.neg %[[sin]]
// CHECK:       %[[mul:.*]] = bmodelica.mul_ew %[[neg]], %[[der_x]]
// CHECK:       bmodelica.variable_set @y, %[[mul]]
// CHECK:   }

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
