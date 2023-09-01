// RUN: modelica-opt %s --split-input-file --auto-diff | FileCheck %s

// CHECK-LABEL: @sin_der
// CHECK:   modelica.algorithm {
// CHECK:       %[[der_x:.*]] = modelica.variable_get @der_x
// CHECK:       %[[x:.*]] = modelica.variable_get @x
// CHECK:       %[[cos:.*]] = modelica.cos %[[x]]
// CHECK:       %[[mul:.*]] = modelica.mul_ew %[[cos]], %[[der_x]]
// CHECK:       modelica.variable_set @y, %[[mul]]
// CHECK:   }

modelica.function @sin_der {
    modelica.variable @x : !modelica.variable<!modelica.real, input>
    modelica.variable @der_x : !modelica.variable<!modelica.real, input>
    modelica.variable @y : !modelica.variable<!modelica.real, output>

    modelica.algorithm {
        %0 = modelica.variable_get @x : !modelica.real
        %1 = modelica.sin %0 : !modelica.real -> !modelica.real
        %2 = modelica.der %1 : !modelica.real -> !modelica.real
        modelica.variable_set @y, %2 : !modelica.real
    }
}

// -----

// CHECK-LABEL: @cos_der
// CHECK:  modelica.algorithm {
// CHECK:       %[[der_x:.*]] = modelica.variable_get @der_x
// CHECK:       %[[x:.*]] = modelica.variable_get @x
// CHECK:       %[[sin:.*]] = modelica.sin %[[x]]
// CHECK:       %[[neg:.*]] = modelica.neg %[[sin]]
// CHECK:       %[[mul:.*]] = modelica.mul_ew %[[neg]], %[[der_x]]
// CHECK:       modelica.variable_set @y, %[[mul]]
// CHECK:   }

modelica.function @cos_der {
    modelica.variable @x : !modelica.variable<!modelica.real, input>
    modelica.variable @der_x : !modelica.variable<!modelica.real, input>
    modelica.variable @y : !modelica.variable<!modelica.real, output>

    modelica.algorithm {
        %0 = modelica.variable_get @x : !modelica.real
        %1 = modelica.cos %0 : !modelica.real -> !modelica.real
        %2 = modelica.der %1 : !modelica.real -> !modelica.real
        modelica.variable_set @y, %2 : !modelica.real
    }
}
