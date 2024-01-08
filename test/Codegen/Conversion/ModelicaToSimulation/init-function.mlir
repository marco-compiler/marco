// RUN: modelica-opt %s --split-input-file --convert-modelica-to-simulation | FileCheck %s

// CHECK-DAG: modelica.global_variable @[[var0:.*]] : !modelica.array<!modelica.int>
// CHECK-DAG: modelica.global_variable @[[var1:.*]] : !modelica.array<3x!modelica.int>

// CHECK:       simulation.init_function {
// CHECK-DAG:       %[[x_value:.*]] = modelica.constant #modelica.int<1>
// CHECK-DAG:       %[[y_value:.*]] = modelica.constant #modelica.int<0>
// CHECK-DAG:       %[[x:.*]] = modelica.global_variable_get @[[var0]]
// CHECK-DAG:       %[[y:.*]] = modelica.global_variable_get @[[var1]]
// CHECK-DAG:       modelica.store %[[x]][], %[[x_value]]
// CHECK-DAG:       modelica.array_fill %[[y]], %[[y_value]]
// CHECK-NEXT:      simulation.yield
// CHECK-NEXT:  }

module {
    modelica.model @Test {
        modelica.variable @x : !modelica.variable<!modelica.int>
        modelica.variable @y : !modelica.variable<3x!modelica.int>

        modelica.start @x {
            %0 = modelica.constant #modelica.int<1>
            modelica.yield %0 : !modelica.int
        } {each = false, fixed = false}
    }

    modelica.simulation_variable @x : !modelica.variable<!modelica.int>
    modelica.simulation_variable @y : !modelica.variable<3x!modelica.int>
}
