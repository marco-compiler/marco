// RUN: modelica-opt %s --split-input-file --convert-modelica-to-simulation | FileCheck %s

// CHECK-DAG: modelica.global_variable @[[global_var:.*]] : !modelica.array<!modelica.int>

// CHECK:       simulation.init_function {
// CHECK-DAG:       %[[zero:.*]] = modelica.constant #modelica.int<0>
// CHECK-DAG:       %[[x:.*]] = modelica.global_variable_get @[[global_var]]
// CHECK:           modelica.store %[[x]][], %[[zero]]
// CHECK-NEXT:      simulation.yield
// CHECK-NEXT:  }

module {
    modelica.model @Test {
        modelica.variable @x : !modelica.variable<!modelica.int>
    }
}

// -----

// CHECK-DAG: modelica.global_variable @[[global_var:.*]] : !modelica.array<3x!modelica.int>

// CHECK:       simulation.init_function {
// CHECK-DAG:       %[[zero:.*]] = modelica.constant #modelica.int<0>
// CHECK-DAG:       %[[x:.*]] = modelica.global_variable_get @[[global_var]]
// CHECK:           modelica.array_fill %[[x]], %[[zero]]
// CHECK-NEXT:      simulation.yield
// CHECK-NEXT:  }

module {
    modelica.model @Test {
        modelica.variable @x : !modelica.variable<3x!modelica.int>
    }
}
