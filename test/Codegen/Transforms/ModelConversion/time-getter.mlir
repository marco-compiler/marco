// RUN: modelica-opt %s --split-input-file --test-model-conversion | FileCheck %s

// CHECK:       simulation.function @getTime() -> f64 {
// CHECK:           %[[time_get:.*]] = modelica.global_variable_get @time : !modelica.array<!modelica.real>
// CHECK:           %[[time_load:.*]] = modelica.load %[[time_get]][]
// CHECK:           %[[result:.*]] = modelica.cast %[[time_load]] : !modelica.real -> f64
// CHECK:           simulation.return %[[result]] : f64
// CHECK-NEXT:  }

modelica.model @Test {

}
