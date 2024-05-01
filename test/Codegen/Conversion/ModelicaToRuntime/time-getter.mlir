// RUN: modelica-opt %s --split-input-file --convert-modelica-to-runtime | FileCheck %s

// CHECK:       runtime.function @getTime() -> f64 {
// CHECK:           %[[time_get:.*]] = bmodelica.global_variable_get @time : !bmodelica.array<!bmodelica.real>
// CHECK:           %[[time_load:.*]] = bmodelica.load %[[time_get]][]
// CHECK:           %[[result:.*]] = bmodelica.cast %[[time_load]] : !bmodelica.real -> f64
// CHECK:           runtime.return %[[result]] : f64
// CHECK-NEXT:  }

bmodelica.model @Test {

}
