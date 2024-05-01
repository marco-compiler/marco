// RUN: modelica-opt %s --split-input-file --convert-modelica-to-runtime | FileCheck %s

// CHECK:       runtime.function @setTime(%[[newTime:.*]]: f64) {
// CHECK:           %[[time_get:.*]] = bmodelica.global_variable_get @time : !bmodelica.array<!bmodelica.real>
// CHECK:           %[[newTime_cast:.*]] = bmodelica.cast %[[newTime]] : f64 -> !bmodelica.real
// CHECK:           bmodelica.store %[[time_get]][], %[[newTime_cast]]
// CHECK:           runtime.return
// CHECK-NEXT:  }

bmodelica.model @Test {

}
