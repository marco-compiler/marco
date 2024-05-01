// RUN: modelica-opt %s --split-input-file --convert-modelica-to-runtime | FileCheck %s

// CHECK:       runtime.function @setTime(%[[newTime:.*]]: f64) {
// CHECK:           %[[time_get:.*]] = modelica.global_variable_get @time : !modelica.array<!modelica.real>
// CHECK:           %[[newTime_cast:.*]] = modelica.cast %[[newTime]] : f64 -> !modelica.real
// CHECK:           modelica.store %[[time_get]][], %[[newTime_cast]]
// CHECK:           runtime.return
// CHECK-NEXT:  }

modelica.model @Test {

}
