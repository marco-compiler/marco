// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-runtime | FileCheck %s

// COM: Model with both start and end time annotations.

// CHECK-LABEL: module {
// CHECK:       runtime.start_time 0.000000e+00
// CHECK:       runtime.end_time 1.000000e+00

bmodelica.model @Test attributes {experiment.startTime = 0.000000e+00 : f64, experiment.endTime = 1.000000e+00 : f64} {

}

// -----

// COM: Model with only start time annotation.

// CHECK-LABEL: module {
// CHECK:       runtime.start_time 5.000000e-01
// CHECK:       runtime.end_time

bmodelica.model @Test2 attributes {experiment.startTime = 5.000000e-01 : f64} {

}

// -----

// COM: Model with only end time annotation.

// CHECK-LABEL: module {
// CHECK:       runtime.start_time
// CHECK:       runtime.end_time 2.000000e+00

bmodelica.model @Test3 attributes {experiment.endTime = 2.000000e+00 : f64} {

}

// -----

// COM: Model without experiment annotations.

// CHECK-LABEL: module {
// CHECK:       runtime.start_time
// CHECK:       runtime.end_time

bmodelica.model @Test4 {

}
