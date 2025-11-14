// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-runtime | FileCheck %s

// COM: Model with both start and end time annotations.

// CHECK-LABEL: module @BothStartAndEndTime
// CHECK:       runtime.start_time 0.000000e+00
// CHECK:       runtime.end_time 1.000000e+00

module @BothStartAndEndTime {
  bmodelica.model @Test attributes {
    experiment.startTime = 0.000000e+00 : f64, experiment.endTime = 1.000000e+00 : f64
  } {}
}

// -----

// COM: Model with only start time annotation.

// CHECK-LABEL: module @OnlyStartTime
// CHECK:       runtime.start_time 5.000000e-01
// CHECK:       runtime.end_time

module @OnlyStartTime {
  bmodelica.model @Test attributes {
    experiment.startTime = 5.000000e-01 : f64
  } {}
}

// -----

// COM: Model with only end time annotation.

// CHECK-LABEL: module @OnlyEndTime
// CHECK:       runtime.start_time
// CHECK:       runtime.end_time 2.000000e+00

module @OnlyEndTime {
  bmodelica.model @Test attributes {
    experiment.endTime = 2.000000e+00 : f64
  } {}
}

// -----

// COM: Model without experiment annotations.

// CHECK-LABEL: module @NoExperimentAnnotations
// CHECK:       runtime.start_time
// CHECK:       runtime.end_time

module @NoExperimentAnnotations {
  bmodelica.model @Test {}
}
