// RUN: modelica-opt %s --split-input-file --convert-runtime-to-func | FileCheck %s

// COM: StartTimeOp with value.

// CHECK:       func.func @hasExperimentStartTime() -> i1 {
// CHECK-NEXT:      %[[cst:.*]] = arith.constant true
// CHECK-NEXT:      return %[[cst]] : i1
// CHECK-NEXT:  }
// CHECK:       func.func @getExperimentStartTime() -> f64 {
// CHECK-NEXT:      %[[cst:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:      return %[[cst]] : f64
// CHECK-NEXT:  }

runtime.start_time 0.0

// -----

// COM: StartTimeOp without value.

// CHECK:       func.func @hasExperimentStartTime() -> i1 {
// CHECK:           %[[cst:.*]] = arith.constant false
// CHECK:           return %[[cst]] : i1
// CHECK:       }

runtime.start_time unspecified

// -----

// COM: EndTimeOp with value.

// CHECK:       func.func @hasExperimentEndTime() -> i1 {
// CHECK-NEXT:      %[[cst:.*]] = arith.constant true
// CHECK-NEXT:      return %[[cst]] : i1
// CHECK-NEXT:  }
// CHECK:       func.func @getExperimentEndTime() -> f64 {
// CHECK-NEXT:      %[[cst:.*]] = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:      return %[[cst]] : f64
// CHECK-NEXT:  }

runtime.end_time 1.0

// -----

// COM: EndTimeOp without value.

// CHECK:       func.func @hasExperimentEndTime() -> i1 {
// CHECK:           %[[cst:.*]] = arith.constant false
// CHECK:           return %[[cst]] : i1
// CHECK:       }

runtime.end_time unspecified

// -----

// COM: Both StartTimeOp and EndTimeOp with values.

// CHECK:       func.func @hasExperimentStartTime() -> i1 {
// CHECK-NEXT:      %[[cst:.*]] = arith.constant true
// CHECK-NEXT:      return %[[cst]] : i1
// CHECK-NEXT:  }
// CHECK:       func.func @getExperimentStartTime() -> f64 {
// CHECK-NEXT:      %[[cst:.*]] = arith.constant 2.500000e-01 : f64
// CHECK-NEXT:      return %[[cst]] : f64
// CHECK-NEXT:  }
// CHECK:       func.func @hasExperimentEndTime() -> i1 {
// CHECK-NEXT:      %[[cst:.*]] = arith.constant true
// CHECK-NEXT:      return %[[cst]] : i1
// CHECK-NEXT:  }
// CHECK:       func.func @getExperimentEndTime() -> f64 {
// CHECK-NEXT:      %[[cst:.*]] = arith.constant 5.000000e+00 : f64
// CHECK-NEXT:      return %[[cst]] : f64
// CHECK-NEXT:  }

runtime.start_time 0.25
runtime.end_time 5.0

