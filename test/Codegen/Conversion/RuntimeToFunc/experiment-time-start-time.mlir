// RUN: modelica-opt %s --convert-runtime-to-func | FileCheck %s

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
