// RUN: modelica-opt %s --convert-runtime-to-func | FileCheck %s

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
