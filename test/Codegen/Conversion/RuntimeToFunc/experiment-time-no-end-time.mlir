// RUN: modelica-opt %s --convert-runtime-to-func | FileCheck %s

// COM: EndTimeOp without value.

// CHECK:       func.func @hasExperimentEndTime() -> i1 {
// CHECK:           %[[cst:.*]] = arith.constant false
// CHECK:           return %[[cst]] : i1
// CHECK:       }

runtime.end_time
