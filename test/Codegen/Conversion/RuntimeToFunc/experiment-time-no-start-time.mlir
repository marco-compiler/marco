// RUN: modelica-opt %s --convert-runtime-to-func | FileCheck %s

// COM: StartTimeOp without value.

// CHECK:       func.func @hasExperimentStartTime() -> i1 {
// CHECK:           %[[cst:.*]] = arith.constant false
// CHECK:           return %[[cst]] : i1
// CHECK:       }

runtime.start_time
