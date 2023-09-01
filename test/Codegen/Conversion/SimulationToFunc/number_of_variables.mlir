// RUN: modelica-opt %s --split-input-file --convert-simulation-to-func | FileCheck %s

// CHECK:       func.func @getNumOfVariables() -> i64 {
// CHECK-NEXT:      %[[result:.*]] = arith.constant 2 : i64
// CHECK-NEXT:      return %[[result]]
// CHECK-NEXT:  }

simulation.number_of_variables 2
