// RUN: modelica-opt %s --split-input-file --convert-runtime-model-metadata | FileCheck %s

// CHECK:       func.func @getNumOfVariables() -> i64 {
// CHECK-NEXT:      %[[result:.*]] = arith.constant 2 : i64
// CHECK-NEXT:      return %[[result]]
// CHECK-NEXT:  }

runtime.number_of_variables 2
