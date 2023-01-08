// RUN: modelica-opt %s --split-input-file --convert-simulation-to-func | FileCheck %s

// No variables

// CHECK:       func.func @getNumOfVariables() -> i64 {
// CHECK-NEXT:      %[[result:.*]] = arith.constant 0 : i64
// CHECK-NEXT:      return %[[result]]
// CHECK-NEXT:  }

simulation.module attributes {variables = []} {

}

// -----

// CHECK:       func.func @getNumOfVariables() -> i64 {
// CHECK-NEXT:      %[[result:.*]] = arith.constant 2 : i64
// CHECK-NEXT:      return %[[result]]
// CHECK-NEXT:  }

#var0 = #simulation.variable<name = "var0">
#var1 = #simulation.variable<name = "var1">

simulation.module attributes {variables = [#var0, #var1]} {

}
