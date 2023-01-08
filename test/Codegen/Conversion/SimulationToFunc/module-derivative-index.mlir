// RUN: modelica-opt %s --split-input-file --convert-simulation-to-func | FileCheck %s

// No variables.

// CHECK:       func.func @getDerivative(%[[variable:.*]]: i64) -> i64 {
// CHECK-NEXT:      %[[default:.*]] = arith.constant -1 : i64
// CHECK-NEXT:      cf.switch %[[variable]] : i64, [
// CHECK-NEXT:          default: ^[[out:.*]](%[[default]] : i64)
// CHECK-NEXT:      ]
// CHECK-NEXT:  ^[[out]](%[[result:.*]]: i64):
// CHECK-NEXT:      return %[[result]]
// CHECK-NEXT:  }

simulation.module {

}

// -----

// No derivatives

// CHECK:       func.func @getDerivative(%[[variable:.*]]: i64) -> i64 {
// CHECK-NEXT:      %[[default:.*]] = arith.constant -1 : i64
// CHECK-NEXT:      cf.switch %[[variable]] : i64, [
// CHECK-NEXT:          default: ^[[out:.*]](%[[default]] : i64)
// CHECK-NEXT:      ]
// CHECK-NEXT:  ^[[out]](%[[result:.*]]: i64):
// CHECK-NEXT:      return %[[result]]
// CHECK-NEXT:  }

#var0 = #simulation.variable<name = "x">

simulation.module attributes {variables = [#var0]} {

}

// -----

// CHECK:       func.func @getDerivative(%[[variable:.*]]: i64) -> i64 {
// CHECK-NEXT:      %[[default:.*]] = arith.constant -1 : i64
// CHECK-NEXT:      cf.switch %[[variable]] : i64, [
// CHECK-NEXT:          default: ^[[out:.*]](%[[default]] : i64)
// CHECK-NEXT:          0: ^[[xBlock:.*]]
// CHECK-NEXT:      ]
// CHECK-NEXT:  ^[[xBlock]]:
// CHECK-NEXT:      %[[xDer:.*]] = arith.constant 1 : i64
// CHECK-NEXT:      cf.br ^[[out]](%[[xDer]] : i64)
// CHECK-NEXT:  ^[[out]](%[[result:.*]]: i64):
// CHECK-NEXT:      return %[[result]]
// CHECK-NEXT:  }

#var0 = #simulation.variable<name = "x">
#var1 = #simulation.variable<name = "der_x">
#der0 = #simulation.derivative<#var0 -> #var1>

simulation.module attributes {variables = [#var0, #var1], derivatives = [#der0]} {

}
