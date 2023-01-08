// RUN: modelica-opt %s --split-input-file --convert-simulation-to-func | FileCheck %s

// No variables

// CHECK:       func.func @isPrintable(%[[variable:.*]]: i64) -> i1 {
// CHECK-NEXT:      %[[false:.*]] = arith.constant false
// CHECK-NEXT:      cf.switch %[[variable]] : i64, [
// CHECK-NEXT:          default: ^[[out:.*]](%[[false]] : i1)
// CHECK-NEXT:      ]
// CHECK-NEXT:      ^[[out]](%[[result:.*]]: i1):
// CHECK-NEXT:          return %[[result]]
// CHECK-NEXT:  }

simulation.module {

}

// -----

// Default printable property

// CHECK:       func.func @isPrintable(%[[variable:.*]]: i64) -> i1 {
// CHECK-NEXT:      %[[false:.*]] = arith.constant false
// CHECK-NEXT:      cf.switch %[[variable]] : i64, [
// CHECK-NEXT:          default: ^[[out:.*]](%[[false]] : i1)
// CHECK-NEXT:      ]
// CHECK-NEXT:      ^[[out]](%[[result:.*]]: i1):
// CHECK-NEXT:          return %[[result]]
// CHECK-NEXT:  }

#var0 = #simulation.variable<name = "x">

simulation.module attributes {variables = [#var0]} {

}

// -----

// Variable explicitly set as non printable

// CHECK:       func.func @isPrintable(%[[variable:.*]]: i64) -> i1 {
// CHECK-NEXT:      %[[false:.*]] = arith.constant false
// CHECK-NEXT:      cf.switch %[[variable]] : i64, [
// CHECK-NEXT:          default: ^[[out:.*]](%[[false]] : i1)
// CHECK-NEXT:      ]
// CHECK-NEXT:      ^[[out]](%[[result:.*]]: i1):
// CHECK-NEXT:          return %[[result]]
// CHECK-NEXT:  }

#var0 = #simulation.variable<name = "x", printable = false>

simulation.module attributes {variables = [#var0]} {

}

// -----

// Variable explicitly set as printable

// CHECK:       func.func @isPrintable(%[[variable:.*]]: i64) -> i1 {
// CHECK-NEXT:      %[[false:.*]] = arith.constant false
// CHECK-NEXT:      cf.switch %[[variable]] : i64, [
// CHECK-DAG:           default: ^[[out:.*]](%[[false]] : i1)
// CHECK-DAG:           0: ^[[printable:.*]]
// CHECK-NEXT:      ]
// CHECK-NEXT:      ^[[printable]]:
// CHECK-NEXT:          %[[true:.*]] = arith.constant true
// CHECK-NEXT:          cf.br ^[[out]](%[[true]] : i1)
// CHECK-NEXT:      ^[[out]](%[[result:.*]]: i1):
// CHECK-NEXT:          return %[[result]]
// CHECK-NEXT:  }

#var0 = #simulation.variable<name = "x", printable = true>

simulation.module attributes {variables = [#var0]} {

}

// -----

// Multiple printable variables

// CHECK:       func.func @isPrintable(%[[variable:.*]]: i64) -> i1 {
// CHECK-DAG:       %[[false:.*]] = arith.constant false
// CHECK-NEXT:      cf.switch %[[variable]] : i64, [
// CHECK-DAG:           default: ^[[out:.*]](%[[false]] : i1)
// CHECK-DAG:           0: ^[[printable:.*]]
// CHECK-DAG:           1: ^[[printable:.*]]
// CHECK-NEXT:      ]
// CHECK-NEXT:      ^[[printable]]:
// CHECK-DAG:           %[[true:.*]] = arith.constant true
// CHECK-NEXT:          cf.br ^[[out]](%[[true]] : i1)
// CHECK-NEXT:      ^[[out]](%[[result:.*]]: i1):
// CHECK-NEXT:          return %[[result]]
// CHECK-NEXT:  }

#var0 = #simulation.variable<name = "x", printable = true>
#var1 = #simulation.variable<name = "y", printable = true>

simulation.module attributes {variables = [#var0, #var1]} {

}
