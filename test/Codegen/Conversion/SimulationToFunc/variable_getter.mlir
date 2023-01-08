// RUN: modelica-opt %s --split-input-file --convert-simulation-to-func | FileCheck %s

// No variables

// CHECK:       func.func @getVariableValue(%[[opaquePtr:.*]]: !llvm.ptr<i8>, %[[var:.*]]: i64, %[[indices:.*]]: !llvm.ptr<i64>) -> f64 {
// CHECK:           %[[default:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:           cf.switch %[[var]] : i64, [
// CHECK-NEXT:          default: ^[[out:.*]](%[[default]] : f64)
// CHECK-NEXT:      ]
// CHECK:           ^[[out]](%[[result:.*]]: f64):
// CHECK-NEXT:          return %[[result]] : f64

simulation.module {

}

// -----

// Single variable

// CHECK:       func.func @getVariableValue(%[[opaquePtr:.*]]: !llvm.ptr<i8>, %[[var:.*]]: i64, %[[indices:.*]]: !llvm.ptr<i64>) -> f64 {
// CHECK-NEXT:      %[[ptr:.*]] = llvm.bitcast %arg0 : !llvm.ptr<i8> to !llvm.ptr<struct<(ptr<i8>, f64, f64)>>
// CHECK-NEXT:      %[[data:.*]] = llvm.load %[[ptr]] : !llvm.ptr<struct<(ptr<i8>, f64, f64)>>
// CHECK-NEXT:      %[[default:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:           cf.switch %[[var]] : i64, [
// CHECK-DAG:           default: ^[[out:.*]](%[[default]] : f64)
// CHECK-DAG:           0: ^[[var0Block:.*]]
// CHECK-NEXT:      ]
// CHECK:           ^[[var0Block]]:
// CHECK-NEXT:          %[[var0:.*]] = llvm.extractvalue %[[data]][2] : !llvm.struct<(ptr<i8>, f64, f64)>
// CHECK-NEXT:          %[[call0:.*]] = call @getVariableValue0(%[[var0]], %[[indices]]) : (f64, !llvm.ptr<i64>) -> f64
// CHECK-NEXT:          cf.br ^[[out]](%[[call0]] : f64)
// CHECK:           ^[[out]](%[[result:.*]]: f64):
// CHECK-NEXT:          return %[[result]] : f64

#var0 = #simulation.variable<name = "x"> : f64

simulation.module attributes {variables = [#var0]} {
    simulation.variable_getter [#var0](%arg0: f64) -> f64 {
        simulation.yield %arg0 : f64
    }
}

// -----

// Cast to float

// CHECK:       func.func @getVariableValue0(%[[variable:.*]]: i64, %[[indices:.*]]: !llvm.ptr<i64>) -> f64 {
// CHECK-NEXT:      %[[result:.*]] = arith.sitofp %[[variable]] : i64 to f64
// CHECK-NEXT:      return %[[result]] : f64
// CHECK-NEXT:  }

#var0 = #simulation.variable<name = "x"> : i64

simulation.module attributes {variables = [#var0]} {
    simulation.variable_getter [#var0](%arg0: i64) -> i64 {
        simulation.yield %arg0 : i64
    }
}

// -----

// Getter with multiple variables

// CHECK:       func.func @getVariableValue0(%[[variable:.*]]: f64, %[[indices:.*]]: !llvm.ptr<i64>) -> f64 {
// CHECK-NEXT:      return %[[variable]] : f64
// CHECK-NEXT:  }

// CHECK:       func.func @getVariableValue(%[[opaquePtr:.*]]: !llvm.ptr<i8>, %[[var:.*]]: i64, %[[indices:.*]]: !llvm.ptr<i64>) -> f64 {
// CHECK-NEXT:      %[[ptr:.*]] = llvm.bitcast %arg0 : !llvm.ptr<i8> to !llvm.ptr<struct<(ptr<i8>, f64, f64, f64)>>
// CHECK-NEXT:      %[[data:.*]] = llvm.load %[[ptr]] : !llvm.ptr<struct<(ptr<i8>, f64, f64, f64)>>
// CHECK-NEXT:      %[[default:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:           cf.switch %arg1 : i64, [
// CHECK-DAG:           default: ^[[out:.*]](%[[default]] : f64),
// CHECK-DAG:           0: ^[[var0Block:.*]],
// CHECK-DAG:           1: ^[[var1Block:.*]]
// CHECK-NEXT:      ]
// CHECK:           ^[[var0Block]]:
// CHECK-NEXT:          %[[var0:.*]] = llvm.extractvalue %[[data]][2] : !llvm.struct<(ptr<i8>, f64, f64, f64)>
// CHECK-NEXT:          %[[call0:.*]] = call @getVariableValue0(%[[var0]], %[[indices]]) : (f64, !llvm.ptr<i64>) -> f64
// CHECK-NEXT:          cf.br ^[[out]](%[[call0]] : f64)
// CHECK:           ^[[var1Block]]:
// CHECK-NEXT:          %[[var1:.*]] = llvm.extractvalue %[[data]][3] : !llvm.struct<(ptr<i8>, f64, f64, f64)>
// CHECK-NEXT:          %[[call1:.*]] = call @getVariableValue0(%[[var1]], %[[indices]]) : (f64, !llvm.ptr<i64>) -> f64
// CHECK-NEXT:          cf.br ^[[out]](%[[call1]] : f64)
// CHECK:           ^[[out]](%[[result:.*]]: f64):
// CHECK-NEXT:          return %[[result]] : f64

#var0 = #simulation.variable<name = "x"> : f64
#var1 = #simulation.variable<name = "y"> : f64

simulation.module attributes {variables = [#var0, #var1]} {
    simulation.variable_getter [#var0, #var1](%arg0: f64) -> f64 {
        simulation.yield %arg0 : f64
    }
}
