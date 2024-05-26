// RUN: modelica-opt %s --split-input-file --convert-runtime-model-metadata | FileCheck %s

// CHECK:       func.func @getVariableValue(%[[variable:.*]]: i64, %[[indices:.*]]: !llvm.ptr) -> f64 {
// CHECK-NEXT:      %[[default:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:      cf.switch %[[variable]] : i64, [
// CHECK-DAG:           default: ^[[out:.*]](%cst : f64),
// CHECK-DAG:           0: ^[[var0:.*]],
// CHECK-DAG:           1: ^[[var1:.*]]
// CHECK-NEXT:  ]
// CHECK-NEXT:  ^[[var0]]:
// CHECK-NEXT:      %[[var0_result:.*]] = call @getter0(%[[indices]])
// CHECK-NEXT:      cf.br ^bb3(%[[var0_result]] : f64)
// CHECK-NEXT:  ^[[var1]]:
// CHECK-NEXT:      %[[var1_result:.*]] = call @getter1(%[[indices]])
// CHECK-NEXT:      cf.br ^bb3(%[[var1_result]] : f64)
// CHECK-NEXT:  ^[[out]](%[[result:.*]]: f64):
// CHECK-NEXT:      return %[[result]]
// CHECK-NEXT:  }

func.func @getter0(%arg0: !llvm.ptr) -> f64 {
    %0 = arith.constant 0.0 : f64
    func.return %0 : f64
}

func.func @getter1(%arg0: !llvm.ptr) -> f64 {
    %0 = arith.constant 0.0 : f64
    func.return %0 : f64
}

runtime.variable_getters [@getter0, @getter1]
