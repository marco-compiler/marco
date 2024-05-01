// RUN: modelica-opt %s --split-input-file --convert-runtime-to-func | FileCheck %s

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


runtime.variable_getter @getter0() -> f64 {
    %0 = bmodelica.constant #bmodelica.real<0.0>
    %1 = bmodelica.cast %0 : !bmodelica.real -> f64
    runtime.return %1 : f64
}

runtime.variable_getter @getter1(%arg0: index, %arg1: index, %arg2: index) -> f64 {
    %0 = bmodelica.alloc : <2x3x4x!bmodelica.real>
    %1 = bmodelica.load %0[%arg0, %arg1, %arg2] : !bmodelica.array<2x3x4x!bmodelica.real>
    %2 = bmodelica.cast %1 : !bmodelica.real -> f64
    runtime.return %2 : f64
}

runtime.variable_getters [@getter0, @getter1]
