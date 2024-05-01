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
    %0 = modelica.constant #modelica.real<0.0>
    %1 = modelica.cast %0 : !modelica.real -> f64
    runtime.return %1 : f64
}

runtime.variable_getter @getter1(%arg0: index, %arg1: index, %arg2: index) -> f64 {
    %0 = modelica.alloc : <2x3x4x!modelica.real>
    %1 = modelica.load %0[%arg0, %arg1, %arg2] : !modelica.array<2x3x4x!modelica.real>
    %2 = modelica.cast %1 : !modelica.real -> f64
    runtime.return %2 : f64
}

runtime.variable_getters [@getter0, @getter1]
