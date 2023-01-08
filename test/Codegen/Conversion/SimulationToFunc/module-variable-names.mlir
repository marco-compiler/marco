// RUN: modelica-opt %s --split-input-file --convert-simulation-to-func | FileCheck %s --check-prefix="CHECK-VAR-UNKNOWN"
// RUN: modelica-opt %s --split-input-file --convert-simulation-to-func | FileCheck %s --check-prefix="CHECK-VAR-0"
// RUN: modelica-opt %s --split-input-file --convert-simulation-to-func | FileCheck %s --check-prefix="CHECK-VAR-1"

// CHECK-VAR-UNKNOWN: llvm.mlir.global internal constant @[[var:.*]]("\00")

// CHECK-VAR-UNKNOWN:       func.func @getVariableName(%[[varNumber:.*]]: i64) -> !llvm.ptr<i8>
// CHECK-VAR-UNKNOWN-DAG:   %[[addr:.*]] = llvm.mlir.addressof @[[var]]
// CHECK-VAR-UNKNOWN-DAG:   %[[zero:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-VAR-UNKNOWN:       %[[opaqueAddr:.*]] = llvm.getelementptr %[[addr]][%[[zero]], %[[zero]]]
// CHECK-VAR-UNKNOWN:       cf.switch %[[varNumber]] : i64, [
// CHECK-VAR-UNKNOWN:           default: ^[[outBlock:.*]](%[[opaqueAddr]] : !llvm.ptr<i8>)
// CHECK-VAR-UNKNOWN:       ^[[outBlock]](%[[result:.*]]: !llvm.ptr<i8>):
// CHECK-VAR-UNKNOWN:           return %[[result]]

// CHECK-VAR-0: llvm.mlir.global internal constant @[[var:.*]]("x\00")

// CHECK-VAR-0:     func.func @getVariableName(%[[varNumber:.*]]: i64) -> !llvm.ptr<i8>
// CHECK-VAR-0:         cf.switch %[[varNumber]] : i64, [
// CHECK-VAR-0:             0: ^[[var_block:.*]],
// CHECK-VAR-0:         ]
// CHECK-VAR-0:         ^[[var_block]]:
// CHECK-VAR-0-DAG:         %[[addr:.*]] = llvm.mlir.addressof @[[var]]
// CHECK-VAR-0-DAG:         %[[zero:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-VAR-0:             %[[opaqueAddr:.*]] = llvm.getelementptr %[[addr]][%[[zero]], %[[zero]]]
// CHECK-VAR-0:             cf.br ^[[outBlock:.*]](%[[opaqueAddr]] : !llvm.ptr<i8>)
// CHECK-VAR-0:         ^[[outBlock]](%[[result:.*]]: !llvm.ptr<i8>):
// CHECK-VAR-0:             return %[[result]]

// CHECK-VAR-1: llvm.mlir.global internal constant @[[var:.*]]("y\00")

// CHECK-VAR-1:     func.func @getVariableName(%[[varNumber:.*]]: i64) -> !llvm.ptr<i8>
// CHECK-VAR-1:         cf.switch %[[varNumber]] : i64, [
// CHECK-VAR-1:             1: ^[[var_block:.*]]
// CHECK-VAR-1:         ]
// CHECK-VAR-1:         ^[[var_block]]:
// CHECK-VAR-1-DAG:         %[[addr:.*]] = llvm.mlir.addressof @[[var]]
// CHECK-VAR-1-DAG:         %[[zero:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-VAR-1:             %[[opaqueAddr:.*]] = llvm.getelementptr %[[addr]][%[[zero]], %[[zero]]]
// CHECK-VAR-1:             cf.br ^[[outBlock:.*]](%[[opaqueAddr]] : !llvm.ptr<i8>)
// CHECK-VAR-1:         ^[[outBlock]](%[[result:.*]]: !llvm.ptr<i8>):
// CHECK-VAR-1:             return %[[result]]

#var0 = #simulation.variable<name = "x">
#var1 = #simulation.variable<name = "y">

simulation.module attributes {variables = [#var0, #var1]} {

}
