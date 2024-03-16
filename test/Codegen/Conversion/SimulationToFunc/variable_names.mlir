// RUN: modelica-opt %s --split-input-file --convert-simulation-to-func | FileCheck %s --check-prefix="CHECK-VAR-UNKNOWN"
// RUN: modelica-opt %s --split-input-file --convert-simulation-to-func | FileCheck %s --check-prefix="CHECK-VAR-0"
// RUN: modelica-opt %s --split-input-file --convert-simulation-to-func | FileCheck %s --check-prefix="CHECK-VAR-1"

// CHECK-VAR-UNKNOWN: llvm.mlir.global internal constant @[[var:.*]]("\00")

// CHECK-VAR-UNKNOWN:   func.func @getVariableName(%[[varNumber:.*]]: i64) -> !llvm.ptr
// CHECK-VAR-UNKNOWN:       %[[addr:.*]] = llvm.mlir.addressof @[[var]]
// CHECK-VAR-UNKNOWN:       %[[elementPtr:.*]] = llvm.getelementptr %[[addr]][0, 0]
// CHECK-VAR-UNKNOWN:       cf.switch %[[varNumber]] : i64, [
// CHECK-VAR-UNKNOWN:           default: ^[[outBlock:.*]](%[[elementPtr]] : !llvm.ptr)
// CHECK-VAR-UNKNOWN:       ^[[outBlock]](%[[result:.*]]: !llvm.ptr):
// CHECK-VAR-UNKNOWN:           return %[[result]]

// CHECK-VAR-0: llvm.mlir.global internal constant @[[var:.*]]("x\00")

// CHECK-VAR-0:     func.func @getVariableName(%[[varNumber:.*]]: i64) -> !llvm.ptr
// CHECK-VAR-0:         cf.switch %[[varNumber]] : i64, [
// CHECK-VAR-0:             0: ^[[var_block:.*]],
// CHECK-VAR-0:         ]
// CHECK-VAR-0:         ^[[var_block]]:
// CHECK-VAR-0:             %[[addr:.*]] = llvm.mlir.addressof @[[var]]
// CHECK-VAR-0:             %[[elementPtr:.*]] = llvm.getelementptr %[[addr]][0, 0]
// CHECK-VAR-0:             cf.br ^[[outBlock:.*]](%[[elementPtr]] : !llvm.ptr)
// CHECK-VAR-0:         ^[[outBlock]](%[[result:.*]]: !llvm.ptr):
// CHECK-VAR-0:             return %[[result]]

// CHECK-VAR-1: llvm.mlir.global internal constant @[[var:.*]]("y\00")

// CHECK-VAR-1:     func.func @getVariableName(%[[varNumber:.*]]: i64) -> !llvm.ptr
// CHECK-VAR-1:         cf.switch %[[varNumber]] : i64, [
// CHECK-VAR-1:             1: ^[[var_block:.*]]
// CHECK-VAR-1:         ]
// CHECK-VAR-1:         ^[[var_block]]:
// CHECK-VAR-1:             %[[addr:.*]] = llvm.mlir.addressof @[[var]]
// CHECK-VAR-1:             %[[elementPtr:.*]] = llvm.getelementptr %[[addr]][0, 0]
// CHECK-VAR-1:             cf.br ^[[outBlock:.*]](%[[elementPtr]] : !llvm.ptr)
// CHECK-VAR-1:         ^[[outBlock]](%[[result:.*]]: !llvm.ptr):
// CHECK-VAR-1:             return %[[result]]

simulation.variable_names ["x", "y"]
