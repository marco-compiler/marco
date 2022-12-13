// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(convert-model{model=Test})" | FileCheck %s --check-prefix="CHECK-VAR-UNKNOWN"
// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(convert-model{model=Test})" | FileCheck %s --check-prefix="CHECK-VAR-0"
// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(convert-model{model=Test})" | FileCheck %s --check-prefix="CHECK-VAR-1"

// CHECK-VAR-UNKNOWN: llvm.mlir.global internal constant @[[var:.*]]("unknown\00")

// CHECK-VAR-UNKNOWN-LABEL: @getVariableName
// CHECK-VAR-UNKNOWN-SAME:  (%[[varNumber:.*]]: i64) -> !llvm.ptr<i8>
// CHECK-VAR-UNKNOWN-DAG:   %[[addr:.*]] = llvm.mlir.addressof @[[var]]
// CHECK-VAR-UNKNOWN-DAG:   %[[zero:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-VAR-UNKNOWN:       %[[opaqueAddr:.*]] = llvm.getelementptr %[[addr]][%[[zero]], %[[zero]]]
// CHECK-VAR-UNKNOWN:       cf.switch %[[varNumber]] : i64, [
// CHECK-VAR-UNKNOWN:           default: ^[[outBlock:.*]](%[[opaqueAddr]] : !llvm.ptr<i8>)
// CHECK-VAR-UNKNOWN:       ^[[outBlock]](%[[result:.*]]: !llvm.ptr<i8>):
// CHECK-VAR-UNKNOWN:           return %[[result]]

// CHECK-VAR-0: llvm.mlir.global internal constant @[[var:.*]]("x\00")

// CHECK-VAR-0-LABEL:   @getVariableName
// CHECK-VAR-0-SAME:    (%[[varNumber:.*]]: i64) -> !llvm.ptr<i8>
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

// CHECK-VAR-1-LABEL:   @getVariableName
// CHECK-VAR-1-SAME:    (%[[varNumber:.*]]: i64) -> !llvm.ptr<i8>
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

modelica.model @Test {
    %0 = modelica.member_create @x : !modelica.member<!modelica.real>
    %1 = modelica.member_create @y : !modelica.member<3x!modelica.real>
    modelica.yield %0, %1 : !modelica.member<!modelica.real>, !modelica.member<3x!modelica.real>
} body {
^bb0(%arg0: !modelica.array<!modelica.real>, %arg1: !modelica.array<3x!modelica.real>):

}
