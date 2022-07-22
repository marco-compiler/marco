// RUN: modelica-opt %s --split-input-file --convert-modelica-to-arith | FileCheck %s

// CHECK-LABEL: @singleResult
// CHECK-DAG:   %[[condition:.*]] = arith.constant true
// CHECK-DAG:   %[[trueValue:.*]] = arith.constant 0
// CHECK-DAG:   %[[falseValue:.*]] = arith.constant 1
// CHECK:       %[[result:.*]] = arith.select %[[condition]], %[[trueValue]], %[[falseValue]]
// CHECK:       return %[[result]]

func.func @singleResult() -> i64 {
    %condition = arith.constant true
    %trueValue = arith.constant 0 : i64
    %falseValue = arith.constant 1 : i64
    %select = modelica.select (%condition : i1), (%trueValue : i64), (%falseValue : i64) -> i64
    func.return %select : i64
}

// -----

// CHECK-LABEL: @multipleResults
// CHECK-DAG:   %[[condition:.*]] = arith.constant true
// CHECK-DAG:   %[[trueValue0:.*]] = arith.constant 0
// CHECK-DAG:   %[[trueValue1:.*]] = arith.constant 1
// CHECK-DAG:   %[[falseValue0:.*]] = arith.constant 2
// CHECK-DAG:   %[[falseValue1:.*]] = arith.constant 3
// CHECK:       %[[select0:.*]] = arith.select %[[condition]], %[[trueValue0]], %[[falseValue0]]
// CHECK:       %[[select1:.*]] = arith.select %[[condition]], %[[trueValue1]], %[[falseValue1]]
// CHECK:       %[[result:.*]] = arith.addi %[[select0]], %[[select1]]
// CHECK:       return %[[result]]

func.func @multipleResults() -> i64 {
    %condition = arith.constant true
    %trueValue0 = arith.constant 0 : i64
    %trueValue1 = arith.constant 1 : i64
    %falseValue0 = arith.constant 2 : i64
    %falseValue1 = arith.constant 3 : i64
    %select:2 = modelica.select (%condition : i1), (%trueValue0, %trueValue1 : i64, i64), (%falseValue0, %falseValue1 : i64, i64) -> (i64, i64)
    %result = arith.addi %select#0, %select#1 : i64
    func.return %result : i64
}
