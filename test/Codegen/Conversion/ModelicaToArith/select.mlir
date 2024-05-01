// RUN: modelica-opt %s --split-input-file --convert-modelica-to-arith | FileCheck %s

// Single result

// CHECK-LABEL: @foo
// CHECK-DAG:   %[[condition:.*]] = arith.constant true
// CHECK-DAG:   %[[trueValue:.*]] = arith.constant 0
// CHECK-DAG:   %[[falseValue:.*]] = arith.constant 1
// CHECK:       %[[result:.*]] = arith.select %[[condition]], %[[trueValue]], %[[falseValue]]
// CHECK:       %[[result_cast:.*]] = builtin.unrealized_conversion_cast %[[result]]
// CHECK:       return %[[result_cast]]

func.func @foo() -> !bmodelica.int {
    %condition = bmodelica.constant #bmodelica.bool<true>
    %trueValue = bmodelica.constant #bmodelica.int<0>
    %falseValue = bmodelica.constant #bmodelica.int<1>
    %select = bmodelica.select (%condition : !bmodelica.bool), (%trueValue : !bmodelica.int), (%falseValue : !bmodelica.int) -> !bmodelica.int
    func.return %select : !bmodelica.int
}

// -----

// Multiple results

// CHECK-LABEL: @foo
// CHECK-DAG:   %[[condition:.*]] = arith.constant true
// CHECK-DAG:   %[[trueValue0:.*]] = arith.constant 0
// CHECK-DAG:   %[[trueValue1:.*]] = arith.constant 1
// CHECK-DAG:   %[[falseValue0:.*]] = arith.constant 2
// CHECK-DAG:   %[[falseValue1:.*]] = arith.constant 3
// CHECK:       %[[select0:.*]] = arith.select %[[condition]], %[[trueValue0]], %[[falseValue0]]
// CHECK:       %[[select1:.*]] = arith.select %[[condition]], %[[trueValue1]], %[[falseValue1]]
// CHECK:       %[[result:.*]] = arith.addi %[[select0]], %[[select1]]
// CHECK:       %[[result_cast:.*]] = builtin.unrealized_conversion_cast %[[result]]
// CHECK:       return %[[result_cast]]

func.func @foo() -> !bmodelica.int {
    %condition = bmodelica.constant #bmodelica.bool<true>
    %trueValue0 = bmodelica.constant #bmodelica.int<0>
    %trueValue1 = bmodelica.constant #bmodelica.int<1>
    %falseValue0 = bmodelica.constant #bmodelica.int<2>
    %falseValue1 = bmodelica.constant #bmodelica.int<3>
    %select:2 = bmodelica.select (%condition : !bmodelica.bool), (%trueValue0, %trueValue1 : !bmodelica.int, !bmodelica.int), (%falseValue0, %falseValue1 : !bmodelica.int, !bmodelica.int) -> (!bmodelica.int, !bmodelica.int)
    %result = bmodelica.add %select#0, %select#1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    func.return %result : !bmodelica.int
}
