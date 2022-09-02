// RUN: modelica-opt %s --split-input-file --convert-modelica-to-arith | FileCheck %s

// Single result

// CHECK-LABEL: @foo
// CHECK-DAG:   %[[condition:.*]] = arith.constant true
// CHECK-DAG:   %[[trueValue:.*]] = arith.constant 0
// CHECK-DAG:   %[[falseValue:.*]] = arith.constant 1
// CHECK:       %[[result:.*]] = arith.select %[[condition]], %[[trueValue]], %[[falseValue]]
// CHECK:       %[[result_cast:.*]] = builtin.unrealized_conversion_cast %[[result]]
// CHECK:       return %[[result_cast]]

func.func @foo() -> !modelica.int {
    %condition = modelica.constant #modelica.bool<true>
    %trueValue = modelica.constant #modelica.int<0>
    %falseValue = modelica.constant #modelica.int<1>
    %select = modelica.select (%condition : !modelica.bool), (%trueValue : !modelica.int), (%falseValue : !modelica.int) -> !modelica.int
    func.return %select : !modelica.int
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

func.func @foo() -> !modelica.int {
    %condition = modelica.constant #modelica.bool<true>
    %trueValue0 = modelica.constant #modelica.int<0>
    %trueValue1 = modelica.constant #modelica.int<1>
    %falseValue0 = modelica.constant #modelica.int<2>
    %falseValue1 = modelica.constant #modelica.int<3>
    %select:2 = modelica.select (%condition : !modelica.bool), (%trueValue0, %trueValue1 : !modelica.int, !modelica.int), (%falseValue0, %falseValue1 : !modelica.int, !modelica.int) -> (!modelica.int, !modelica.int)
    %result = modelica.add %select#0, %select#1 : (!modelica.int, !modelica.int) -> !modelica.int
    func.return %result : !modelica.int
}
