// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @foo
// CHECK-DAG: %[[i_lb:.*]] = modelica.constant #modelica.int<1>
// CHECK-DAG: %[[i_ub:.*]] = modelica.constant #modelica.int<9>
// CHECK-DAG: %[[i_step:.*]] = modelica.constant #modelica.int<3>
// CHECK-DAG: %[[i_space:.*]] = modelica.range %[[i_lb]], %[[i_ub]], %[[i_step]]
// CHECK-DAG: %[[j_lb:.*]] = modelica.constant #modelica.int<4>
// CHECK-DAG: %[[j_ub:.*]] = modelica.constant #modelica.int<6>
// CHECK-DAG: %[[j_step:.*]] = modelica.constant #modelica.int<2>
// CHECK-DAG: %[[j_space:.*]] = modelica.range %[[j_lb]], %[[j_ub]], %[[j_step]]
// CHECK:       modelica.reduction "max", iterables = [%[[i_space]], %[[j_space]]], inductions = [%[[i:.*]]: !modelica.int, %[[j:.*]]: !modelica.int] {
// CHECK-DAG:       %[[i_offset:.*]] = modelica.constant -1
// CHECK-DAG:       %[[index_0:.*]] = modelica.add %[[i]], %[[i_offset]]
// CHECK-DAG:       %[[j_offset:.*]] = modelica.constant -1
// CHECK-DAG:       %[[index_1:.*]] = modelica.add %[[j]], %[[j_offset]]
// CHECK-DAG:       %[[x:.*]] = modelica.variable_get @x
// CHECK:           %[[subscription:.*]] = modelica.subscription %[[x]][%[[index_0]], %[[index_1]]]
// CHECK:           %[[load:.*]] = modelica.load %[[subscription]][]
// CHECK:           modelica.yield %[[load]]
// CHECK:       } : (!modelica<range !modelica.int>, !modelica<range !modelica.int>) -> !modelica.real

function foo
    input Real[9,6] x;
    output Real y;
algorithm
    y := max(x[i,j] for i in 1:3:9, j in 4:2:6);
end foo;
