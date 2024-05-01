// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @foo
// CHECK-DAG: %[[i_lb:.*]] = bmodelica.constant #bmodelica.int<1>
// CHECK-DAG: %[[i_ub:.*]] = bmodelica.constant #bmodelica.int<9>
// CHECK-DAG: %[[i_step:.*]] = bmodelica.constant #bmodelica.int<3>
// CHECK-DAG: %[[i_space:.*]] = bmodelica.range %[[i_lb]], %[[i_ub]], %[[i_step]]
// CHECK-DAG: %[[j_lb:.*]] = bmodelica.constant #bmodelica.int<4>
// CHECK-DAG: %[[j_ub:.*]] = bmodelica.constant #bmodelica.int<6>
// CHECK-DAG: %[[j_step:.*]] = bmodelica.constant #bmodelica.int<2>
// CHECK-DAG: %[[j_space:.*]] = bmodelica.range %[[j_lb]], %[[j_ub]], %[[j_step]]
// CHECK:       bmodelica.reduction "mul", iterables = [%[[i_space]], %[[j_space]]], inductions = [%[[i:.*]]: !bmodelica.int, %[[j:.*]]: !bmodelica.int] {
// CHECK-DAG:       %[[i_offset:.*]] = bmodelica.constant -1
// CHECK-DAG:       %[[index_0:.*]] = bmodelica.add %[[i]], %[[i_offset]]
// CHECK-DAG:       %[[j_offset:.*]] = bmodelica.constant -1
// CHECK-DAG:       %[[index_1:.*]] = bmodelica.add %[[j]], %[[j_offset]]
// CHECK-DAG:       %[[x:.*]] = bmodelica.variable_get @x
// CHECK:           %[[subscription:.*]] = bmodelica.subscription %[[x]][%[[index_0]], %[[index_1]]]
// CHECK:           %[[load:.*]] = bmodelica.load %[[subscription]][]
// CHECK:           bmodelica.yield %[[load]]
// CHECK:       } : (!bmodelica<range !bmodelica.int>, !bmodelica<range !bmodelica.int>) -> !bmodelica.real

function foo
    input Real[9,6] x;
    output Real y;
algorithm
    y := product(x[i,j] for i in 1:3:9, j in 4:2:6);
end foo;
