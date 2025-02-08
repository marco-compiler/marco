// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-tensor | FileCheck %s

// CHECK-LABEL: @noSubscripts
// CHECK-SAME:  (%[[source:.*]]: tensor<10x20xi64>, %[[destination:.*]]: tensor<10x20xi64>) -> tensor<10x20xi64>
// CHECK: %[[result:.*]] = tensor.insert_slice %[[source]] into %[[destination]][0, 0] [10, 20] [1, 1]
// CHECK: return %[[result]]

func.func @noSubscripts(%source: tensor<10x20xi64>, %destination: tensor<10x20xi64>) -> tensor<10x20xi64> {
    %0 = bmodelica.tensor_insert_slice %source, %destination[] : tensor<10x20xi64>, tensor<10x20xi64> -> tensor<10x20xi64>
    func.return %0 : tensor<10x20xi64>
}

// -----

// CHECK-LABEL: @reducedRankSource
// CHECK-SAME:  (%[[source:.*]]: tensor<20xi64>, %[[destination:.*]]: tensor<10x20xi64>, %[[s0:.*]]: index) -> tensor<10x20xi64>
// CHECK: %[[result:.*]] = tensor.insert_slice %[[source]] into %[[destination]][%[[s0]], 0] [1, 20] [1, 1]
// CHECK: return %[[result]]

func.func @reducedRankSource(%source: tensor<20xi64>, %destination: tensor<10x20xi64>, %s0: index) -> tensor<10x20xi64> {
    %0 = bmodelica.tensor_insert_slice %source, %destination[%s0] : tensor<20xi64>, tensor<10x20xi64>, index -> tensor<10x20xi64>
    func.return %0 : tensor<10x20xi64>
}

// -----

// CHECK-LABEL: @rangeSameRank
// CHECK-SAME:  (%[[source:.*]]: tensor<5xi64>, %[[destination:.*]]: tensor<10xi64>, %[[s0:.*]]: !bmodelica<range index>) -> tensor<10xi64>
// CHECK-DAG:   %[[begin:.*]] = bmodelica.range_begin %[[s0]]
// CHECK-DAG:   %[[end:.*]] = bmodelica.range_end %[[s0]]
// CHECK-DAG:   %[[step:.*]] = bmodelica.range_step %[[s0]]
// CHECK-DAG:   %[[zero:.*]] = arith.constant 0 : index
// CHECK:       %[[cmp:.*]] = arith.cmpi sge, %[[step]], %[[zero]]
// CHECK:       %[[offset:.*]] = arith.select %[[cmp]], %[[begin]], %[[end]]
// CHECK:       %[[result:.*]] = tensor.insert_slice %[[source]] into %[[destination]][%[[offset]]] [5] [%[[step]]]
// CHECK:       return %[[result]]

func.func @rangeSameRank(%source: tensor<5xi64>, %destination: tensor<10xi64>, %s0: !bmodelica<range index>) -> tensor<10xi64> {
    %0 = bmodelica.tensor_insert_slice %source, %destination[%s0] : tensor<5xi64>, tensor<10xi64>, !bmodelica<range index> -> tensor<10xi64>
    func.return %0 : tensor<10xi64>
}

// -----

// CHECK-LABEL: @rangeScalar
// CHECK-SAME:  (%[[source:.*]]: tensor<5xi64>, %[[destination:.*]]: tensor<10x10xi64>, %[[s0:.*]]: !bmodelica<range index>, %[[s1:.*]]: index) -> tensor<10x10xi64>
// CHECK-DAG:   %[[begin:.*]] = bmodelica.range_begin %[[s0]]
// CHECK-DAG:   %[[end:.*]] = bmodelica.range_end %[[s0]]
// CHECK-DAG:   %[[step:.*]] = bmodelica.range_step %[[s0]]
// CHECK-DAG:   %[[zero:.*]] = arith.constant 0 : index
// CHECK:       %[[cmp:.*]] = arith.cmpi sge, %[[step]], %[[zero]]
// CHECK:       %[[offset:.*]] = arith.select %[[cmp]], %[[begin]], %[[end]]
// CHECK:       %[[result:.*]] = tensor.insert_slice %[[source]] into %[[destination]][%[[offset]], %[[s1]]] [5, 1] [%[[step]], 1]
// CHECK:       return %[[result]]

func.func @rangeScalar(%source: tensor<5xi64>, %destination: tensor<10x10xi64>, %s0: !bmodelica<range index>, %s1: index) -> tensor<10x10xi64> {
    %0 = bmodelica.tensor_insert_slice %source, %destination[%s0, %s1] : tensor<5xi64>, tensor<10x10xi64>, !bmodelica<range index>, index -> tensor<10x10xi64>
    func.return %0 : tensor<10x10xi64>
}

// -----

// CHECK-LABEL: @scalarRange
// CHECK-SAME:  (%[[source:.*]]: tensor<5xi64>, %[[destination:.*]]: tensor<10x10xi64>, %[[s0:.*]]: index, %[[s1:.*]]: !bmodelica<range index>) -> tensor<10x10xi64>
// CHECK-DAG:   %[[begin:.*]] = bmodelica.range_begin %[[s1]]
// CHECK-DAG:   %[[end:.*]] = bmodelica.range_end %[[s1]]
// CHECK-DAG:   %[[step:.*]] = bmodelica.range_step %[[s1]]
// CHECK-DAG:   %[[zero:.*]] = arith.constant 0 : index
// CHECK:       %[[cmp:.*]] = arith.cmpi sge, %[[step]], %[[zero]]
// CHECK:       %[[offset:.*]] = arith.select %[[cmp]], %[[begin]], %[[end]]
// CHECK:       %[[result:.*]] = tensor.insert_slice %[[source]] into %[[destination]][%[[s0]], %[[offset]]] [1, 5] [1, %[[step]]]
// CHECK:       return %[[result]]

func.func @scalarRange(%source: tensor<5xi64>, %destination: tensor<10x10xi64>, %s0: index, %s1: !bmodelica<range index>) -> tensor<10x10xi64> {
    %0 = bmodelica.tensor_insert_slice %source, %destination[%s0, %s1] : tensor<5xi64>, tensor<10x10xi64>, index, !bmodelica<range index>-> tensor<10x10xi64>
    func.return %0 : tensor<10x10xi64>
}
