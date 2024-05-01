// RUN: modelica-opt %s --split-input-file --convert-modelica-to-arith --cse | FileCheck %s

// Matrix base and scalar exponent

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.array<?x?x!bmodelica.real>, %[[arg1:.*]]: !bmodelica.int) -> !bmodelica.array<?x?x!bmodelica.real>
// CHECK-DAG:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[c1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[arg0_dim0:.*]] = bmodelica.dim %[[arg0]], %[[c0]]
// CHECK-DAG:   %[[arg0_dim1:.*]] = bmodelica.dim %[[arg0]], %[[c1]]
// CHECK-DAG:   %[[dims_cmp:.*]] = arith.cmpi eq, %[[arg0_dim0]], %[[arg0_dim1]]
// CHECK-DAG:   cf.assert %[[dims_cmp]]
// CHECK-DAG:   %[[result:.*]] = bmodelica.identity %[[arg0_dim1]] : index -> !bmodelica.array<?x?x!bmodelica.real>
// CHECK-DAG:   %[[upper_bound:.*]] = bmodelica.cast %[[arg1]] : !bmodelica.int -> index
// CHECK-DAG:   %[[upper_bound_1based:.*]] = arith.addi %[[upper_bound]], %[[c1]] : index
// CHECK:       scf.for %[[index_0:.*]] = %[[c1]] to %[[upper_bound_1based]] step %[[c1]] {
// CHECK-DAG:       %[[result_dim0:.*]] = bmodelica.dim %[[result]], %[[c0]]
// CHECK-DAG:       %[[result_dim1:.*]] = bmodelica.dim %[[result]], %[[c1]]
// CHECK-DAG:       %[[temp_result:.*]] = bmodelica.alloc %[[result_dim0]], %[[arg0_dim1]] : <?x?x!bmodelica.real>
// CHECK:           scf.for %[[index_0:.*]] = %[[c0]] to %[[result_dim0]] step %[[c1]] {
// CHECK:               scf.for %[[index_1:.*]] = %[[c0]] to %[[arg0_dim1]] step %[[c1]] {
// CHECK:                   %[[zero:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:                   %[[cross_product:.*]] = scf.for %[[index_2:.*]] = %[[c0]] to %[[result_dim1]] step %[[c1]] iter_args(%[[acc:.*]] = %[[zero]]) -> (f64) {
// CHECK-DAG:                   %[[lhs:.*]] = bmodelica.load %[[result]][%[[index_0]], %[[index_2]]]
// CHECK-DAG:                   %[[rhs:.*]] = bmodelica.load %[[arg0]][%[[index_2]], %[[index_1]]]
// CHECK-DAG:                   %[[lhs_casted:.*]] = builtin.unrealized_conversion_cast %[[lhs]] : !bmodelica.real to f64
// CHECK-DAG:                   %[[rhs_casted:.*]] = builtin.unrealized_conversion_cast %[[rhs]] : !bmodelica.real to f64
// CHECK:                       %[[mul:.*]] = arith.mulf %[[lhs_casted]], %[[rhs_casted]] : f64
// CHECK:                       %[[add:.*]] = arith.addf %[[mul]], %[[acc]] : f64
// CHECK:                       %[[add_casted_1:.*]] = builtin.unrealized_conversion_cast %[[add]] : f64 to !bmodelica.real
// CHECK:                       %[[add_casted_2:.*]] = builtin.unrealized_conversion_cast %[[add_casted_1]] : !bmodelica.real to f64
// CHECK:                       scf.yield %[[add_casted_2]] : f64
// CHECK:                   }
// CHECK:                   %[[cross_product_casted:.*]] = builtin.unrealized_conversion_cast %[[cross_product]] : f64 to !bmodelica.real
// CHECK:                   bmodelica.store %[[temp_result]][%[[index_0]], %[[index_1]]], %[[cross_product_casted]]
// CHECK:               }
// CHECK:           }
// CHECK:           bmodelica.assignment %[[result]], %[[temp_result]]
// CHECK:       }
// CHECK:       return %[[result]]

func.func @foo(%arg0 : !bmodelica.array<?x?x!bmodelica.real>, %arg1 : !bmodelica.int) -> !bmodelica.array<?x?x!bmodelica.real> {
    %0 = bmodelica.pow %arg0, %arg1 : (!bmodelica.array<?x?x!bmodelica.real>, !bmodelica.int) -> !bmodelica.array<?x?x!bmodelica.real>
    func.return %0 : !bmodelica.array<?x?x!bmodelica.real>
}
