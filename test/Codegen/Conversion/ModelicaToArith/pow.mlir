// RUN: modelica-opt %s --split-input-file --convert-modelica-to-arith --cse | FileCheck %s

// Matrix base and scalar exponent

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !modelica.array<?x?x!modelica.real>, %[[arg1:.*]]: !modelica.int) -> !modelica.array<?x?x!modelica.real>
// CHECK-DAG:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[c1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[arg0_dim0:.*]] = modelica.dim %[[arg0]], %[[c0]]
// CHECK-DAG:   %[[arg0_dim1:.*]] = modelica.dim %[[arg0]], %[[c1]]
// CHECK-DAG:   %[[dims_cmp:.*]] = arith.cmpi eq, %[[arg0_dim0]], %[[arg0_dim1]]
// CHECK-DAG:   cf.assert %[[dims_cmp]]
// CHECK-DAG:   %[[result:.*]] = modelica.identity %[[arg0_dim1]] : index -> !modelica.array<?x?x!modelica.real>
// CHECK-DAG:   %[[upper_bound:.*]] = modelica.cast %[[arg1]] : !modelica.int -> index
// CHECK-DAG:   %[[upper_bound_1based:.*]] = arith.addi %[[upper_bound]], %[[c1]] : index
// CHECK:       scf.for %[[index_0:.*]] = %[[c1]] to %[[upper_bound_1based]] step %[[c1]] {
// CHECK-DAG:       %[[result_dim0:.*]] = modelica.dim %[[result]], %[[c0]]
// CHECK-DAG:       %[[result_dim1:.*]] = modelica.dim %[[result]], %[[c1]]
// CHECK-DAG:       %[[temp_result:.*]] = modelica.alloc %[[result_dim0]], %[[arg0_dim1]] : <?x?x!modelica.real>
// CHECK:           scf.for %[[index_0:.*]] = %[[c0]] to %[[result_dim0]] step %[[c1]] {
// CHECK:               scf.for %[[index_1:.*]] = %[[c0]] to %[[arg0_dim1]] step %[[c1]] {
// CHECK:                   %[[zero:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:                   %[[cross_product:.*]] = scf.for %[[index_2:.*]] = %[[c0]] to %[[result_dim1]] step %[[c1]] iter_args(%[[acc:.*]] = %[[zero]]) -> (f64) {
// CHECK-DAG:                   %[[lhs:.*]] = modelica.load %[[result]][%[[index_0]], %[[index_2]]]
// CHECK-DAG:                   %[[rhs:.*]] = modelica.load %[[arg0]][%[[index_2]], %[[index_1]]]
// CHECK-DAG:                   %[[lhs_casted:.*]] = builtin.unrealized_conversion_cast %[[lhs]] : !modelica.real to f64
// CHECK-DAG:                   %[[rhs_casted:.*]] = builtin.unrealized_conversion_cast %[[rhs]] : !modelica.real to f64
// CHECK:                       %[[mul:.*]] = arith.mulf %[[lhs_casted]], %[[rhs_casted]] : f64
// CHECK:                       %[[add:.*]] = arith.addf %[[mul]], %[[acc]] : f64
// CHECK:                       %[[add_casted_1:.*]] = builtin.unrealized_conversion_cast %[[add]] : f64 to !modelica.real
// CHECK:                       %[[add_casted_2:.*]] = builtin.unrealized_conversion_cast %[[add_casted_1]] : !modelica.real to f64
// CHECK:                       scf.yield %[[add_casted_2]] : f64
// CHECK:                   }
// CHECK:                   %[[cross_product_casted:.*]] = builtin.unrealized_conversion_cast %[[cross_product]] : f64 to !modelica.real
// CHECK:                   modelica.store %[[temp_result]][%[[index_0]], %[[index_1]]], %[[cross_product_casted]]
// CHECK:               }
// CHECK:           }
// CHECK:           modelica.assignment %[[result]], %[[temp_result]]
// CHECK:       }
// CHECK:       return %[[result]]

func.func @foo(%arg0 : !modelica.array<?x?x!modelica.real>, %arg1 : !modelica.int) -> !modelica.array<?x?x!modelica.real> {
    %0 = modelica.pow %arg0, %arg1 : (!modelica.array<?x?x!modelica.real>, !modelica.int) -> !modelica.array<?x?x!modelica.real>
    func.return %0 : !modelica.array<?x?x!modelica.real>
}
