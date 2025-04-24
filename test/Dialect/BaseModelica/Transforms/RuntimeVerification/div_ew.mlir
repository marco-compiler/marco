// RUN: modelica-opt %s --split-input-file --generate-runtime-verification | FileCheck %s

// CHECK-LABEL: @scalar
// CHECK-SAME: (%{{.*}}: f64, %[[rhs:.*]]: f64)

func.func @scalar(%arg0: f64, %arg1: f64) -> f64 {
    // CHECK: bmodelica.assert
    // CHECK: %[[zero:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00>
    // CHECK: %[[condition:.*]] = bmodelica.neq %[[rhs]], %[[zero]]
    // CHECK: bmodelica.yield %[[condition]]

    %0 = bmodelica.div_ew %arg0, %arg1 : (f64, f64) -> f64
    func.return %0 : f64
}

// -----

// CHECK-LABEL: @tensor
// CHECK-SAME: (%{{.*}}: f64, %[[rhs:.*]]: tensor<?x?xf64>)

func.func @tensor(%arg0: f64, %arg1: tensor<?x?xf64>) -> tensor<?x?xf64> {
    // CHECK-DAG:   %[[lowerBound:.*]] = bmodelica.constant 0 : index
    // CHECK-DAG:   %[[dimIndex0:.*]] = arith.constant 0 : index
    // CHECK-DAG:   %[[upperBound0:.*]] = tensor.dim %[[rhs]], %[[dimIndex0]]
    // CHECK-DAG:   %[[dimIndex1:.*]] = arith.constant 1 : index
    // CHECK-DAG:   %[[upperBound1:.*]] = tensor.dim %[[rhs]], %[[dimIndex1]]
    // CHECK-DAG:   %[[step:.*]] = bmodelica.constant 1 : index
    // CHECK-DAG:   scf.for %[[i0:.*]] = %[[lowerBound]] to %[[upperBound0]] step %[[step]]
    // CHECK-DAG:       scf.for %[[i1:.*]] = %[[lowerBound]] to %[[upperBound1]] step %[[step]]
    // CHECK:               %[[element:.*]] = bmodelica.tensor_extract %[[rhs]][%[[i0]], %[[i1]]]
    // CHECK:               bmodelica.assert
    // CHECK:                   %[[zero:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00>
    // CHECK:                   %[[condition:.*]] = bmodelica.neq %[[element]], %[[zero]]
    // CHECK:                   bmodelica.yield %[[condition]]

    %0 = bmodelica.div_ew %arg0, %arg1 : (f64, tensor<?x?xf64>) -> tensor<?x?xf64>
    func.return %0 : tensor<?x?xf64>
}
