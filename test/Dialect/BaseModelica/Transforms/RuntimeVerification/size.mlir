// RUN: modelica-opt %s --split-input-file --generate-runtime-verification | FileCheck %s

// CHECK-LABEL: @Test
// CHECK-SAME: (%{{.*}}: tensor<?x?xf64>, %[[arg1:.*]]: index)

func.func @Test(%arg0: tensor<?x?xf64>, %arg1: index) -> index {
    // CHECK:       bmodelica.assert
    // CHECK-DAG:   %[[lowerBound:.*]] = bmodelica.constant 0 : index
    // CHECK-DAG:   %[[lbCondition:.*]] = bmodelica.gte %[[arg1]], %[[lowerBound]]
    // CHECK-DAG:   %[[upperBound:.*]] = bmodelica.constant 2 : index
    // CHECK-DAG:   %[[ubCondition:.*]] = bmodelica.lt %[[arg1]], %[[upperBound]]
    // CHECK:       %[[condition:.*]] = bmodelica.and %[[lbCondition]], %[[ubCondition]]
    // CHECK:       bmodelica.yield %[[condition]]

    %0 = bmodelica.size %arg0, %arg1 : (tensor<?x?xf64>, index) -> index
    func.return %0 : index
}
