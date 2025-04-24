// RUN: modelica-opt %s --split-input-file --generate-runtime-verification | FileCheck %s

// CHECK-LABEL: @Test
// CHECK-SAME: (%[[arg0:.*]]: f64)

func.func @Test(%arg0: f64) -> f64 {
    // CHECK:       bmodelica.assert
    // CHECK-SAME:  level = 1
    // CHECK-DAG:   %[[lowerBound:.*]] = bmodelica.constant #bmodelica<real -1.000000e+00>
    // CHECK-DAG:   %[[upperBound:.*]] = bmodelica.constant #bmodelica<real 1.000000e+00>
    // CHECK-DAG:   %[[lbCondition:.*]] = bmodelica.gte %[[arg0]], %[[lowerBound]]
    // CHECK-DAG:   %[[ubCondition:.*]] = bmodelica.lte %[[arg0]], %[[upperBound]]
    // CHECK:       %[[condition:.*]] = bmodelica.and %[[lbCondition]], %[[ubCondition]]
    // CHECK:       bmodelica.yield %[[condition]]

    %0 = bmodelica.asin %arg0 : f64 -> f64
    func.return %0 : f64
}
