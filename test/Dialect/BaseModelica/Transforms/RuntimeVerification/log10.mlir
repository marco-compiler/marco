// RUN: modelica-opt %s --split-input-file --generate-runtime-verification | FileCheck %s

// CHECK-LABEL: @Test
// CHECK-SAME: (%[[arg0:.*]]: f64)

func.func @Test(%arg0: f64) -> f64 {
    // CHECK: bmodelica.assert
    // CHECK: %[[zero:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00>
    // CHECK: %[[condition:.*]] = bmodelica.gt %[[arg0]], %[[zero]]
    // CHECK: bmodelica.yield %[[condition]]

    %0 = bmodelica.log10 %arg0 : f64 -> f64
    func.return %0 : f64
}
