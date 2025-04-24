// RUN: modelica-opt %s --split-input-file --generate-runtime-verification | FileCheck %s

// CHECK-LABEL: @Test
// CHECK-SAME: (%{{.*}}: i64, %[[rhs:.*]]: i64)

func.func @Test(%arg0: i64, %arg1: i64) -> f64 {
    // CHECK: bmodelica.assert
    // CHECK: %[[zero:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00>
    // CHECK: %[[condition:.*]] = bmodelica.neq %[[rhs]], %[[zero]]
    // CHECK: bmodelica.yield %[[condition]]

    %0 = bmodelica.div_trunc %arg0, %arg1 : (i64, i64) -> f64
    func.return %0 : f64
}
