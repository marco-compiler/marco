// RUN: modelica-opt %s --split-input-file --generate-runtime-verification | FileCheck %s

// CHECK-LABEL: @Test
// CHECK-SAME: (%{{.*}}: !bmodelica.array<?x?x?xf64>, %[[arg1:.*]]: index)

func.func @Test(%arg0: !bmodelica.array<?x?x?xf64>, %arg1: index) -> index {
    // CHECK:       bmodelica.assert
    // CHECK-DAG:   %[[zero:.*]] = bmodelica.constant 0 : index
    // CHECK-DAG:   %[[rank:.*]] = bmodelica.constant 3 : index
    // CHECK-DAG:   %[[lbCondition:.*]] = bmodelica.gte %[[arg1]], %[[zero]]
    // CHECK-DAG:   %[[ubCoundition:.*]] = bmodelica.lt %[[arg1]], %[[rank]]
    // CHECK:       %[[cond:.*]] = bmodelica.and %[[lbCondition]], %[[ubCoundition]]
    // CHECK:       bmodelica.yield %[[cond]]

    %0 = bmodelica.dim %arg0, %arg1 : !bmodelica.array<?x?x?xf64>
    func.return %0 : index
}
