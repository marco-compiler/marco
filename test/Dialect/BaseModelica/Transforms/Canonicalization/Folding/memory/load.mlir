// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @LoadFromArraySubscript
// CHECK-SAME: (%[[array:.*]]: !bmodelica.array<10x20xf64>, %[[i:.*]]: index, %[[j:.*]]: index)
// CHECK: %[[value:.*]] = bmodelica.array.load %[[array]][%[[i]], %[[j]]]
// CHECK: return %[[value]]

func.func @LoadFromArraySubscript(%array: !bmodelica.array<10x20xf64>, %i: index, %j: index) -> f64 {
    %sub = bmodelica.array.subscription %array[%i] : !bmodelica.array<10x20xf64>, index -> !bmodelica.array<20xf64>
    %value = bmodelica.array.load %sub[%j] : !bmodelica.array<20xf64>
    return %value : f64
}

// -----

// CHECK-LABEL: @LoadFromArraySlice
// CHECK-SAME: (%[[array:.*]]: !bmodelica.array<10x20xf64>, %[[range:.*]]: !bmodelica<range index>, %[[i:.*]]: index, %[[j:.*]]: index)
// CHECK: %[[sub:.*]] = bmodelica.array.subscription %[[array]][%[[range]]]
// CHECK: %[[value:.*]] = bmodelica.array.load %[[sub]][%[[i]], %[[j]]]
// CHECK: return %[[value]]

func.func @LoadFromArraySlice(%dst: !bmodelica.array<10x20xf64>, %range: !bmodelica<range index>, %i: index, %j: index) -> f64 {
    %sub = bmodelica.array.subscription %dst[%range] : !bmodelica.array<10x20xf64>, !bmodelica<range index> -> !bmodelica.array<?x20xf64>
    %value = bmodelica.array.load %sub[%i, %j] : !bmodelica.array<?x20xf64>
    return %value : f64
}
