// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @StoreInArraySubscript
// CHECK-SAME: (%[[array:.*]]: !bmodelica.array<10x20xf64>, %[[value:.*]]: f64, %[[i:.*]]: index, %[[j:.*]]: index)
// CHECK: bmodelica.array.store %[[array]][%[[i]], %[[j]]], %[[value]]

func.func @StoreInArraySubscript(%array: !bmodelica.array<10x20xf64>, %value: f64, %i: index, %j: index) {
    %sub = bmodelica.array.subscription %array[%i] : !bmodelica.array<10x20xf64>, index -> !bmodelica.array<20xf64>
    bmodelica.array.store %sub[%j], %value : !bmodelica.array<20xf64>
    return
}

// -----

// CHECK-LABEL: @StoreInArraySlice
// CHECK-SAME: (%[[array:.*]]: !bmodelica.array<10x20xf64>, %[[value:.*]]: f64, %[[range:.*]]: !bmodelica<range index>, %[[i:.*]]: index, %[[j:.*]]: index)
// CHECK: %[[sub:.*]] = bmodelica.array.subscription %[[array]][%[[range]]]
// CHECK: bmodelica.array.store %[[sub]][%[[i]], %[[j]]], %[[value]]

func.func @StoreInArraySlice(%array: !bmodelica.array<10x20xf64>, %value: f64, %range: !bmodelica<range index>, %i: index, %j: index) {
    %sub = bmodelica.array.subscription %array[%range] : !bmodelica.array<10x20xf64>, !bmodelica<range index> -> !bmodelica.array<?x20xf64>
    bmodelica.array.store %sub[%i, %j], %value : !bmodelica.array<?x20xf64>
    return
}
