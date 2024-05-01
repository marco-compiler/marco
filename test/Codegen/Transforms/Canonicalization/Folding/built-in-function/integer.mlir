// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @test
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.int
// CHECK-SAME: -4
// CHECK-NEXT: return %[[VALUE]]

func.func @test() -> (!bmodelica.int) {
    %x = bmodelica.constant #bmodelica.real<-3.14>
    %result = bmodelica.integer %x : !bmodelica.real -> !bmodelica.int
    return %result : !bmodelica.int
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.int
// CHECK-SAME: 3
// CHECK-NEXT: return %[[VALUE]]

func.func @test() -> (!bmodelica.int) {
    %x = bmodelica.constant #bmodelica.real<3.14>
    %result = bmodelica.integer %x : !bmodelica.real -> !bmodelica.int
    return %result : !bmodelica.int
}
