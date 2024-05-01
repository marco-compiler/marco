// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @test
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.real
// CHECK-SAME: 1.000000e+00
// CHECK-NEXT: return %[[VALUE]]

func.func @test() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica.real<0.0>
    %result = bmodelica.exp %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.real
// CHECK-SAME: 2.718281828459
// CHECK-NEXT: return %[[VALUE]]

func.func @test() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica.real<1.0>
    %result = bmodelica.exp %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.real
// CHECK-SAME: 7.389056098930
// CHECK-NEXT: return %[[VALUE]]

func.func @test() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica.real<2.0>
    %result = bmodelica.exp %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.real
// CHECK-SAME: 0.135335283236
// CHECK-NEXT: return %[[VALUE]]

func.func @test() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica.real<-2.0>
    %result = bmodelica.exp %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
}
