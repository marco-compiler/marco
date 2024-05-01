// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @test
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.real<0.000000e+00>
// CHECK-NEXT: return %[[VALUE]]

func.func @test() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica.real<1.0>
    %result = bmodelica.acos %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.real
// CHECK-SAME: 0.523598777167
// CHECK-NEXT: return %[[VALUE]]

func.func @test() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica.real<0.866025403>
    %result = bmodelica.acos %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.real
// CHECK-SAME: 0.785398163661
// CHECK-NEXT: return %[[VALUE]]

func.func @test() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica.real<0.707106781>
    %result = bmodelica.acos %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.real
// CHECK-SAME: 1.570796326794
// CHECK-NEXT: return %[[VALUE]]

func.func @test() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica.real<0.0>
    %result = bmodelica.acos %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.real
// CHECK-SAME: 2.356194489928
// CHECK-NEXT: return %[[VALUE]]

func.func @test() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica.real<-0.707106781>
    %result = bmodelica.acos %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.real
// CHECK-SAME: 2.617993876422
// CHECK-NEXT: return %[[VALUE]]

func.func @test() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica.real<-0.866025403>
    %result = bmodelica.acos %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.real
// CHECK-SAME: 3.141592653589
// CHECK-NEXT: return %[[VALUE]]

func.func @test() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica.real<-1.0>
    %result = bmodelica.acos %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
}
