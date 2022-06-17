// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @test
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.real
// CHECK-SAME: 1.000000e+00
// CHECK-NEXT: return %[[VALUE]]

func.func @test() -> (!modelica.real) {
    %x = modelica.constant #modelica.real<0.0>
    %result = modelica.exp %x : !modelica.real -> !modelica.real
    return %result : !modelica.real
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.real
// CHECK-SAME: 2.718281828459
// CHECK-NEXT: return %[[VALUE]]

func.func @test() -> (!modelica.real) {
    %x = modelica.constant #modelica.real<1.0>
    %result = modelica.exp %x : !modelica.real -> !modelica.real
    return %result : !modelica.real
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.real
// CHECK-SAME: 7.389056098930
// CHECK-NEXT: return %[[VALUE]]

func.func @test() -> (!modelica.real) {
    %x = modelica.constant #modelica.real<2.0>
    %result = modelica.exp %x : !modelica.real -> !modelica.real
    return %result : !modelica.real
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.real
// CHECK-SAME: 0.135335283236
// CHECK-NEXT: return %[[VALUE]]

func.func @test() -> (!modelica.real) {
    %x = modelica.constant #modelica.real<-2.0>
    %result = modelica.exp %x : !modelica.real -> !modelica.real
    return %result : !modelica.real
}
