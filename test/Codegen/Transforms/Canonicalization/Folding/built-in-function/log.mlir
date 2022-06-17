// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @test
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.real
// CHECK-SAME: 0.000000e+00
// CHECK-NEXT: return %[[VALUE]]

func.func @test() -> (!modelica.real) {
    %x = modelica.constant #modelica.real<1.0>
    %result = modelica.log %x : !modelica.real -> !modelica.real
    return %result : !modelica.real
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.real
// CHECK-SAME: 0.999999999831
// CHECK-NEXT: return %[[VALUE]]

func.func @test() -> (!modelica.real) {
    %x = modelica.constant #modelica.real<2.718281828>
    %result = modelica.log %x : !modelica.real -> !modelica.real
    return %result : !modelica.real
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.real
// CHECK-SAME: 2.000000000009
// CHECK-NEXT: return %[[VALUE]]

func.func @test() -> (!modelica.real) {
    %x = modelica.constant #modelica.real<7.389056099>
    %result = modelica.log %x : !modelica.real -> !modelica.real
    return %result : !modelica.real
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.real
// CHECK-SAME: -1.000000003184
// CHECK-NEXT: return %[[VALUE]]

func.func @test() -> (!modelica.real) {
    %x = modelica.constant #modelica.real<0.36787944>
    %result = modelica.log %x : !modelica.real -> !modelica.real
    return %result : !modelica.real
}
