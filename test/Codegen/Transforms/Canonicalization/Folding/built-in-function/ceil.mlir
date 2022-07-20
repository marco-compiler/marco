// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @test
// CHECK-NEXT: %[[VALUE:.*]] = modelica.constant #modelica.real
// CHECK-SAME: -3.000000e+00
// CHECK-NEXT: return %[[VALUE]]

func.func @test() -> (!modelica.real) {
    %x = modelica.constant #modelica.real<-3.14>
    %result = modelica.ceil %x : !modelica.real -> !modelica.real
    return %result : !modelica.real
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[VALUE:.*]] = modelica.constant #modelica.real
// CHECK-SAME: 4.000000e+00
// CHECK-NEXT: return %[[VALUE]]

func.func @test() -> (!modelica.real) {
    %x = modelica.constant #modelica.real<3.14>
    %result = modelica.ceil %x : !modelica.real -> !modelica.real
    return %result : !modelica.real
}
