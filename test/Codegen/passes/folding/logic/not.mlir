// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @test
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.bool<true>
// CHECK-NEXT: return %[[VALUE]]

func @test() -> (!modelica.bool) {
    %x = modelica.constant #modelica.bool<false>
    %result = modelica.not %x : !modelica.bool -> !modelica.bool
    return %result : !modelica.bool
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.bool<false>
// CHECK-NEXT: return %[[VALUE]]

func @test() -> (!modelica.bool) {
    %x = modelica.constant #modelica.bool<true>
    %result = modelica.not %x : !modelica.bool -> !modelica.bool
    return %result : !modelica.bool
}
