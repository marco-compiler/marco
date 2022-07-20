// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @test
// CHECK-NEXT: %[[VALUE:.*]] = modelica.constant #modelica.int
// CHECK-SAME: -4
// CHECK-NEXT: return %[[VALUE]]

func.func @test() -> (!modelica.int) {
    %x = modelica.constant #modelica.real<-3.14>
    %result = modelica.integer %x : !modelica.real -> !modelica.int
    return %result : !modelica.int
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[VALUE:.*]] = modelica.constant #modelica.int
// CHECK-SAME: 3
// CHECK-NEXT: return %[[VALUE]]

func.func @test() -> (!modelica.int) {
    %x = modelica.constant #modelica.real<3.14>
    %result = modelica.integer %x : !modelica.real -> !modelica.int
    return %result : !modelica.int
}
