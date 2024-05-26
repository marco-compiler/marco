// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @test
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica<bool true>
// CHECK-NEXT: return %[[VALUE]]

func.func @test() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica<bool false>
    %result = bmodelica.not %x : !bmodelica.bool -> !bmodelica.bool
    return %result : !bmodelica.bool
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica<bool false>
// CHECK-NEXT: return %[[VALUE]]

func.func @test() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica<bool true>
    %result = bmodelica.not %x : !bmodelica.bool -> !bmodelica.bool
    return %result : !bmodelica.bool
}
