// RUN: modelica-opt %s --canonicalize | FileCheck %s

// CHECK-LABEL: @test_integerScalars
// CHECK-NEXT: %[[VALUE:.*]] = modelica.constant #modelica.int<3>
// CHECK-NEXT: return %[[VALUE]]

func.func @test_integerScalars() -> (!modelica.int) {
    %x = modelica.constant #modelica.int<6>
    %y = modelica.constant #modelica.int<2>
    %result = modelica.div %x, %y : (!modelica.int, !modelica.int) -> !modelica.int
    return %result : !modelica.int
}

// CHECK-LABEL: @test_realScalars
// CHECK-NEXT: %[[VALUE:.*]] = modelica.constant #modelica.real
// CHECK-SAME: 3.000000e+00
// CHECK-NEXT: return %[[VALUE]]

func.func @test_realScalars() -> (!modelica.real) {
    %x = modelica.constant #modelica.real<6.0>
    %y = modelica.constant #modelica.real<2.0>
    %result = modelica.div %x, %y : (!modelica.real, !modelica.real) -> !modelica.real
    return %result : !modelica.real
}

// CHECK-LABEL: @test_mixedScalars1
// CHECK-NEXT: %[[VALUE:.*]] = modelica.constant #modelica.real
// CHECK-SAME: 3.000000e+00
// CHECK-NEXT: return %[[VALUE]]

func.func @test_mixedScalars1() -> (!modelica.real) {
    %x = modelica.constant #modelica.real<6.0>
    %y = modelica.constant #modelica.int<2>
    %result = modelica.div %x, %y : (!modelica.real, !modelica.int) -> !modelica.real
    return %result : !modelica.real
}

// CHECK-LABEL: @test_mixedScalars2
// CHECK-NEXT: %[[VALUE:.*]] = modelica.constant #modelica.real
// CHECK-SAME: 3.000000e+00
// CHECK-NEXT: return %[[VALUE]]

func.func @test_mixedScalars2() -> (!modelica.real) {
    %x = modelica.constant #modelica.real<6.0>
    %y = modelica.constant #modelica.int<2>
    %result = modelica.div %x, %y : (!modelica.real, !modelica.int) -> !modelica.real
    return %result : !modelica.real
}
