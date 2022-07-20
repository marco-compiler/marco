// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @test
// CHECK-NEXT: %[[value:.*]] = modelica.constant #modelica.int<2>
// CHECK-NEXT: return %[[value]]

func.func @test() -> (!modelica.int) {
    %x = modelica.constant #modelica.int<6>
    %y = modelica.constant #modelica.int<3>
    %result = modelica.div_trunc %x, %y : (!modelica.int, !modelica.int) -> !modelica.int
    return %result : !modelica.int
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[value:.*]] = modelica.constant #modelica.int<2>
// CHECK-NEXT: return %[[value]]

func.func @test() -> (!modelica.int) {
    %x = modelica.constant #modelica.int<8>
    %y = modelica.constant #modelica.int<3>
    %result = modelica.div_trunc %x, %y : (!modelica.int, !modelica.int) -> !modelica.int
    return %result : !modelica.int
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[value:.*]] = modelica.constant #modelica.int<-3>
// CHECK-NEXT: return %[[value]]

func.func @test() -> (!modelica.int) {
    %x = modelica.constant #modelica.int<10>
    %y = modelica.constant #modelica.int<-3>
    %result = modelica.div_trunc %x, %y : (!modelica.int, !modelica.int) -> !modelica.int
    return %result : !modelica.int
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[value:.*]] = modelica.constant #modelica.int<-3>
// CHECK-NEXT: return %[[value]]

func.func @test() -> (!modelica.int) {
    %x = modelica.constant #modelica.int<-10>
    %y = modelica.constant #modelica.int<3>
    %result = modelica.div_trunc %x, %y : (!modelica.int, !modelica.int) -> !modelica.int
    return %result : !modelica.int
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[value:.*]] = modelica.constant #modelica.real<2.000000e+00>
// CHECK-NEXT: return %[[value]]

func.func @test() -> (!modelica.real) {
    %x = modelica.constant #modelica.real<6.0>
    %y = modelica.constant #modelica.real<3.0>
    %result = modelica.div_trunc %x, %y : (!modelica.real, !modelica.real) -> !modelica.real
    return %result : !modelica.real
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[value:.*]] = modelica.constant #modelica.real<2.000000e+00>
// CHECK-NEXT: return %[[value]]

func.func @test() -> (!modelica.real) {
    %x = modelica.constant #modelica.real<8.5>
    %y = modelica.constant #modelica.real<3.0>
    %result = modelica.div_trunc %x, %y : (!modelica.real, !modelica.real) -> !modelica.real
    return %result : !modelica.real
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[value:.*]] = modelica.constant #modelica.real<2.000000e+00>
// CHECK-NEXT: return %[[value]]

func.func @test() -> (!modelica.real) {
    %x = modelica.constant #modelica.real<3.0>
    %y = modelica.constant #modelica.real<1.4>
    %result = modelica.div_trunc %x, %y : (!modelica.real, !modelica.real) -> !modelica.real
    return %result : !modelica.real
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[value:.*]] = modelica.constant #modelica.real<-2.000000e+00>
// CHECK-NEXT: return %[[value]]

func.func @test() -> (!modelica.real) {
    %x = modelica.constant #modelica.real<-3.0>
    %y = modelica.constant #modelica.real<1.4>
    %result = modelica.div_trunc %x, %y : (!modelica.real, !modelica.real) -> !modelica.real
    return %result : !modelica.real
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[value:.*]] = modelica.constant #modelica.real<-2.000000e+00>
// CHECK-NEXT: return %[[value]]

func.func @test() -> (!modelica.real) {
    %x = modelica.constant #modelica.real<3.0>
    %y = modelica.constant #modelica.real<-1.4>
    %result = modelica.div_trunc %x, %y : (!modelica.real, !modelica.real) -> !modelica.real
    return %result : !modelica.real
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[value:.*]] = modelica.constant #modelica.real<3.000000e+00>
// CHECK-NEXT: return %[[value]]

func.func @test() -> (!modelica.real) {
    %x = modelica.constant #modelica.real<10.0>
    %y = modelica.constant #modelica.int<3>
    %result = modelica.div_trunc %x, %y : (!modelica.real, !modelica.int) -> !modelica.real
    return %result : !modelica.real
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[value:.*]] = modelica.constant #modelica.real<3.000000e+00>
// CHECK-NEXT: return %[[value]]

func.func @test() -> (!modelica.real) {
    %x = modelica.constant #modelica.int<10>
    %y = modelica.constant #modelica.real<3.0>
    %result = modelica.div_trunc %x, %y : (!modelica.int, !modelica.real) -> !modelica.real
    return %result : !modelica.real
}
