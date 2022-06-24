// RUN: modelica-opt %s                             \
// RUN:     --convert-modelica-to-cf                \
// RUN:     --convert-modelica-to-llvm              \
// RUN:     --convert-scf-to-cf                     \
// RUN:     --convert-func-to-llvm                  \
// RUN:     --convert-cf-to-llvm                    \
// RUN:     --reconcile-unrealized-casts            \
// RUN: | mlir-cpu-runner                           \
// RUN:     -e main -entry-point-result=void -O0    \
// RUN:     -shared-libs=%runtime_lib               \
// RUN: | FileCheck %s

// CHECK{LITERAL}: -1

func.func @test_integerScalars() -> () {
    %x = modelica.constant #modelica.int<2>
    %y = modelica.constant #modelica.int<3>
    %result = modelica.sub %x, %y : (!modelica.int, !modelica.int) -> !modelica.int
    modelica.print %result : !modelica.int
    return
}

// CHECK-NEXT{LITERAL}: -1.500000e+00

func.func @test_realScalars() -> () {
    %x = modelica.constant #modelica.real<2.0>
    %y = modelica.constant #modelica.real<3.5>
    %result = modelica.sub %x, %y : (!modelica.real, !modelica.real) -> !modelica.real
    modelica.print %result : !modelica.real
    return
}

// CHECK-NEXT{LITERAL}: -1.500000e+00

func.func @test_mixedScalars() -> () {
    %x = modelica.constant #modelica.int<2>
    %y = modelica.constant #modelica.real<3.5>
    %result = modelica.sub %x, %y : (!modelica.int, !modelica.real) -> !modelica.real
    modelica.print %result : !modelica.real
    return
}

// CHECK-NEXT{LITERAL}: [-1, 1, 3, -3]

func.func @test_staticIntegerArrays() -> () {
    %x = modelica.alloca : !modelica.array<4x!modelica.int>
    %y = modelica.alloca : !modelica.array<4x!modelica.int>

    %c0 = arith.constant 0 : index
    %x0 = modelica.constant #modelica.int<1>
    %y0 = modelica.constant #modelica.int<2>
    modelica.store %x[%c0], %x0 : !modelica.array<4x!modelica.int>
    modelica.store %y[%c0], %y0 : !modelica.array<4x!modelica.int>

    %c1 = arith.constant 1 : index
    %x1 = modelica.constant #modelica.int<-1>
    %y1 = modelica.constant #modelica.int<-2>
    modelica.store %x[%c1], %x1 : !modelica.array<4x!modelica.int>
    modelica.store %y[%c1], %y1 : !modelica.array<4x!modelica.int>

    %c2 = arith.constant 2 : index
    %x2 = modelica.constant #modelica.int<1>
    %y2 = modelica.constant #modelica.int<-2>
    modelica.store %x[%c2], %x2 : !modelica.array<4x!modelica.int>
    modelica.store %y[%c2], %y2 : !modelica.array<4x!modelica.int>

    %c3 = arith.constant 3 : index
    %x3 = modelica.constant #modelica.int<-1>
    %y3 = modelica.constant #modelica.int<2>
    modelica.store %x[%c3], %x3 : !modelica.array<4x!modelica.int>
    modelica.store %y[%c3], %y3 : !modelica.array<4x!modelica.int>

    %result = modelica.sub %x, %y : (!modelica.array<4x!modelica.int>, !modelica.array<4x!modelica.int>) -> !modelica.array<4x!modelica.int>
    modelica.print %result : !modelica.array<4x!modelica.int>
    return
}

// CHECK-NEXT{LITERAL}: [-5.000000e-01, 5.000000e-01, 3.500000e+00, -3.500000e+00]

func.func @test_staticRealArrays() -> () {
    %x = modelica.alloca : !modelica.array<4x!modelica.real>
    %y = modelica.alloca : !modelica.array<4x!modelica.real>

    %c0 = arith.constant 0 : index
    %x0 = modelica.constant #modelica.real<1.5>
    %y0 = modelica.constant #modelica.real<2.0>
    modelica.store %x[%c0], %x0 : !modelica.array<4x!modelica.real>
    modelica.store %y[%c0], %y0 : !modelica.array<4x!modelica.real>

    %c1 = arith.constant 1 : index
    %x1 = modelica.constant #modelica.real<-1.5>
    %y1 = modelica.constant #modelica.real<-2.0>
    modelica.store %x[%c1], %x1 : !modelica.array<4x!modelica.real>
    modelica.store %y[%c1], %y1 : !modelica.array<4x!modelica.real>

    %c2 = arith.constant 2 : index
    %x2 = modelica.constant #modelica.real<1.5>
    %y2 = modelica.constant #modelica.real<-2.0>
    modelica.store %x[%c2], %x2 : !modelica.array<4x!modelica.real>
    modelica.store %y[%c2], %y2 : !modelica.array<4x!modelica.real>

    %c3 = arith.constant 3 : index
    %x3 = modelica.constant #modelica.real<-1.5>
    %y3 = modelica.constant #modelica.real<2.0>
    modelica.store %x[%c3], %x3 : !modelica.array<4x!modelica.real>
    modelica.store %y[%c3], %y3 : !modelica.array<4x!modelica.real>

    %result = modelica.sub %x, %y : (!modelica.array<4x!modelica.real>, !modelica.array<4x!modelica.real>) -> !modelica.array<4x!modelica.real>
    modelica.print %result : !modelica.array<4x!modelica.real>
    return
}

// CHECK-NEXT{LITERAL}: [-1, 1, 3, -3]

func.func @test_dynamicIntegerArrays() -> () {
    %s = arith.constant 4 : index

    %x = modelica.alloc %s : !modelica.array<?x!modelica.int>
    %y = modelica.alloc %s : !modelica.array<?x!modelica.int>

    %c0 = arith.constant 0 : index
    %x0 = modelica.constant #modelica.int<1>
    %y0 = modelica.constant #modelica.int<2>
    modelica.store %x[%c0], %x0 : !modelica.array<?x!modelica.int>
    modelica.store %y[%c0], %y0 : !modelica.array<?x!modelica.int>

    %c1 = arith.constant 1 : index
    %x1 = modelica.constant #modelica.int<-1>
    %y1 = modelica.constant #modelica.int<-2>
    modelica.store %x[%c1], %x1 : !modelica.array<?x!modelica.int>
    modelica.store %y[%c1], %y1 : !modelica.array<?x!modelica.int>

    %c2 = arith.constant 2 : index
    %x2 = modelica.constant #modelica.int<1>
    %y2 = modelica.constant #modelica.int<-2>
    modelica.store %x[%c2], %x2 : !modelica.array<?x!modelica.int>
    modelica.store %y[%c2], %y2 : !modelica.array<?x!modelica.int>

    %c3 = arith.constant 3 : index
    %x3 = modelica.constant #modelica.int<-1>
    %y3 = modelica.constant #modelica.int<2>
    modelica.store %x[%c3], %x3 : !modelica.array<?x!modelica.int>
    modelica.store %y[%c3], %y3 : !modelica.array<?x!modelica.int>

    %result = modelica.sub %x, %y : (!modelica.array<?x!modelica.int>, !modelica.array<?x!modelica.int>) -> !modelica.array<?x!modelica.int>
    modelica.free %x : !modelica.array<?x!modelica.int>
    modelica.free %y : !modelica.array<?x!modelica.int>
    modelica.print %result : !modelica.array<?x!modelica.int>
    modelica.free %result : !modelica.array<?x!modelica.int>
    return
}

// CHECK-NEXT{LITERAL}: [-5.000000e-01, 5.000000e-01, 3.500000e+00, -3.500000e+00]

func.func @test_dynamicRealArrays() -> () {
    %s = arith.constant 4 : index

    %x = modelica.alloc %s : !modelica.array<?x!modelica.real>
    %y = modelica.alloc %s : !modelica.array<?x!modelica.real>

    %c0 = arith.constant 0 : index
    %x0 = modelica.constant #modelica.real<1.5>
    %y0 = modelica.constant #modelica.real<2.0>
    modelica.store %x[%c0], %x0 : !modelica.array<?x!modelica.real>
    modelica.store %y[%c0], %y0 : !modelica.array<?x!modelica.real>

    %c1 = arith.constant 1 : index
    %x1 = modelica.constant #modelica.real<-1.5>
    %y1 = modelica.constant #modelica.real<-2.0>
    modelica.store %x[%c1], %x1 : !modelica.array<?x!modelica.real>
    modelica.store %y[%c1], %y1 : !modelica.array<?x!modelica.real>

    %c2 = arith.constant 2 : index
    %x2 = modelica.constant #modelica.real<1.5>
    %y2 = modelica.constant #modelica.real<-2.0>
    modelica.store %x[%c2], %x2 : !modelica.array<?x!modelica.real>
    modelica.store %y[%c2], %y2 : !modelica.array<?x!modelica.real>

    %c3 = arith.constant 3 : index
    %x3 = modelica.constant #modelica.real<-1.5>
    %y3 = modelica.constant #modelica.real<2.0>
    modelica.store %x[%c3], %x3 : !modelica.array<?x!modelica.real>
    modelica.store %y[%c3], %y3 : !modelica.array<?x!modelica.real>

    %result = modelica.sub %x, %y : (!modelica.array<?x!modelica.real>, !modelica.array<?x!modelica.real>) -> !modelica.array<?x!modelica.real>
    modelica.free %x : !modelica.array<?x!modelica.real>
    modelica.free %y : !modelica.array<?x!modelica.real>
    modelica.print %result : !modelica.array<?x!modelica.real>
    modelica.free %result : !modelica.array<?x!modelica.real>
    return
}

func.func @main() -> () {
    call @test_integerScalars() : () -> ()
    call @test_realScalars() : () -> ()
    call @test_mixedScalars() : () -> ()

    call @test_staticIntegerArrays() : () -> ()
    call @test_staticRealArrays() : () -> ()

    call @test_dynamicIntegerArrays() : () -> ()
    call @test_dynamicRealArrays() : () -> ()

    return
}
