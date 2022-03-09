// RUN: modelica-opt %s                             \
// RUN:     --convert-modelica                      \
// RUN:     --convert-modelica-to-cfg               \
// RUN:     --convert-to-llvm                       \
// RUN: | mlir-cpu-runner                           \
// RUN:     -e main -entry-point-result=void -O0    \
// RUN:     -shared-libs=%runtime_lib               \
// RUN: | FileCheck %s

// CHECK{LITERAL}: 5

func @test_integerScalars() -> () {
    %x = modelica.constant #modelica.int<2> : !modelica.int
    %y = modelica.constant #modelica.int<3> : !modelica.int
    %result = modelica.add %x, %y : (!modelica.int, !modelica.int) -> !modelica.int
    modelica.print %result : !modelica.int
    return
}

// CHECK-NEXT{LITERAL}: 5.500000e+00

func @test_realScalars() -> () {
    %x = modelica.constant #modelica.real<2.0> : !modelica.real
    %y = modelica.constant #modelica.real<3.5> : !modelica.real
    %result = modelica.add %x, %y : (!modelica.real, !modelica.real) -> !modelica.real
    modelica.print %result : !modelica.real
    return
}

// CHECK-NEXT{LITERAL}: 5.500000e+00

func @test_mixedScalars() -> () {
    %x = modelica.constant #modelica.int<2> : !modelica.int
    %y = modelica.constant #modelica.real<3.5> : !modelica.real
    %result = modelica.add %x, %y : (!modelica.int, !modelica.real) -> !modelica.real
    modelica.print %result : !modelica.real
    return
}

// CHECK-NEXT{LITERAL}: [3, -3, -1, 1]

func @test_staticIntegerArrays() -> () {
    %x = modelica.alloca : !modelica.array<stack, 4x!modelica.int>
    %y = modelica.alloca : !modelica.array<stack, 4x!modelica.int>

    %c0 = constant 0 : index
    %x0 = modelica.constant #modelica.int<1> : !modelica.int
    %y0 = modelica.constant #modelica.int<2> : !modelica.int
    modelica.store %x[%c0], %x0 : !modelica.array<stack, 4x!modelica.int>
    modelica.store %y[%c0], %y0 : !modelica.array<stack, 4x!modelica.int>

    %c1 = constant 1 : index
    %x1 = modelica.constant #modelica.int<-1> : !modelica.int
    %y1 = modelica.constant #modelica.int<-2> : !modelica.int
    modelica.store %x[%c1], %x1 : !modelica.array<stack, 4x!modelica.int>
    modelica.store %y[%c1], %y1 : !modelica.array<stack, 4x!modelica.int>

    %c2 = constant 2 : index
    %x2 = modelica.constant #modelica.int<1> : !modelica.int
    %y2 = modelica.constant #modelica.int<-2> : !modelica.int
    modelica.store %x[%c2], %x2 : !modelica.array<stack, 4x!modelica.int>
    modelica.store %y[%c2], %y2 : !modelica.array<stack, 4x!modelica.int>

    %c3 = constant 3 : index
    %x3 = modelica.constant #modelica.int<-1> : !modelica.int
    %y3 = modelica.constant #modelica.int<2> : !modelica.int
    modelica.store %x[%c3], %x3 : !modelica.array<stack, 4x!modelica.int>
    modelica.store %y[%c3], %y3 : !modelica.array<stack, 4x!modelica.int>

    %result = modelica.add %x, %y : (!modelica.array<stack, 4x!modelica.int>, !modelica.array<stack, 4x!modelica.int>) -> !modelica.array<stack, 4x!modelica.int>
    modelica.print %result : !modelica.array<stack, 4x!modelica.int>
    return
}

// CHECK-NEXT{LITERAL}: [3.500000e+00, -3.500000e+00, -5.000000e-01, 5.000000e-01]

func @test_staticRealArrays() -> () {
    %x = modelica.alloca : !modelica.array<stack, 4x!modelica.real>
    %y = modelica.alloca : !modelica.array<stack, 4x!modelica.real>

    %c0 = constant 0 : index
    %x0 = modelica.constant #modelica.real<1.5> : !modelica.real
    %y0 = modelica.constant #modelica.real<2.0> : !modelica.real
    modelica.store %x[%c0], %x0 : !modelica.array<stack, 4x!modelica.real>
    modelica.store %y[%c0], %y0 : !modelica.array<stack, 4x!modelica.real>

    %c1 = constant 1 : index
    %x1 = modelica.constant #modelica.real<-1.5> : !modelica.real
    %y1 = modelica.constant #modelica.real<-2.0> : !modelica.real
    modelica.store %x[%c1], %x1 : !modelica.array<stack, 4x!modelica.real>
    modelica.store %y[%c1], %y1 : !modelica.array<stack, 4x!modelica.real>

    %c2 = constant 2 : index
    %x2 = modelica.constant #modelica.real<1.5> : !modelica.real
    %y2 = modelica.constant #modelica.real<-2.0> : !modelica.real
    modelica.store %x[%c2], %x2 : !modelica.array<stack, 4x!modelica.real>
    modelica.store %y[%c2], %y2 : !modelica.array<stack, 4x!modelica.real>

    %c3 = constant 3 : index
    %x3 = modelica.constant #modelica.real<-1.5> : !modelica.real
    %y3 = modelica.constant #modelica.real<2.0> : !modelica.real
    modelica.store %x[%c3], %x3 : !modelica.array<stack, 4x!modelica.real>
    modelica.store %y[%c3], %y3 : !modelica.array<stack, 4x!modelica.real>

    %result = modelica.add %x, %y : (!modelica.array<stack, 4x!modelica.real>, !modelica.array<stack, 4x!modelica.real>) -> !modelica.array<stack, 4x!modelica.real>
    modelica.print %result : !modelica.array<stack, 4x!modelica.real>
    return
}

// CHECK-NEXT{LITERAL}: [3, -3, -1, 1]

func @test_dynamicIntegerArrays() -> () {
    %s = modelica.constant 4 : index

    %x = modelica.alloc %s : index -> !modelica.array<heap, ?x!modelica.int>
    %y = modelica.alloc %s : index -> !modelica.array<heap, ?x!modelica.int>

    %c0 = constant 0 : index
    %x0 = modelica.constant #modelica.int<1> : !modelica.int
    %y0 = modelica.constant #modelica.int<2> : !modelica.int
    modelica.store %x[%c0], %x0 : !modelica.array<heap, ?x!modelica.int>
    modelica.store %y[%c0], %y0 : !modelica.array<heap, ?x!modelica.int>

    %c1 = constant 1 : index
    %x1 = modelica.constant #modelica.int<-1> : !modelica.int
    %y1 = modelica.constant #modelica.int<-2> : !modelica.int
    modelica.store %x[%c1], %x1 : !modelica.array<heap, ?x!modelica.int>
    modelica.store %y[%c1], %y1 : !modelica.array<heap, ?x!modelica.int>

    %c2 = constant 2 : index
    %x2 = modelica.constant #modelica.int<1> : !modelica.int
    %y2 = modelica.constant #modelica.int<-2> : !modelica.int
    modelica.store %x[%c2], %x2 : !modelica.array<heap, ?x!modelica.int>
    modelica.store %y[%c2], %y2 : !modelica.array<heap, ?x!modelica.int>

    %c3 = constant 3 : index
    %x3 = modelica.constant #modelica.int<-1> : !modelica.int
    %y3 = modelica.constant #modelica.int<2> : !modelica.int
    modelica.store %x[%c3], %x3 : !modelica.array<heap, ?x!modelica.int>
    modelica.store %y[%c3], %y3 : !modelica.array<heap, ?x!modelica.int>

    %result = modelica.add %x, %y : (!modelica.array<heap, ?x!modelica.int>, !modelica.array<heap, ?x!modelica.int>) -> !modelica.array<heap, ?x!modelica.int>
    modelica.free %x : !modelica.array<heap, ?x!modelica.int>
    modelica.free %y : !modelica.array<heap, ?x!modelica.int>
    modelica.print %result : !modelica.array<heap, ?x!modelica.int>
    modelica.free %result : !modelica.array<heap, ?x!modelica.int>
    return
}

// CHECK-NEXT{LITERAL}: [3.500000e+00, -3.500000e+00, -5.000000e-01, 5.000000e-01]

func @test_dynamicRealArrays() -> () {
    %s = modelica.constant 4 : index

    %x = modelica.alloc %s : index -> !modelica.array<heap, ?x!modelica.real>
    %y = modelica.alloc %s : index -> !modelica.array<heap, ?x!modelica.real>

    %c0 = constant 0 : index
    %x0 = modelica.constant #modelica.real<1.5> : !modelica.real
    %y0 = modelica.constant #modelica.real<2.0> : !modelica.real
    modelica.store %x[%c0], %x0 : !modelica.array<heap, ?x!modelica.real>
    modelica.store %y[%c0], %y0 : !modelica.array<heap, ?x!modelica.real>

    %c1 = constant 1 : index
    %x1 = modelica.constant #modelica.real<-1.5> : !modelica.real
    %y1 = modelica.constant #modelica.real<-2.0> : !modelica.real
    modelica.store %x[%c1], %x1 : !modelica.array<heap, ?x!modelica.real>
    modelica.store %y[%c1], %y1 : !modelica.array<heap, ?x!modelica.real>

    %c2 = constant 2 : index
    %x2 = modelica.constant #modelica.real<1.5> : !modelica.real
    %y2 = modelica.constant #modelica.real<-2.0> : !modelica.real
    modelica.store %x[%c2], %x2 : !modelica.array<heap, ?x!modelica.real>
    modelica.store %y[%c2], %y2 : !modelica.array<heap, ?x!modelica.real>

    %c3 = constant 3 : index
    %x3 = modelica.constant #modelica.real<-1.5> : !modelica.real
    %y3 = modelica.constant #modelica.real<2.0> : !modelica.real
    modelica.store %x[%c3], %x3 : !modelica.array<heap, ?x!modelica.real>
    modelica.store %y[%c3], %y3 : !modelica.array<heap, ?x!modelica.real>

    %result = modelica.add %x, %y : (!modelica.array<heap, ?x!modelica.real>, !modelica.array<heap, ?x!modelica.real>) -> !modelica.array<heap, ?x!modelica.real>
    modelica.free %x : !modelica.array<heap, ?x!modelica.real>
    modelica.free %y : !modelica.array<heap, ?x!modelica.real>
    modelica.print %result : !modelica.array<heap, ?x!modelica.real>
    modelica.free %result : !modelica.array<heap, ?x!modelica.real>
    return
}

func @main() -> () {
    call @test_integerScalars() : () -> ()
    call @test_realScalars() : () -> ()
    call @test_mixedScalars() : () -> ()

    call @test_staticIntegerArrays() : () -> ()
    call @test_staticRealArrays() : () -> ()

    call @test_dynamicIntegerArrays() : () -> ()
    call @test_dynamicRealArrays() : () -> ()

    return
}
