// RUN: modelica-opt %s                             \
// RUN:     --convert-modelica                      \
// RUN:     --convert-modelica-to-cfg               \
// RUN:     --convert-to-llvm                       \
// RUN: | mlir-cpu-runner                           \
// RUN:     -e main -entry-point-result=void -O0    \
// RUN:     -shared-libs=%runtime_lib               \
// RUN: | FileCheck %s

// CHECK{LITERAL}: -2
// CHECK-NEXT{LITERAL}: 3

func @test_integerScalars() -> () {
    %x = modelica.constant #modelica.int<2>
    %xNeg = modelica.neg %x : !modelica.int -> !modelica.int
    modelica.print %xNeg : !modelica.int

    %y = modelica.constant #modelica.int<-3>
    %yNeg = modelica.neg %y : !modelica.int -> !modelica.int
    modelica.print %yNeg : !modelica.int

    return
}

// CHECK-NEXT{LITERAL}: -2.500000e+00
// CHECK-NEXT{LITERAL}: 3.500000e+00

func @test_realScalars() -> () {
    %x = modelica.constant #modelica.real<2.5>
    %xNeg = modelica.neg %x : !modelica.real -> !modelica.real
    modelica.print %xNeg : !modelica.real

    %y = modelica.constant #modelica.real<-3.5>
    %yNeg = modelica.neg %y : !modelica.real -> !modelica.real
    modelica.print %yNeg : !modelica.real

    return
}

// CHECK-NEXT{LITERAL}: [-2, 3]

func @test_staticIntegerArray() -> () {
    %array = modelica.alloca : !modelica.array<2x!modelica.int>

    %c0 = constant 0 : index
    %0 = modelica.constant #modelica.int<2>
    modelica.store %array[%c0], %0 : !modelica.array<2x!modelica.int>

    %c1 = constant 1 : index
    %1 = modelica.constant #modelica.int<-3>
    modelica.store %array[%c1], %1 : !modelica.array<2x!modelica.int>

    %result = modelica.neg %array : !modelica.array<2x!modelica.int> -> !modelica.array<2x!modelica.int>
    modelica.print %result : !modelica.array<2x!modelica.int>
    return
}

// CHECK-NEXT{LITERAL}: [-2.500000e+00, 3.500000e+00]

func @test_staticRealArray() -> () {
    %array = modelica.alloca : !modelica.array<2x!modelica.real>

    %c0 = constant 0 : index
    %0 = modelica.constant #modelica.real<2.5>
    modelica.store %array[%c0], %0 : !modelica.array<2x!modelica.real>

    %c1 = constant 1 : index
    %1 = modelica.constant #modelica.real<-3.5>
    modelica.store %array[%c1], %1 : !modelica.array<2x!modelica.real>

    %result = modelica.neg %array : !modelica.array<2x!modelica.real> -> !modelica.array<2x!modelica.real>
    modelica.print %result : !modelica.array<2x!modelica.real>
    return
}

// CHECK-NEXT{LITERAL}: [-2, 3]

func @test_dynamicIntegerArray() -> () {
    %s = constant 2 : index

    %array = modelica.alloc %s : !modelica.array<?x!modelica.int>

    %c0 = constant 0 : index
    %0 = modelica.constant #modelica.int<2>
    modelica.store %array[%c0], %0 : !modelica.array<?x!modelica.int>

    %c1 = constant 1 : index
    %1 = modelica.constant #modelica.int<-3>
    modelica.store %array[%c1], %1 : !modelica.array<?x!modelica.int>

    %result = modelica.neg %array : !modelica.array<?x!modelica.int> -> !modelica.array<?x!modelica.int>
    modelica.free %array : !modelica.array<?x!modelica.int>
    modelica.print %result : !modelica.array<?x!modelica.int>
    modelica.free %result : !modelica.array<?x!modelica.int>
    return
}

// CHECK-NEXT{LITERAL}: [-2.500000e+00, 3.500000e+00]

func @test_dynamicRealArray() -> () {
    %s = constant 2 : index

    %array = modelica.alloc %s : !modelica.array<?x!modelica.real>

    %c0 = constant 0 : index
    %0 = modelica.constant #modelica.real<2.5>
    modelica.store %array[%c0], %0 : !modelica.array<?x!modelica.real>

    %c1 = constant 1 : index
    %1 = modelica.constant #modelica.real<-3.5>
    modelica.store %array[%c1], %1 : !modelica.array<?x!modelica.real>

    %result = modelica.neg %array : !modelica.array<?x!modelica.real> -> !modelica.array<?x!modelica.real>
    modelica.free %array : !modelica.array<?x!modelica.real>
    modelica.print %result : !modelica.array<?x!modelica.real>
    modelica.free %result : !modelica.array<?x!modelica.real>
    return
}

func @main() -> () {
    call @test_integerScalars() : () -> ()
    call @test_realScalars() : () -> ()

    call @test_staticIntegerArray() : () -> ()
    call @test_staticRealArray() : () -> ()

    call @test_dynamicIntegerArray() : () -> ()
    call @test_dynamicRealArray() : () -> ()

    return
}
