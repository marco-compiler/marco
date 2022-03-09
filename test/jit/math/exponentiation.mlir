// RUN: modelica-opt %s                             \
// RUN:     --convert-modelica                      \
// RUN:     --convert-modelica-to-cfg               \
// RUN:     --convert-to-llvm                       \
// RUN: | mlir-opt                                  \
// RUN:      --convert-scf-to-std                   \
// RUN: | mlir-cpu-runner                           \
// RUN:     -e main -entry-point-result=void -O0    \
// RUN:     -shared-libs=%runtime_lib               \
// RUN: | FileCheck %s

// CHECK{LITERAL}: 5
// CHECK-NEXT{LITERAL}: 81
// CHECK-NEXT{LITERAL}: 1
// CHECK-NEXT{LITERAL}: 16
// CHECK-NEXT{LITERAL}: 0
// CHECK-NEXT{LITERAL}: -8
// CHECK-NEXT{LITERAL}: 4

func @test_scalarBase() -> () {
    %size = constant 7 : index

    %base = modelica.alloca %size : index -> !modelica.array<stack, ?x!modelica.int>
    %exp = modelica.alloca %size : index -> !modelica.array<stack, ?x!modelica.int>

    %c0 = constant 0 : index
    %b0 = modelica.constant #modelica.int<5> : !modelica.int
    %e0 = modelica.constant #modelica.int<1> : !modelica.int
    modelica.store %base[%c0], %b0 : !modelica.array<stack, ?x!modelica.int>
    modelica.store %exp[%c0], %e0 : !modelica.array<stack, ?x!modelica.int>

    %c1 = constant 1 : index
    %b1 = modelica.constant #modelica.int<3> : !modelica.int
    %e1 = modelica.constant #modelica.int<4> : !modelica.int
    modelica.store %base[%c1], %b1 : !modelica.array<stack, ?x!modelica.int>
    modelica.store %exp[%c1], %e1 : !modelica.array<stack, ?x!modelica.int>

    %c2 = constant 2 : index
    %b2 = modelica.constant #modelica.int<2> : !modelica.int
    %e2 = modelica.constant #modelica.int<0> : !modelica.int
    modelica.store %base[%c2], %b2 : !modelica.array<stack, ?x!modelica.int>
    modelica.store %exp[%c2], %e2 : !modelica.array<stack, ?x!modelica.int>

    %c3 = constant 3 : index
    %b3 = modelica.constant #modelica.int<4> : !modelica.int
    %e3 = modelica.constant #modelica.int<2> : !modelica.int
    modelica.store %base[%c3], %b3 : !modelica.array<stack, ?x!modelica.int>
    modelica.store %exp[%c3], %e3 : !modelica.array<stack, ?x!modelica.int>

    %c4 = constant 4 : index
    %b4 = modelica.constant #modelica.int<0> : !modelica.int
    %e4 = modelica.constant #modelica.int<3> : !modelica.int
    modelica.store %base[%c4], %b4 : !modelica.array<stack, ?x!modelica.int>
    modelica.store %exp[%c4], %e4 : !modelica.array<stack, ?x!modelica.int>

    %c5 = constant 5 : index
    %b5 = modelica.constant #modelica.int<-2> : !modelica.int
    %e5 = modelica.constant #modelica.int<3> : !modelica.int
    modelica.store %base[%c5], %b5 : !modelica.array<stack, ?x!modelica.int>
    modelica.store %exp[%c5], %e5 : !modelica.array<stack, ?x!modelica.int>

    %c6 = constant 6 : index
    %b6 = modelica.constant #modelica.int<-2> : !modelica.int
    %e6 = modelica.constant #modelica.int<2> : !modelica.int
    modelica.store %base[%c6], %b6 : !modelica.array<stack, ?x!modelica.int>
    modelica.store %exp[%c6], %e6 : !modelica.array<stack, ?x!modelica.int>

    scf.for %i = %c0 to %size step %c1 {
      %b = modelica.load %base[%i] : !modelica.array<stack, ?x!modelica.int>
      %e = modelica.load %exp[%i] : !modelica.array<stack, ?x!modelica.int>
      %result = modelica.pow %b, %e : (!modelica.int, !modelica.int) -> !modelica.int
      modelica.print %result : !modelica.int
    }

    return
}

// CHECK-NEXT{LITERAL}: [[37, 54], [81, 118]]

func @test_matrixBase() -> () {
    %size = constant 2 : index
    %base = modelica.alloca %size, %size : (index, index) -> !modelica.array<stack, ?x?x!modelica.int>

    %c0 = constant 0 : index
    %c1 = constant 1 : index

    %00 = modelica.constant #modelica.int<1> : !modelica.int
    modelica.store %base[%c0, %c0], %00 : !modelica.array<stack, ?x?x!modelica.int>

    %01 = modelica.constant #modelica.int<2> : !modelica.int
    modelica.store %base[%c0, %c1], %01 : !modelica.array<stack, ?x?x!modelica.int>

    %10 = modelica.constant #modelica.int<3> : !modelica.int
    modelica.store %base[%c1, %c0], %10 : !modelica.array<stack, ?x?x!modelica.int>

    %11 = modelica.constant #modelica.int<4> : !modelica.int
    modelica.store %base[%c1, %c1], %11 : !modelica.array<stack, ?x?x!modelica.int>

    %exp = modelica.constant #modelica.int<3> : !modelica.int

    %result = modelica.pow %base, %exp : (!modelica.array<stack, ?x?x!modelica.int>, !modelica.int) -> !modelica.array<stack, ?x?x!modelica.int>
    modelica.print %result : !modelica.array<stack, ?x?x!modelica.int>

    return
}

func @main() -> () {
    call @test_scalarBase() : () -> ()
    call @test_matrixBase() : () -> ()

    return
}
