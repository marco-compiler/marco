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

// CHECK: 1
// CHECK-NEXT: 1
// CHECK-NEXT: -2
// CHECK-NEXT: -2
// CHECK-NEXT: -1
// CHECK-NEXT: -1

func @test_scalars() -> () {
    %size = constant 6 : index

    %y = modelica.alloca %size : index -> !modelica.array<stack, ?x!modelica.int>
    %x = modelica.alloca %size : index -> !modelica.array<stack, ?x!modelica.int>

    %c0 = constant 0 : index
    %x0 = modelica.constant #modelica.int<1> : !modelica.int
    %y0 = modelica.constant #modelica.int<2> : !modelica.int
    modelica.store %x[%c0], %x0 : !modelica.array<stack, ?x!modelica.int>
    modelica.store %y[%c0], %y0 : !modelica.array<stack, ?x!modelica.int>

    %c1 = constant 1 : index
    %x1 = modelica.constant #modelica.int<2> : !modelica.int
    %y1 = modelica.constant #modelica.int<1> : !modelica.int
    modelica.store %x[%c1], %x1 : !modelica.array<stack, ?x!modelica.int>
    modelica.store %y[%c1], %y1 : !modelica.array<stack, ?x!modelica.int>

    %c2 = constant 2 : index
    %x2 = modelica.constant #modelica.int<-1> : !modelica.int
    %y2 = modelica.constant #modelica.int<-2> : !modelica.int
    modelica.store %x[%c2], %x2 : !modelica.array<stack, ?x!modelica.int>
    modelica.store %y[%c2], %y2 : !modelica.array<stack, ?x!modelica.int>

    %c3 = constant 3 : index
    %x3 = modelica.constant #modelica.int<-2> : !modelica.int
    %y3 = modelica.constant #modelica.int<-1> : !modelica.int
    modelica.store %x[%c3], %x3 : !modelica.array<stack, ?x!modelica.int>
    modelica.store %y[%c3], %y3 : !modelica.array<stack, ?x!modelica.int>

    %c4 = constant 4 : index
    %x4 = modelica.constant #modelica.int<1> : !modelica.int
    %y4 = modelica.constant #modelica.int<-1> : !modelica.int
    modelica.store %x[%c4], %x4 : !modelica.array<stack, ?x!modelica.int>
    modelica.store %y[%c4], %y4 : !modelica.array<stack, ?x!modelica.int>

    %c5 = constant 5 : index
    %x5 = modelica.constant #modelica.int<-1> : !modelica.int
    %y5 = modelica.constant #modelica.int<1> : !modelica.int
    modelica.store %x[%c5], %x5 : !modelica.array<stack, ?x!modelica.int>
    modelica.store %y[%c5], %y5 : !modelica.array<stack, ?x!modelica.int>

    scf.for %i = %c0 to %size step %c1 {
      %xi = modelica.load %x[%i] : !modelica.array<stack, ?x!modelica.int>
      %yi = modelica.load %y[%i] : !modelica.array<stack, ?x!modelica.int>
      %result = modelica.min %xi, %yi : (!modelica.int, !modelica.int) -> !modelica.int
      modelica.print %result : !modelica.int
    }

    return
}

// CHECK-NEXT: -3

func @test_array() -> () {
    %size = constant 6 : index
    %array = modelica.alloca %size : index -> !modelica.array<stack, ?x!modelica.int>

    %c0 = constant 0 : index
    %0 = modelica.constant #modelica.int<-1> : !modelica.int
    modelica.store %array[%c0], %0 : !modelica.array<stack, ?x!modelica.int>

    %c1 = constant 1 : index
    %1 = modelica.constant #modelica.int<9> : !modelica.int
    modelica.store %array[%c1], %1 : !modelica.array<stack, ?x!modelica.int>

    %c2 = constant 2 : index
    %2 = modelica.constant #modelica.int<-3> : !modelica.int
    modelica.store %array[%c2], %2 : !modelica.array<stack, ?x!modelica.int>

    %c3 = constant 3 : index
    %3 = modelica.constant #modelica.int<0> : !modelica.int
    modelica.store %array[%c3], %3 : !modelica.array<stack, ?x!modelica.int>

    %c4 = constant 4 : index
    %4 = modelica.constant #modelica.int<4> : !modelica.int
    modelica.store %array[%c4], %4 : !modelica.array<stack, ?x!modelica.int>

    %c5 = constant 5 : index
    %5 = modelica.constant #modelica.int<3> : !modelica.int
    modelica.store %array[%c5], %5 : !modelica.array<stack, ?x!modelica.int>

    %result = modelica.min %array : !modelica.array<stack, ?x!modelica.int> -> !modelica.int
    modelica.print %result : !modelica.int

    return
}

func @main() -> () {
    call @test_scalars() : () -> ()
    call @test_array() : () -> ()
    return
}
