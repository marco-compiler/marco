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

// CHECK{LITERAL}: [1, 2, 3]

func @arrayAssignment() -> () {
    %x = modelica.alloca : !modelica.array<3x!modelica.int>
    %y = modelica.alloca : !modelica.array<3x!modelica.int>

    %c0 = constant 0 : index
    %0 = modelica.constant #modelica.int<1>
    modelica.store %x[%c0], %0 : !modelica.array<3x!modelica.int>

    %c1 = constant 1 : index
    %1 = modelica.constant #modelica.int<2>
    modelica.store %x[%c1], %1 : !modelica.array<3x!modelica.int>

    %c2 = constant 2 : index
    %2 = modelica.constant #modelica.int<3>
    modelica.store %x[%c2], %2 : !modelica.array<3x!modelica.int>

    modelica.assignment %y, %x : !modelica.array<3x!modelica.int>, !modelica.array<3x!modelica.int>
    modelica.print %y : !modelica.array<3x!modelica.int>

    return
}

// CHECK-NEXT{LITERAL}: [[1, 2], [3, 4], [5, 6]]

func @arraySliceAssignment() -> () {
    %x = modelica.alloca : !modelica.array<2x!modelica.int>
    %y = modelica.alloca : !modelica.array<2x!modelica.int>
    %z = modelica.alloca : !modelica.array<2x!modelica.int>
    %t = modelica.alloca : !modelica.array<3x2x!modelica.int>

    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c2 = constant 2 : index

    %0 = modelica.constant #modelica.int<1>
    modelica.store %x[%c0], %0 : !modelica.array<2x!modelica.int>

    %1 = modelica.constant #modelica.int<2>
    modelica.store %x[%c1], %1 : !modelica.array<2x!modelica.int>

    %2 = modelica.constant #modelica.int<3>
    modelica.store %y[%c0], %2 : !modelica.array<2x!modelica.int>

    %3 = modelica.constant #modelica.int<4>
    modelica.store %y[%c1], %3 : !modelica.array<2x!modelica.int>

    %4 = modelica.constant #modelica.int<5>
    modelica.store %z[%c0], %4 : !modelica.array<2x!modelica.int>

    %5 = modelica.constant #modelica.int<6>
    modelica.store %z[%c1], %5 : !modelica.array<2x!modelica.int>

    %slice0 = modelica.subscription %t[%c0] : (!modelica.array<3x2x!modelica.int>, index) -> !modelica.array<2x!modelica.int>
    modelica.assignment %slice0, %x : !modelica.array<2x!modelica.int>, !modelica.array<2x!modelica.int>

    %slice1 = modelica.subscription %t[%c1] : (!modelica.array<3x2x!modelica.int>, index) -> !modelica.array<2x!modelica.int>
    modelica.assignment %slice1, %y : !modelica.array<2x!modelica.int>, !modelica.array<2x!modelica.int>

    %slice2 = modelica.subscription %t[%c2] : (!modelica.array<3x2x!modelica.int>, index) -> !modelica.array<2x!modelica.int>
    modelica.assignment %slice2, %z : !modelica.array<2x!modelica.int>, !modelica.array<2x!modelica.int>

    modelica.print %t : !modelica.array<3x2x!modelica.int>

    return
}

func @main() -> () {
    call @arrayAssignment() : () -> ()
    call @arraySliceAssignment() : () -> ()
    return
}