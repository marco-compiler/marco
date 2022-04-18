// RUN: modelica-opt %s                             \
// RUN:     --convert-modelica                      \
// RUN:     --convert-modelica-to-cfg               \
// RUN:     --convert-to-llvm                       \
// RUN:     --remove-unrealized-casts               \
// RUN: | mlir-opt                                  \
// RUN:      --convert-scf-to-std                   \
// RUN: | mlir-cpu-runner                           \
// RUN:     -e main -entry-point-result=void -O0    \
// RUN:     -shared-libs=%runtime_lib               \
// RUN: | FileCheck %s

// CHECK{LITERAL}: [[1, 2, 3], [2, 5, 6], [3, 6, 9]]

func @test() -> () {
    %matrix = modelica.alloca : !modelica.array<3x3x!modelica.int>

    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c2 = constant 2 : index

    %00 = modelica.constant #modelica.int<1>
    modelica.store %matrix[%c0, %c0], %00 : !modelica.array<3x3x!modelica.int>

    %01 = modelica.constant #modelica.int<2>
    modelica.store %matrix[%c0, %c1], %01 : !modelica.array<3x3x!modelica.int>

    %02 = modelica.constant #modelica.int<3>
    modelica.store %matrix[%c0, %c2], %02 : !modelica.array<3x3x!modelica.int>

    %10 = modelica.constant #modelica.int<4>
    modelica.store %matrix[%c1, %c0], %10 : !modelica.array<3x3x!modelica.int>

    %11 = modelica.constant #modelica.int<5>
    modelica.store %matrix[%c1, %c1], %11 : !modelica.array<3x3x!modelica.int>

    %12 = modelica.constant #modelica.int<6>
    modelica.store %matrix[%c1, %c2], %12 : !modelica.array<3x3x!modelica.int>

    %20 = modelica.constant #modelica.int<7>
    modelica.store %matrix[%c2, %c0], %20 : !modelica.array<3x3x!modelica.int>

    %21 = modelica.constant #modelica.int<8>
    modelica.store %matrix[%c2, %c1], %21 : !modelica.array<3x3x!modelica.int>

    %22 = modelica.constant #modelica.int<9>
    modelica.store %matrix[%c2, %c2], %22 : !modelica.array<3x3x!modelica.int>

    %result = modelica.symmetric %matrix : !modelica.array<3x3x!modelica.int> -> !modelica.array<3x3x!modelica.int>
    modelica.print %result : !modelica.array<3x3x!modelica.int>

    return
}

func @main() -> () {
    call @test() : () -> ()
    return
}
