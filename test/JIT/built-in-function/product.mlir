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

// CHECK: 120

func @test() -> () {
    %array = modelica.alloca : !modelica.array<5x!modelica.int>

    %c0 = constant 0 : index
    %0 = modelica.constant #modelica.int<1>
    modelica.store %array[%c0], %0 : !modelica.array<5x!modelica.int>

    %c1 = constant 1 : index
    %1 = modelica.constant #modelica.int<2>
    modelica.store %array[%c1], %1 : !modelica.array<5x!modelica.int>

    %c2 = constant 2 : index
    %2 = modelica.constant #modelica.int<3>
    modelica.store %array[%c2], %2 : !modelica.array<5x!modelica.int>

    %c3 = constant 3 : index
    %3 = modelica.constant #modelica.int<4>
    modelica.store %array[%c3], %3 : !modelica.array<5x!modelica.int>

    %c4 = constant 4 : index
    %4 = modelica.constant #modelica.int<5>
    modelica.store %array[%c4], %4 : !modelica.array<5x!modelica.int>

    %result = modelica.product %array : !modelica.array<5x!modelica.int> -> !modelica.int
    modelica.print %result : !modelica.int

    return
}

func @main() -> () {
    call @test() : () -> ()
    return
}
