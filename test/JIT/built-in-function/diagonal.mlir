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

// CHECK{LITERAL}: [[1, 0, 0], [0, 2, 0], [0, 0, 3]]

func @test() -> () {
    %diagonal = modelica.alloca : !modelica.array<3x!modelica.int>

    %c0 = constant 0 : index
    %0 = modelica.constant #modelica.int<1>
    modelica.store %diagonal[%c0], %0 : !modelica.array<3x!modelica.int>

    %c1 = constant 1 : index
    %1 = modelica.constant #modelica.int<2>
    modelica.store %diagonal[%c1], %1 : !modelica.array<3x!modelica.int>

    %c2 = constant 2 : index
    %2 = modelica.constant #modelica.int<3>
    modelica.store %diagonal[%c2], %2 : !modelica.array<3x!modelica.int>

    %result = modelica.diagonal %diagonal : !modelica.array<3x!modelica.int> -> !modelica.array<3x3x!modelica.int>
    modelica.print %result : !modelica.array<3x3x!modelica.int>

    return
}

func @main() -> () {
    call @test() : () -> ()
    return
}
