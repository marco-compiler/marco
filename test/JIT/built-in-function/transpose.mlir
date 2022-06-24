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

// CHECK{LITERAL}: [[1, 3, 5], [2, 4, 6]]

func.func @test() -> () {
    %matrix = modelica.alloca : !modelica.array<3x2x!modelica.int>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    %00 = modelica.constant #modelica.int<1>
    modelica.store %matrix[%c0, %c0], %00 : !modelica.array<3x2x!modelica.int>

    %01 = modelica.constant #modelica.int<2>
    modelica.store %matrix[%c0, %c1], %01 : !modelica.array<3x2x!modelica.int>

    %10 = modelica.constant #modelica.int<3>
    modelica.store %matrix[%c1, %c0], %10 : !modelica.array<3x2x!modelica.int>

    %11 = modelica.constant #modelica.int<4>
    modelica.store %matrix[%c1, %c1], %11 : !modelica.array<3x2x!modelica.int>

    %20 = modelica.constant #modelica.int<5>
    modelica.store %matrix[%c2, %c0], %20 : !modelica.array<3x2x!modelica.int>

    %21 = modelica.constant #modelica.int<6>
    modelica.store %matrix[%c2, %c1], %21 : !modelica.array<3x2x!modelica.int>

    %result = modelica.transpose %matrix : !modelica.array<3x2x!modelica.int> -> !modelica.array<2x3x!modelica.int>
    modelica.print %result : !modelica.array<2x3x!modelica.int>

    return
}

func.func @main() -> () {
    call @test() : () -> ()
    return
}
