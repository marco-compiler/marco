// RUN: modelica-opt %s                             \
// RUN:     --convert-modelica-to-cfg               \
// RUN:     --convert-modelica-to-llvm              \
// RUN:     --convert-scf-to-cf                     \
// RUN:     --convert-func-to-llvm                  \
// RUN:     --convert-cf-to-llvm                    \
// RUN:     --reconcile-unrealized-casts            \
// RUN: | mlir-cpu-runner                           \
// RUN:     -e main -entry-point-result=void -O0    \
// RUN:     -shared-libs=%runtime_lib               \
// RUN: | FileCheck %s

// CHECK: 2

func.func @test_staticArray() -> () {
    %array = modelica.alloca : !modelica.array<3x4x!modelica.int>

    %result = modelica.ndims %array : !modelica.array<3x4x!modelica.int> -> !modelica.int
    modelica.print %result : !modelica.int

    return
}

// CHECK-NEXT: 3

func.func @test_dynamicArray() -> () {
    %0 = arith.constant 4 : index
    %1 = arith.constant 5 : index
    %2 = arith.constant 6 : index
    %array = modelica.alloca %0, %1, %2 : !modelica.array<?x?x?x!modelica.int>

    %result = modelica.ndims %array : !modelica.array<?x?x?x!modelica.int> -> !modelica.int
    modelica.print %result : !modelica.int

    return
}

func.func @main() -> () {
    call @test_staticArray() : () -> ()
    call @test_dynamicArray() : () -> ()
    return
}
