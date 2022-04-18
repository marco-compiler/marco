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

// CHECK: 2

func @test_staticArray() -> () {
    %array = modelica.alloca : !modelica.array<3x4x!modelica.int>

    %result = modelica.ndims %array : !modelica.array<3x4x!modelica.int> -> !modelica.int
    modelica.print %result : !modelica.int

    return
}

// CHECK-NEXT: 3

func @test_dynamicArray() -> () {
    %0 = constant 4 : index
    %1 = constant 5 : index
    %2 = constant 6 : index
    %array = modelica.alloca %0, %1, %2 : !modelica.array<?x?x?x!modelica.int>

    %result = modelica.ndims %array : !modelica.array<?x?x?x!modelica.int> -> !modelica.int
    modelica.print %result : !modelica.int

    return
}

func @main() -> () {
    call @test_staticArray() : () -> ()
    call @test_dynamicArray() : () -> ()
    return
}
