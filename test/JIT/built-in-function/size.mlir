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

// CHECK{LITERAL}: [2, 3]

func @test_staticArray() -> () {
    %array = modelica.alloca : !modelica.array<2x3x!modelica.int>

    %result = modelica.size %array : !modelica.array<2x3x!modelica.int> -> !modelica.array<2x!modelica.int>
    modelica.print %result : !modelica.array<2x!modelica.int>

    return
}

// CHECK-NEXT: 2
// CHECK-NEXT: 3

func @test_staticArrayDimension() -> () {
    %array = modelica.alloca : !modelica.array<2x3x!modelica.int>

    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %dims = constant 2 : index

    scf.for %i = %c0 to %dims step %c1 {
      %result = modelica.size %array, %i : (!modelica.array<2x3x!modelica.int>, index) -> !modelica.int
      modelica.print %result : !modelica.int
    }

    return
}

// CHECK-NEXT{LITERAL}: [2, 3]

func @test_dynamicArray() -> () {
    %0 = constant 2 : index
    %1 = constant 3 : index

    %array = modelica.alloca %0, %1 : !modelica.array<?x?x!modelica.int>

    %result = modelica.size %array : !modelica.array<?x?x!modelica.int> -> !modelica.array<2x!modelica.int>
    modelica.print %result : !modelica.array<2x!modelica.int>

    return
}

// CHECK-NEXT: 2
// CHECK-NEXT: 3

func @test_dynamicArrayDimension() -> () {
    %0 = constant 2 : index
    %1 = constant 3 : index
    %array = modelica.alloca %0, %1 : !modelica.array<?x?x!modelica.int>

    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %dims = constant 2 : index

    scf.for %i = %c0 to %dims step %c1 {
      %result = modelica.size %array, %i : (!modelica.array<?x?x!modelica.int>, index) -> !modelica.int
      modelica.print %result : !modelica.int
    }

    return
}

func @main() -> () {
    call @test_staticArray() : () -> ()
    call @test_staticArrayDimension() : () -> ()
    call @test_dynamicArray() : () -> ()
    call @test_dynamicArrayDimension() : () -> ()
    return
}
