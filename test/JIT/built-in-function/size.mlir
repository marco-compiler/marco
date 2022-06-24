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

// CHECK{LITERAL}: [2, 3]

func.func @test_staticArray() -> () {
    %array = modelica.alloca : !modelica.array<2x3x!modelica.int>

    %result = modelica.size %array : !modelica.array<2x3x!modelica.int> -> !modelica.array<2x!modelica.int>
    modelica.print %result : !modelica.array<2x!modelica.int>

    return
}

// CHECK-NEXT: 2
// CHECK-NEXT: 3

func.func @test_staticArrayDimension() -> () {
    %array = modelica.alloca : !modelica.array<2x3x!modelica.int>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dims = arith.constant 2 : index

    scf.for %i = %c0 to %dims step %c1 {
      %result = modelica.size %array, %i : (!modelica.array<2x3x!modelica.int>, index) -> !modelica.int
      modelica.print %result : !modelica.int
    }

    return
}

// CHECK-NEXT{LITERAL}: [2, 3]

func.func @test_dynamicArray() -> () {
    %0 = arith.constant 2 : index
    %1 = arith.constant 3 : index

    %array = modelica.alloca %0, %1 : !modelica.array<?x?x!modelica.int>

    %result = modelica.size %array : !modelica.array<?x?x!modelica.int> -> !modelica.array<2x!modelica.int>
    modelica.print %result : !modelica.array<2x!modelica.int>

    return
}

// CHECK-NEXT: 2
// CHECK-NEXT: 3

func.func @test_dynamicArrayDimension() -> () {
    %0 = arith.constant 2 : index
    %1 = arith.constant 3 : index
    %array = modelica.alloca %0, %1 : !modelica.array<?x?x!modelica.int>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dims = arith.constant 2 : index

    scf.for %i = %c0 to %dims step %c1 {
      %result = modelica.size %array, %i : (!modelica.array<?x?x!modelica.int>, index) -> !modelica.int
      modelica.print %result : !modelica.int
    }

    return
}

func.func @main() -> () {
    call @test_staticArray() : () -> ()
    call @test_staticArrayDimension() : () -> ()
    call @test_dynamicArray() : () -> ()
    call @test_dynamicArrayDimension() : () -> ()
    return
}
