// RUN: modelica-opt %s                             \
// RUN:     --convert-modelica-to-cf                \
// RUN:     --convert-modelica-to-arith             \
// RUN:     --convert-modelica-to-llvm              \
// RUN:     --convert-scf-to-cf                     \
// RUN:     --convert-func-to-llvm                  \
// RUN:     --convert-cf-to-llvm                    \
// RUN:     --reconcile-unrealized-casts            \
// RUN: | mlir-cpu-runner                           \
// RUN:     -e main -entry-point-result=void -O0    \
// RUN:     -shared-libs=%runtime_lib               \
// RUN: | FileCheck %s

// CHECK{LITERAL}: true
// CHECK-NEXT{LITERAL}: false

func.func @test_scalars() -> () {
    %false = modelica.constant #modelica.bool<false>
    %notFalse = modelica.not %false : !modelica.bool -> !modelica.bool
    modelica.print %notFalse : !modelica.bool

    %true = modelica.constant #modelica.bool<true>
    %notTrue = modelica.not %true : !modelica.bool -> !modelica.bool
    modelica.print %notTrue : !modelica.bool

    return
}

// CHECK-NEXT{LITERAL}: [true, false]

func.func @test_staticArray() -> () {
    %array = modelica.alloca : !modelica.array<2x!modelica.bool>

    %c0 = arith.constant 0 : index
    %0 = modelica.constant #modelica.bool<false>
    modelica.store %array[%c0], %0 : !modelica.array<2x!modelica.bool>

    %c1 = arith.constant 1 : index
    %1 = modelica.constant #modelica.bool<true>
    modelica.store %array[%c1], %1 : !modelica.array<2x!modelica.bool>

    %result = modelica.not %array : !modelica.array<2x!modelica.bool> -> !modelica.array<2x!modelica.bool>
    modelica.print %result : !modelica.array<2x!modelica.bool>

    return
}

// CHECK-NEXT{LITERAL}: [true, false]

func.func @test_dynamicArray() -> () {
    %c2 = arith.constant 2 : index
    %array = modelica.alloc %c2 : !modelica.array<?x!modelica.bool>

    %c0 = arith.constant 0 : index
    %0 = modelica.constant #modelica.bool<false>
    modelica.store %array[%c0], %0 : !modelica.array<?x!modelica.bool>

    %c1 = arith.constant 1 : index
    %1 = modelica.constant #modelica.bool<true>
    modelica.store %array[%c1], %1 : !modelica.array<?x!modelica.bool>

    %result = modelica.not %array : !modelica.array<?x!modelica.bool> -> !modelica.array<?x!modelica.bool>
    modelica.print %result : !modelica.array<?x!modelica.bool>

    return
}

func.func @main() -> () {
    call @test_scalars() : () -> ()
    call @test_staticArray() : () -> ()
    call @test_dynamicArray() : () -> ()

    return
}
