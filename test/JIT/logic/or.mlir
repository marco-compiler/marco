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

// CHECK{LITERAL}: false
// CHECK-NEXT{LITERAL}: true
// CHECK-NEXT{LITERAL}: true
// CHECK-NEXT{LITERAL}: true

func.func @test_scalars() -> () {
    %size = arith.constant 4 : index

    %x = modelica.alloca %size : !modelica.array<?x!modelica.bool>
    %y = modelica.alloca %size : !modelica.array<?x!modelica.bool>

    %c0 = arith.constant 0 : index
    %x0 = modelica.constant #modelica.bool<false>
    %y0 = modelica.constant #modelica.bool<false>
    modelica.store %x[%c0], %x0 : !modelica.array<?x!modelica.bool>
    modelica.store %y[%c0], %y0 : !modelica.array<?x!modelica.bool>

    %c1 = arith.constant 1 : index
    %x1 = modelica.constant #modelica.bool<false>
    %y1 = modelica.constant #modelica.bool<true>
    modelica.store %x[%c1], %x1 : !modelica.array<?x!modelica.bool>
    modelica.store %y[%c1], %y1 : !modelica.array<?x!modelica.bool>

    %c2 = arith.constant 2 : index
    %x2 = modelica.constant #modelica.bool<true>
    %y2 = modelica.constant #modelica.bool<false>
    modelica.store %x[%c2], %x2 : !modelica.array<?x!modelica.bool>
    modelica.store %y[%c2], %y2 : !modelica.array<?x!modelica.bool>

    %c3 = arith.constant 3 : index
    %x3 = modelica.constant #modelica.bool<true>
    %y3 = modelica.constant #modelica.bool<true>
    modelica.store %x[%c3], %x3 : !modelica.array<?x!modelica.bool>
    modelica.store %y[%c3], %y3 : !modelica.array<?x!modelica.bool>

    %lb = arith.constant 0 : index
    %step = arith.constant 1 : index

    scf.for %i = %lb to %size step %step {
      %xi = modelica.load %x[%i] : !modelica.array<?x!modelica.bool>
      %yi = modelica.load %y[%i] : !modelica.array<?x!modelica.bool>
      %result = modelica.or %xi, %yi : (!modelica.bool, !modelica.bool) -> !modelica.bool
      modelica.print %result : !modelica.bool
    }

    return
}

// CHECK-NEXT{LITERAL}: [false, true, true, true]

func.func @test_staticArrays() -> () {
    %x = modelica.alloca : !modelica.array<4x!modelica.bool>
    %y = modelica.alloca : !modelica.array<4x!modelica.bool>

    %c0 = arith.constant 0 : index
    %x0 = modelica.constant #modelica.bool<false>
    %y0 = modelica.constant #modelica.bool<false>
    modelica.store %x[%c0], %x0 : !modelica.array<4x!modelica.bool>
    modelica.store %y[%c0], %y0 : !modelica.array<4x!modelica.bool>

    %c1 = arith.constant 1 : index
    %x1 = modelica.constant #modelica.bool<false>
    %y1 = modelica.constant #modelica.bool<true>
    modelica.store %x[%c1], %x1 : !modelica.array<4x!modelica.bool>
    modelica.store %y[%c1], %y1 : !modelica.array<4x!modelica.bool>

    %c2 = arith.constant 2 : index
    %x2 = modelica.constant #modelica.bool<true>
    %y2 = modelica.constant #modelica.bool<false>
    modelica.store %x[%c2], %x2 : !modelica.array<4x!modelica.bool>
    modelica.store %y[%c2], %y2 : !modelica.array<4x!modelica.bool>

    %c3 = arith.constant 3 : index
    %x3 = modelica.constant #modelica.bool<true>
    %y3 = modelica.constant #modelica.bool<true>
    modelica.store %x[%c3], %x3 : !modelica.array<4x!modelica.bool>
    modelica.store %y[%c3], %y3 : !modelica.array<4x!modelica.bool>

    %result = modelica.or %x, %y : (!modelica.array<4x!modelica.bool>, !modelica.array<4x!modelica.bool>) -> !modelica.array<4x!modelica.bool>
    modelica.print %result : !modelica.array<4x!modelica.bool>

    return
}

// CHECK-NEXT{LITERAL}: [false, true, true, true]

func.func @test_dynamicArrays() -> () {
    %size = arith.constant 4 : index

    %x = modelica.alloc %size : !modelica.array<?x!modelica.bool>
    %y = modelica.alloc %size : !modelica.array<?x!modelica.bool>

    %c0 = arith.constant 0 : index
    %x0 = modelica.constant #modelica.bool<false>
    %y0 = modelica.constant #modelica.bool<false>
    modelica.store %x[%c0], %x0 : !modelica.array<?x!modelica.bool>
    modelica.store %y[%c0], %y0 : !modelica.array<?x!modelica.bool>

    %c1 = arith.constant 1 : index
    %x1 = modelica.constant #modelica.bool<false>
    %y1 = modelica.constant #modelica.bool<true>
    modelica.store %x[%c1], %x1 : !modelica.array<?x!modelica.bool>
    modelica.store %y[%c1], %y1 : !modelica.array<?x!modelica.bool>

    %c2 = arith.constant 2 : index
    %x2 = modelica.constant #modelica.bool<true>
    %y2 = modelica.constant #modelica.bool<false>
    modelica.store %x[%c2], %x2 : !modelica.array<?x!modelica.bool>
    modelica.store %y[%c2], %y2 : !modelica.array<?x!modelica.bool>

    %c3 = arith.constant 3 : index
    %x3 = modelica.constant #modelica.bool<true>
    %y3 = modelica.constant #modelica.bool<true>
    modelica.store %x[%c3], %x3 : !modelica.array<?x!modelica.bool>
    modelica.store %y[%c3], %y3 : !modelica.array<?x!modelica.bool>

    %result = modelica.or %x, %y : (!modelica.array<?x!modelica.bool>, !modelica.array<?x!modelica.bool>) -> !modelica.array<?x!modelica.bool>
    modelica.free %x : !modelica.array<?x!modelica.bool>
    modelica.free %y : !modelica.array<?x!modelica.bool>
    modelica.print %result : !modelica.array<?x!modelica.bool>
    modelica.free %result : !modelica.array<?x!modelica.bool>

    return
}

func.func @main() -> () {
    call @test_scalars() : () -> ()
    call @test_staticArrays() : () -> ()
    call @test_dynamicArrays() : () -> ()

    return
}
