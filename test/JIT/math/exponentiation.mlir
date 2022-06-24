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

// CHECK{LITERAL}: 5
// CHECK-NEXT{LITERAL}: 81
// CHECK-NEXT{LITERAL}: 1
// CHECK-NEXT{LITERAL}: 16
// CHECK-NEXT{LITERAL}: 0
// CHECK-NEXT{LITERAL}: -8
// CHECK-NEXT{LITERAL}: 4

func.func @test_scalarBase() -> () {
    %size = arith.constant 7 : index

    %base = modelica.alloca %size : !modelica.array<?x!modelica.int>
    %exp = modelica.alloca %size : !modelica.array<?x!modelica.int>

    %c0 = arith.constant 0 : index
    %b0 = modelica.constant #modelica.int<5>
    %e0 = modelica.constant #modelica.int<1>
    modelica.store %base[%c0], %b0 : !modelica.array<?x!modelica.int>
    modelica.store %exp[%c0], %e0 : !modelica.array<?x!modelica.int>

    %c1 = arith.constant 1 : index
    %b1 = modelica.constant #modelica.int<3>
    %e1 = modelica.constant #modelica.int<4>
    modelica.store %base[%c1], %b1 : !modelica.array<?x!modelica.int>
    modelica.store %exp[%c1], %e1 : !modelica.array<?x!modelica.int>

    %c2 = arith.constant 2 : index
    %b2 = modelica.constant #modelica.int<2>
    %e2 = modelica.constant #modelica.int<0>
    modelica.store %base[%c2], %b2 : !modelica.array<?x!modelica.int>
    modelica.store %exp[%c2], %e2 : !modelica.array<?x!modelica.int>

    %c3 = arith.constant 3 : index
    %b3 = modelica.constant #modelica.int<4>
    %e3 = modelica.constant #modelica.int<2>
    modelica.store %base[%c3], %b3 : !modelica.array<?x!modelica.int>
    modelica.store %exp[%c3], %e3 : !modelica.array<?x!modelica.int>

    %c4 = arith.constant 4 : index
    %b4 = modelica.constant #modelica.int<0>
    %e4 = modelica.constant #modelica.int<3>
    modelica.store %base[%c4], %b4 : !modelica.array<?x!modelica.int>
    modelica.store %exp[%c4], %e4 : !modelica.array<?x!modelica.int>

    %c5 = arith.constant 5 : index
    %b5 = modelica.constant #modelica.int<-2>
    %e5 = modelica.constant #modelica.int<3>
    modelica.store %base[%c5], %b5 : !modelica.array<?x!modelica.int>
    modelica.store %exp[%c5], %e5 : !modelica.array<?x!modelica.int>

    %c6 = arith.constant 6 : index
    %b6 = modelica.constant #modelica.int<-2>
    %e6 = modelica.constant #modelica.int<2>
    modelica.store %base[%c6], %b6 : !modelica.array<?x!modelica.int>
    modelica.store %exp[%c6], %e6 : !modelica.array<?x!modelica.int>

    %lb = arith.constant 0 : index
    %step = arith.constant 1 : index

    scf.for %i = %lb to %size step %step {
      %b = modelica.load %base[%i] : !modelica.array<?x!modelica.int>
      %e = modelica.load %exp[%i] : !modelica.array<?x!modelica.int>
      %result = modelica.pow %b, %e : (!modelica.int, !modelica.int) -> !modelica.int
      modelica.print %result : !modelica.int
    }

    return
}

// CHECK-NEXT{LITERAL}: [[37, 54], [81, 118]]

func.func @test_matrixBase() -> () {
    %size = arith.constant 2 : index
    %base = modelica.alloca %size, %size : !modelica.array<?x?x!modelica.int>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %00 = modelica.constant #modelica.int<1>
    modelica.store %base[%c0, %c0], %00 : !modelica.array<?x?x!modelica.int>

    %01 = modelica.constant #modelica.int<2>
    modelica.store %base[%c0, %c1], %01 : !modelica.array<?x?x!modelica.int>

    %10 = modelica.constant #modelica.int<3>
    modelica.store %base[%c1, %c0], %10 : !modelica.array<?x?x!modelica.int>

    %11 = modelica.constant #modelica.int<4>
    modelica.store %base[%c1, %c1], %11 : !modelica.array<?x?x!modelica.int>

    %exp = modelica.constant #modelica.int<3>

    %result = modelica.pow %base, %exp : (!modelica.array<?x?x!modelica.int>, !modelica.int) -> !modelica.array<?x?x!modelica.int>
    modelica.print %result : !modelica.array<?x?x!modelica.int>

    return
}

func.func @main() -> () {
    call @test_scalarBase() : () -> ()
    call @test_matrixBase() : () -> ()

    return
}
