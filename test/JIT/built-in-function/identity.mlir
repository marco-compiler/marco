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

// CHECK{LITERAL}: [[1]]
// CHECK-NEXT{LITERAL}: [[1, 0], [0, 1]]
// CHECK-NEXT{LITERAL}: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

func.func @test() -> () {
    %size = arith.constant 3 : index
    %dimensions = modelica.alloca %size : !modelica.array<?x!modelica.int>

    %c0 = arith.constant 0 : index
    %0 = modelica.constant #modelica.int<1>
    modelica.store %dimensions[%c0], %0 : !modelica.array<?x!modelica.int>

    %c1 = arith.constant 1 : index
    %1 = modelica.constant #modelica.int<2>
    modelica.store %dimensions[%c1], %1 : !modelica.array<?x!modelica.int>

    %c2 = arith.constant 2 : index
    %2 = modelica.constant #modelica.int<3>
    modelica.store %dimensions[%c2], %2 : !modelica.array<?x!modelica.int>

    %lb = arith.constant 0 : index
    %step = arith.constant 1 : index

    scf.for %i = %lb to %size step %step {
      %dimension = modelica.load %dimensions[%i] : !modelica.array<?x!modelica.int>
      %result = modelica.identity %dimension : !modelica.int -> !modelica.array<?x?x!modelica.int>
      modelica.print %result : !modelica.array<?x?x!modelica.int>
    }

    return
}

func.func @main() -> () {
    call @test() : () -> ()
    return
}

