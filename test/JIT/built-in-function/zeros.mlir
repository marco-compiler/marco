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

// CHECK{LITERAL}: [[0, 0, 0], [0, 0, 0]]
// CHECK-NEXT{LITERAL}: [[0, 0], [0, 0], [0, 0]]

func.func @test() -> () {
    %size = arith.constant 2 : index

    %firstDims = modelica.alloca %size : !modelica.array<?x!modelica.int>
    %secondDims = modelica.alloca %size : !modelica.array<?x!modelica.int>

    %c0 = arith.constant 0 : index
    %firstDim0 = modelica.constant #modelica.int<2>
    %secondDim0 = modelica.constant #modelica.int<3>
    modelica.store %firstDims[%c0], %firstDim0 : !modelica.array<?x!modelica.int>
    modelica.store %secondDims[%c0], %secondDim0 : !modelica.array<?x!modelica.int>

    %c1 = arith.constant 1 : index
    %firstDim1 = modelica.constant #modelica.int<3>
    %secondDim1 = modelica.constant #modelica.int<2>
    modelica.store %firstDims[%c1], %firstDim1 : !modelica.array<?x!modelica.int>
    modelica.store %secondDims[%c1], %secondDim1 : !modelica.array<?x!modelica.int>

    %lb = arith.constant 0 : index
    %step = arith.constant 1 : index

    scf.for %i = %lb to %size step %step {
      %firstDim = modelica.load %firstDims[%i] : !modelica.array<?x!modelica.int>
      %secondDim = modelica.load %secondDims[%i] : !modelica.array<?x!modelica.int>
      %result = modelica.zeros %firstDim, %secondDim : (!modelica.int, !modelica.int) -> !modelica.array<?x?x!modelica.int>
      modelica.print %result : !modelica.array<?x?x!modelica.int>
    }

    return
}

func.func @main() -> () {
    call @test() : () -> ()
    return
}
