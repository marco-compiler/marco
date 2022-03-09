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

// CHECK{LITERAL}: [[0, 0, 0], [0, 0, 0]]
// CHECK-NEXT{LITERAL}: [[0, 0], [0, 0], [0, 0]]

func @test() -> () {
    %size = constant 2 : index

    %firstDims = modelica.alloca %size : index -> !modelica.array<stack, ?x!modelica.int>
    %secondDims = modelica.alloca %size : index -> !modelica.array<stack, ?x!modelica.int>

    %c0 = constant 0 : index
    %firstDim0 = modelica.constant #modelica.int<2> : !modelica.int
    %secondDim0 = modelica.constant #modelica.int<3> : !modelica.int
    modelica.store %firstDims[%c0], %firstDim0 : !modelica.array<stack, ?x!modelica.int>
    modelica.store %secondDims[%c0], %secondDim0 : !modelica.array<stack, ?x!modelica.int>

    %c1 = constant 1 : index
    %firstDim1 = modelica.constant #modelica.int<3> : !modelica.int
    %secondDim1 = modelica.constant #modelica.int<2> : !modelica.int
    modelica.store %firstDims[%c1], %firstDim1 : !modelica.array<stack, ?x!modelica.int>
    modelica.store %secondDims[%c1], %secondDim1 : !modelica.array<stack, ?x!modelica.int>

    scf.for %i = %c0 to %size step %c1 {
      %firstDim = modelica.load %firstDims[%i] : !modelica.array<stack, ?x!modelica.int>
      %secondDim = modelica.load %secondDims[%i] : !modelica.array<stack, ?x!modelica.int>
      %result = modelica.zeros %firstDim, %secondDim : (!modelica.int, !modelica.int) -> !modelica.array<stack, ?x?x!modelica.int>
      modelica.print %result : !modelica.array<stack, ?x?x!modelica.int>
    }

    return
}

func @main() -> () {
    call @test() : () -> ()
    return
}
