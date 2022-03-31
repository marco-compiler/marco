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

// CHECK{LITERAL}: [[1]]
// CHECK-NEXT{LITERAL}: [[1, 0], [0, 1]]
// CHECK-NEXT{LITERAL}: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

func @test() -> () {
    %size = constant 3 : index
    %dimensions = modelica.alloca %size : !modelica.array<?x!modelica.int>

    %c0 = constant 0 : index
    %0 = modelica.constant #modelica.int<1>
    modelica.store %dimensions[%c0], %0 : !modelica.array<?x!modelica.int>

    %c1 = constant 1 : index
    %1 = modelica.constant #modelica.int<2>
    modelica.store %dimensions[%c1], %1 : !modelica.array<?x!modelica.int>

    %c2 = constant 2 : index
    %2 = modelica.constant #modelica.int<3>
    modelica.store %dimensions[%c2], %2 : !modelica.array<?x!modelica.int>

    scf.for %i = %c0 to %size step %c1 {
      %dimension = modelica.load %dimensions[%i] : !modelica.array<?x!modelica.int>
      %result = modelica.identity %dimension : !modelica.int -> !modelica.array<?x?x!modelica.int>
      modelica.print %result : !modelica.array<?x?x!modelica.int>
    }

    return
}

func @main() -> () {
    call @test() : () -> ()
    return
}

