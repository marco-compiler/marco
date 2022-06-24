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

// CHECK: 0.000000e+00
// CHECK-NEXT: 1.000000e+00
// CHECK-NEXT: 2.000000e+00
// CHECK-NEXT: -1.000000e+00

func.func @test() -> () {
    %size = arith.constant 4 : index
    %array = modelica.alloca %size : !modelica.array<?x!modelica.real>

    %c0 = arith.constant 0 : index
    %0 = modelica.constant #modelica.real<1.0>
    modelica.store %array[%c0], %0 : !modelica.array<?x!modelica.real>

    %c1 = arith.constant 1 : index
    %1 = modelica.constant #modelica.real<2.718281828>
    modelica.store %array[%c1], %1 : !modelica.array<?x!modelica.real>

    %c2 = arith.constant 2 : index
    %2 = modelica.constant #modelica.real<7.389056099>
    modelica.store %array[%c2], %2 : !modelica.array<?x!modelica.real>

    %c3 = arith.constant 3 : index
    %3 = modelica.constant #modelica.real<0.367879441>
    modelica.store %array[%c3], %3 : !modelica.array<?x!modelica.real>

    %lb = arith.constant 0 : index
    %step = arith.constant 1 : index

    scf.for %i = %lb to %size step %step {
      %value = modelica.load %array[%i] : !modelica.array<?x!modelica.real>
      %result = modelica.log %value : !modelica.real -> !modelica.real
      modelica.print %result : !modelica.real
    }

    return
}

func.func @main() -> () {
    call @test() : () -> ()
    return
}
