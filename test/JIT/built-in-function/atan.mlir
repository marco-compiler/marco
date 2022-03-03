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

// CHECK: 7.853982e-01
// CHECK-NEXT: 5.235988e-01
// CHECK-NEXT: 0.000000e+00
// CHECK-NEXT: -5.235988e-01
// CHECK-NEXT: -7.853982e-01

func @test() -> () {
    %size = constant 5 : index
    %array = modelica.alloca %size : index -> !modelica.array<stack, ?x!modelica.real>

    %c0 = constant 0 : index
    %0 = modelica.constant #modelica.real<1.0> : !modelica.real
    modelica.store %array[%c0], %0 : !modelica.array<stack, ?x!modelica.real>

    %c1 = constant 1 : index
    %1 = modelica.constant #modelica.real<0.577350269> : !modelica.real
    modelica.store %array[%c1], %1 : !modelica.array<stack, ?x!modelica.real>

    %c2 = constant 2 : index
    %2 = modelica.constant #modelica.real<0.0> : !modelica.real
    modelica.store %array[%c2], %2 : !modelica.array<stack, ?x!modelica.real>

    %c3 = constant 3 : index
    %3 = modelica.constant #modelica.real<-0.577350269> : !modelica.real
    modelica.store %array[%c3], %3 : !modelica.array<stack, ?x!modelica.real>

    %c4 = constant 4 : index
    %4 = modelica.constant #modelica.real<-1.0> : !modelica.real
    modelica.store %array[%c4], %4 : !modelica.array<stack, ?x!modelica.real>

    scf.for %i = %c0 to %size step %c1 {
      %value = modelica.load %array[%i] : !modelica.array<stack, ?x!modelica.real>
      %result = modelica.atan %value : !modelica.real -> !modelica.real
      modelica.print %result : !modelica.real
    }

    return
}

func @main() -> () {
    call @test() : () -> ()
    return
}