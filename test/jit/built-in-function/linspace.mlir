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

// CHECK{LITERAL}: [-1.000000e+00, -9.000000e-01, -8.000000e-01, -7.000000e-01, -6.000000e-01, -5.000000e-01, -4.000000e-01, -3.000000e-01, -2.000000e-01, -1.000000e-01, 0.000000e+00]
// CHECK-NEXT{LITERAL}: [0.000000e+00, 1.000000e-01, 2.000000e-01, 3.000000e-01, 4.000000e-01, 5.000000e-01, 6.000000e-01, 7.000000e-01, 8.000000e-01, 9.000000e-01, 1.000000e+00]

func @test() -> () {
    %size = constant 2 : index

    %begins = modelica.alloca %size : index -> !modelica.array<stack, ?x!modelica.int>
    %ends = modelica.alloca %size : index -> !modelica.array<stack, ?x!modelica.int>
    %stepsCounts = modelica.alloca %size : index -> !modelica.array<stack, ?x!modelica.int>

    %c0 = constant 0 : index
    %begin0 = modelica.constant #modelica.int<-1> : !modelica.int
    %end0 = modelica.constant #modelica.int<0> : !modelica.int
    %stepsCount0 = modelica.constant #modelica.int<11> : !modelica.int
    modelica.store %begins[%c0], %begin0 : !modelica.array<stack, ?x!modelica.int>
    modelica.store %ends[%c0], %end0 : !modelica.array<stack, ?x!modelica.int>
    modelica.store %stepsCounts[%c0], %stepsCount0 : !modelica.array<stack, ?x!modelica.int>

    %c1 = constant 1 : index
    %begin1 = modelica.constant #modelica.int<0> : !modelica.int
    %end1 = modelica.constant #modelica.int<1> : !modelica.int
    %stepsCount1 = modelica.constant #modelica.int<11> : !modelica.int
    modelica.store %begins[%c1], %begin1 : !modelica.array<stack, ?x!modelica.int>
    modelica.store %ends[%c1], %end1 : !modelica.array<stack, ?x!modelica.int>
    modelica.store %stepsCounts[%c1], %stepsCount1 : !modelica.array<stack, ?x!modelica.int>

    scf.for %i = %c0 to %size step %c1 {
      %begin = modelica.load %begins[%i] : !modelica.array<stack, ?x!modelica.int>
      %end = modelica.load %ends[%i] : !modelica.array<stack, ?x!modelica.int>
      %stepsCount = modelica.load %stepsCounts[%i] : !modelica.array<stack, ?x!modelica.int>

      %result = modelica.linspace %begin, %end, %stepsCount : (!modelica.int, !modelica.int, !modelica.int) -> !modelica.array<stack, ?x!modelica.real>
      modelica.print %result : !modelica.array<stack, ?x!modelica.real>
    }

    return
}

func @main() -> () {
    call @test() : () -> ()
    return
}
