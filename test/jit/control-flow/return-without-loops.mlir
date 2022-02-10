// RUN: modelica-opt %s                             \
// RUN:     --convert-modelica-functions            \
// RUN:     --convert-modelica                      \
// RUN:     --convert-modelica-to-cfg               \
// RUN:     --convert-to-llvm                       \
// RUN: | mlir-opt                                  \
// RUN:      --convert-scf-to-std                   \
// RUN: | mlir-cpu-runner                           \
// RUN:     -e main -entry-point-result=void -O0    \
// RUN:     -shared-libs=%runtime_lib               \
// RUN: | FileCheck %s

// CHECK: true
// CHECK-NEXT: false

modelica.function @foo(%arg0 : !modelica.int) -> () attributes {args_names = ["x"], results_names = []} {
    %c0 = modelica.constant #modelica.int<0> : !modelica.int
    %condition = modelica.lt %arg0, %c0 : (!modelica.int, !modelica.int) -> !modelica.bool

    modelica.if (%condition : !modelica.bool) {
        %0 = modelica.constant #modelica.bool<true> : !modelica.bool
        modelica.print %0 : !modelica.bool
        modelica.return
    }

    %0 = modelica.constant #modelica.bool<false> : !modelica.bool
    modelica.print %0 : !modelica.bool

    modelica.function_terminator
}

func @test() -> () {
    %size = constant 2 : index
    %values = modelica.alloca %size : index -> !modelica.array<stack, ?x!modelica.int>

    %c0 = constant 0 : index
    %0 = modelica.constant #modelica.int<-10> : !modelica.int
    modelica.store %values[%c0], %0 : !modelica.array<stack, ?x!modelica.int>

    %c1 = constant 1 : index
    %1 = modelica.constant #modelica.int<10> : !modelica.int
    modelica.store %values[%c1], %1 : !modelica.array<stack, ?x!modelica.int>

    %lb = constant 0 : index
    %step = constant 1 : index

    scf.for %i = %lb to %size step %step {
      %value = modelica.load %values[%i] : !modelica.array<stack, ?x!modelica.int>
      modelica.call @foo(%value) : (!modelica.int) -> ()
    }

    return
}

func @main() -> () {
    call @test() : () -> ()
    return
}
