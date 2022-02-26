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

// CHECK: 0
// CHECK-NEXT: 0
// CHECK-NEXT: 1
// CHECK-NEXT: 2
// CHECK-NEXT: 3
// CHECK-NEXT: 4

modelica.function @foo(%arg0 : !modelica.int) -> () attributes {args_names = ["x"], results_names = []} {
    %i = modelica.alloca : !modelica.array<stack, !modelica.int>
    %c0 = modelica.constant #modelica.int<0> : !modelica.int
    modelica.store %i[], %c0 : !modelica.array<stack, !modelica.int>

    modelica.while {
        %0 = modelica.load %i[] : !modelica.array<stack, !modelica.int>
        %condition = modelica.lt %0, %arg0 : (!modelica.int, !modelica.int) -> !modelica.bool
        modelica.condition (%condition : !modelica.bool)
    } do {
        %0 = modelica.load %i[] : !modelica.array<stack, !modelica.int>
        modelica.print %0 : !modelica.int
        %c1 = modelica.constant #modelica.int<1> : !modelica.int
        %1 = modelica.add %0, %c1 : (!modelica.int, !modelica.int) -> !modelica.int
        modelica.store %i[], %1 : !modelica.array<stack, !modelica.int>
        modelica.yield
    }

    modelica.function_terminator
}

func @test() -> () {
    %size = constant 3 : index
    %values = modelica.alloca %size : index -> !modelica.array<stack, ?x!modelica.int>

    %c0 = constant 0 : index
    %0 = modelica.constant #modelica.int<0> : !modelica.int
    modelica.store %values[%c0], %0 : !modelica.array<stack, ?x!modelica.int>

    %c1 = constant 1 : index
    %1 = modelica.constant #modelica.int<1> : !modelica.int
    modelica.store %values[%c1], %1 : !modelica.array<stack, ?x!modelica.int>

    %c2 = constant 2 : index
    %2 = modelica.constant #modelica.int<5> : !modelica.int
    modelica.store %values[%c2], %2 : !modelica.array<stack, ?x!modelica.int>

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
