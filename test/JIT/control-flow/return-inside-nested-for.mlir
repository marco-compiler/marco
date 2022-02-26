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
// CHECK-NEXT: 0
// CHECK-NEXT: 0
// CHECK-NEXT: 0
// CHECK-NEXT: 0
// CHECK-NEXT: 2

modelica.function @foo(%arg0 : !modelica.int) -> () attributes {args_names = ["x"], results_names = []} {
    %c0 = modelica.constant #modelica.int<0> : !modelica.int

    %i = modelica.alloca : !modelica.array<stack, !modelica.int>
    modelica.store %i[], %c0 : !modelica.array<stack, !modelica.int>

    %j = modelica.alloca : !modelica.array<stack, !modelica.int>
    modelica.store %j[], %c0 : !modelica.array<stack, !modelica.int>

    modelica.for condition {
        %0 = modelica.load %i[] : !modelica.array<stack, !modelica.int>
        %c2 = modelica.constant #modelica.int<2> : !modelica.int
        %1 = modelica.mul %arg0, %c2 : (!modelica.int, !modelica.int) -> !modelica.int
        %condition = modelica.lt %0, %1 : (!modelica.int, !modelica.int) -> !modelica.bool
        modelica.condition (%condition : !modelica.bool)
    } body {
        modelica.for condition {
            %0 = modelica.load %j[] : !modelica.array<stack, !modelica.int>
            %condition = modelica.lt %0, %arg0 : (!modelica.int, !modelica.int) -> !modelica.bool
            modelica.condition (%condition : !modelica.bool)
        } body {
            %0 = modelica.constant #modelica.int<0> : !modelica.int
            modelica.print %0 : !modelica.int

            %1 = modelica.load %j[] : !modelica.array<stack, !modelica.int>
            %c3 = modelica.constant #modelica.int<3> : !modelica.int
            %condition = modelica.gte %1, %c3 : (!modelica.int, !modelica.int) -> !modelica.bool

            modelica.if (%condition : !modelica.bool) {
                %2 = modelica.constant #modelica.int<2> : !modelica.int
                modelica.print %2 : !modelica.int
                modelica.return
            }

            modelica.yield
        } step {
            %0 = modelica.load %j[] : !modelica.array<stack, !modelica.int>
            %c1 = modelica.constant #modelica.int<1> : !modelica.int
            %1 = modelica.add %0, %c1 : (!modelica.int, !modelica.int) -> !modelica.int
            modelica.store %j[], %1 : !modelica.array<stack, !modelica.int>
            modelica.yield
        }

        modelica.yield
    } step {
        %0 = modelica.load %i[] : !modelica.array<stack, !modelica.int>
        %c1 = modelica.constant #modelica.int<1> : !modelica.int
        %1 = modelica.add %0, %c1 : (!modelica.int, !modelica.int) -> !modelica.int
        modelica.store %i[], %1 : !modelica.array<stack, !modelica.int>
        modelica.yield
    }

    %0 = modelica.constant #modelica.int<1> : !modelica.int
    modelica.print %0 : !modelica.int

    modelica.function_terminator
}

func @test() -> () {
    %size = constant 2 : index
    %values = modelica.alloca %size : index -> !modelica.array<stack, ?x!modelica.int>

    %c0 = constant 0 : index
    %0 = modelica.constant #modelica.int<2> : !modelica.int
    modelica.store %values[%c0], %0 : !modelica.array<stack, ?x!modelica.int>

    %c1 = constant 1 : index
    %1 = modelica.constant #modelica.int<4> : !modelica.int
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
