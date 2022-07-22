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

// CHECK: 0
// CHECK-NEXT: 1

modelica.function @foo : (!modelica.int) -> () {
    %x = modelica.member_create @x : !modelica.member<!modelica.int, input>
    %i = modelica.alloca : !modelica.array<!modelica.int>
    %c0 = modelica.constant #modelica.int<0>
    modelica.store %i[], %c0 : !modelica.array<!modelica.int>

    modelica.while {
        %0 = modelica.load %i[] : !modelica.array<!modelica.int>
        %1 = modelica.member_load %x : !modelica.member<!modelica.int, input> -> !modelica.int
        %condition = modelica.lt %0, %1 : (!modelica.int, !modelica.int) -> !modelica.bool
        modelica.condition (%condition : !modelica.bool)
    } do {
        %0 = modelica.load %i[] : !modelica.array<!modelica.int>
        %c1 = modelica.constant #modelica.int<1>
        %1 = modelica.add %0, %c1 : (!modelica.int, !modelica.int) -> !modelica.int
        modelica.store %i[], %1 : !modelica.array<!modelica.int>

        %2 = modelica.constant #modelica.int<0>
        modelica.print %2 : !modelica.int
        modelica.break
    }

    %0 = modelica.constant #modelica.int<1> : !modelica.int
    modelica.print %0 : !modelica.int
}

func.func @test() -> () {
    %size = arith.constant 1 : index
    %values = modelica.alloca %size : !modelica.array<?x!modelica.int>

    %c0 = arith.constant 0 : index
    %0 = modelica.constant #modelica.int<10>
    modelica.store %values[%c0], %0 : !modelica.array<?x!modelica.int>

    %lb = arith.constant 0 : index
    %step = arith.constant 1 : index

    scf.for %i = %lb to %size step %step {
      %value = modelica.load %values[%i] : !modelica.array<?x!modelica.int>
      modelica.call @foo(%value) : (!modelica.int) -> ()
    }

    return
}

func.func @main() -> () {
    call @test() : () -> ()
    return
}
