// RUN: modelica-opt %s                             \
// RUN:     --convert-modelica-to-cfg               \
// RUN:     --convert-modelica-to-llvm              \
// RUN:     --convert-scf-to-cf                     \
// RUN:     --convert-func-to-llvm                  \
// RUN:     --convert-cf-to-llvm                    \
// RUN:     --reconcile-unrealized-casts            \
// RUN: | mlir-cpu-runner                           \
// RUN:     -e main -entry-point-result=void -O0    \
// RUN:     -shared-libs=%runtime_lib               \
// RUN: | FileCheck %s

// CHECK: true
// CHECK-NEXT: false

modelica.function @foo : (!modelica.int) -> () {
    %x = modelica.member_create @x : !modelica.member<!modelica.int, input>
    %0 = modelica.member_load %x : !modelica.member<!modelica.int, input> -> !modelica.int
    %c0 = modelica.constant #modelica.int<0>
    %condition = modelica.lt %0, %c0 : (!modelica.int, !modelica.int) -> !modelica.bool

    modelica.if (%condition : !modelica.bool) {
        %1 = modelica.constant #modelica.bool<true>
        modelica.print %1 : !modelica.bool
        modelica.return
    }

    %1 = modelica.constant #modelica.bool<false>
    modelica.print %1 : !modelica.bool
}

func.func @test() -> () {
    %size = arith.constant 2 : index
    %values = modelica.alloca %size : !modelica.array<?x!modelica.int>

    %c0 = arith.constant 0 : index
    %0 = modelica.constant #modelica.int<-10>
    modelica.store %values[%c0], %0 : !modelica.array<?x!modelica.int>

    %c1 = arith.constant 1 : index
    %1 = modelica.constant #modelica.int<10>
    modelica.store %values[%c1], %1 : !modelica.array<?x!modelica.int>

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
