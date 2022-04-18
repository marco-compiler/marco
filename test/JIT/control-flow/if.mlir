// RUN: modelica-opt %s                             \
// RUN:     --convert-modelica                      \
// RUN:     --convert-modelica-to-cfg               \
// RUN:     --convert-to-llvm                       \
// RUN:     --remove-unrealized-casts               \
// RUN: | mlir-opt                                  \
// RUN:      --convert-scf-to-std                   \
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
    } else {
        %1 = modelica.constant #modelica.bool<false>
        modelica.print %1 : !modelica.bool
    }
}

func @test() -> () {
    %size = constant 2 : index
    %values = modelica.alloca %size : !modelica.array<?x!modelica.int>

    %c0 = constant 0 : index
    %0 = modelica.constant #modelica.int<-10>
    modelica.store %values[%c0], %0 : !modelica.array<?x!modelica.int>

    %c1 = constant 1 : index
    %1 = modelica.constant #modelica.int<10>
    modelica.store %values[%c1], %1 : !modelica.array<?x!modelica.int>

    %lb = constant 0 : index
    %step = constant 1 : index

    scf.for %i = %lb to %size step %step {
      %value = modelica.load %values[%i] : !modelica.array<?x!modelica.int>
      modelica.call @foo(%value) : (!modelica.int) -> ()
    }

    return
}

func @main() -> () {
    call @test() : () -> ()
    return
}
