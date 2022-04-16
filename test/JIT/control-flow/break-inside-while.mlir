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

// CHECK: 0
// CHECK-NEXT: 0
// CHECK-NEXT: 1
// CHECK-NEXT: 0
// CHECK-NEXT: 0
// CHECK-NEXT: 0
// CHECK-NEXT: 1

modelica.function @foo : (!modelica.int) -> () {
    %x = modelica.member_create {name = "x"} : !modelica.member<!modelica.int, input>
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
        %c3 = modelica.constant #modelica.int<3>
        %condition = modelica.gte %0, %c3 : (!modelica.int, !modelica.int) -> !modelica.bool

        modelica.if (%condition : !modelica.bool) {
            modelica.break
        }

        modelica.print %c0 : !modelica.int

        %1 = modelica.load %i[] : !modelica.array<!modelica.int>
        %c1 = modelica.constant #modelica.int<1>
        %2 = modelica.add %1, %c1 : (!modelica.int, !modelica.int) -> !modelica.int
        modelica.store %i[], %2 : !modelica.array<!modelica.int>
        modelica.yield
    }

    %0 = modelica.constant #modelica.int<1>
    modelica.print %0 : !modelica.int
}

func @test() -> () {
    %size = constant 2 : index
    %values = modelica.alloca %size : !modelica.array<?x!modelica.int>

    %c0 = constant 0 : index
    %0 = modelica.constant #modelica.int<2>
    modelica.store %values[%c0], %0 : !modelica.array<?x!modelica.int>

    %c1 = constant 1 : index
    %1 = modelica.constant #modelica.int<4>
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