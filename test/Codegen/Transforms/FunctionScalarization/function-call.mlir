// RUN: modelica-opt %s                             \
// RUN:     --scalarize                             \
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

// CHECK{LITERAL}: [0.000000e+00, 1.000000e+00, 2.000000e+00]

modelica.function @callee : (!modelica.real) -> (!modelica.real) {
    %cst = arith.constant 3 : index
    %3 = modelica.member_create @x : !modelica.member<!modelica.real, input>
    %1 = modelica.member_create @y : !modelica.member<!modelica.real, output>
    %2 = modelica.member_load %3 : !modelica.member<!modelica.real, input> -> !modelica.real
    modelica.member_store %1, %2 : !modelica.member<!modelica.real, output>, !modelica.real
}

func.func @caller() -> () {
    %array = modelica.alloca : !modelica.array<3x!modelica.real>

    %c0 = modelica.constant 0 : index
    %0 = modelica.constant #modelica.real<0.0> : !modelica.real
    modelica.store %array[%c0], %0 : !modelica.array<3x!modelica.real>

    %c1 = modelica.constant 1 : index
    %1 = modelica.constant #modelica.real<1.0> : !modelica.real
    modelica.store %array[%c1], %1 : !modelica.array<3x!modelica.real>

    %c2 = modelica.constant 2 : index
    %2 = modelica.constant #modelica.real<2.0> : !modelica.real
    modelica.store %array[%c2], %2 : !modelica.array<3x!modelica.real>

    %result = modelica.call @callee(%array) : (!modelica.array<3x!modelica.real>) -> (!modelica.array<3x!modelica.real>)
    modelica.print %result : !modelica.array<3x!modelica.real>
    return
}

func.func @main() -> () {
    call @caller() : () -> ()
    return
}
