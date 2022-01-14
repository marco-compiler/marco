// RUN: modelica-opt %s                             \
// RUN:     --vectorize-functions                   \
// RUN:     --convert-modelica-functions            \
// RUN:     --convert-modelica                      \
// RUN:     --convert-modelica-to-cfg               \
// RUN:     --convert-to-llvm                       \
// RUN: | mlir-cpu-runner                           \
// RUN:     -e main -entry-point-result=void -O0    \
// RUN:     -shared-libs=%runtime_lib               \
// RUN: | FileCheck %s

// CHECK{LITERAL}: [0.000000e+00, 1.000000e+00, 2.000000e+00]

modelica.function @callee(%arg0 : !modelica.real) -> (!modelica.real) attributes {args_names = ["x"], results_names = ["y"]} {
    %0 = modelica.member_create {name = "y"} : !modelica.member<stack, !modelica.real>
    modelica.member_store %0, %arg0 : !modelica.member<stack, !modelica.real>
    modelica.function_terminator
}

func @caller() -> () {
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

func @main() -> () {
    call @caller() : () -> ()
    return
}
