// RUN: modelica-opt %s                             \
// RUN:     --auto-diff                             \
// RUN:     --convert-modelica-functions            \
// RUN:     --convert-modelica                      \
// RUN:     --convert-to-llvm                       \
// RUN: | mlir-cpu-runner                           \
// RUN:     -e main -entry-point-result=void -O0    \
// RUN:     -shared-libs=%runtime_lib               \
// RUN: | FileCheck %s

// CHECK: -1.979985e+00

modelica.function @sin_der(%arg0 : !modelica.real, %arg1 : !modelica.real) -> (!modelica.real) attributes {args_names = ["x", "der_x"], results_names = ["y"]} {
    %0 = modelica.member_create {name = "y"} : !modelica.member<stack, !modelica.real>
    %1 = modelica.sin %arg0 : !modelica.real -> !modelica.real
    %2 = modelica.der %1 : !modelica.real -> !modelica.real
    modelica.member_store %0, %2 : !modelica.member<stack, !modelica.real>, !modelica.real
    modelica.function_terminator
}

func @test_sin() -> () {
    %x = modelica.constant #modelica.real<3.0> : !modelica.real
    %der_x = modelica.constant #modelica.real<2.0> : !modelica.real
    %result = modelica.call @sin_der(%x, %der_x) : (!modelica.real, !modelica.real) -> (!modelica.real)
    modelica.print %result : !modelica.real
    return
}

// CHECK: 2.822400e-01

modelica.function @cos_der(%arg0 : !modelica.real, %arg1 : !modelica.real) -> (!modelica.real) attributes {args_names = ["x", "der_x"], results_names = ["y"]} {
    %0 = modelica.member_create {name = "y"} : !modelica.member<stack, !modelica.real>
    %1 = modelica.cos %arg0 : !modelica.real -> !modelica.real
    %2 = modelica.der %1 : !modelica.real -> !modelica.real
    modelica.member_store %0, %2 : !modelica.member<stack, !modelica.real>, !modelica.real
    modelica.function_terminator
}

func @test_cos() -> () {
    %x = modelica.constant #modelica.real<3.0> : !modelica.real
    %der_x = modelica.constant #modelica.real<2.0> : !modelica.real
    %result = modelica.call @cos_der(%x, %der_x) : (!modelica.real, !modelica.real) -> (!modelica.real)
    modelica.print %result : !modelica.real
    return
}

func @main() -> () {
    call @test_sin() : () -> ()
    call @test_cos() : () -> ()
    return
}
