// RUN: modelica-opt %s                             \
// RUN:     --auto-diff                             \
// RUN:     --convert-modelica-functions            \
// RUN:     --convert-modelica                      \
// RUN:     --convert-to-llvm                       \
// RUN: | mlir-cpu-runner                           \
// RUN:     -e main -entry-point-result=void -O0    \
// RUN:     -shared-libs=%runtime_lib               \
// RUN: | FileCheck %s

// d/dt (x) = d/dt (x)
// CHECK: 2.000000e+00

modelica.function @var(%arg0 : !modelica.real) -> (!modelica.real) attributes {args_names = ["x"], results_names = ["y"], derivative = #modelica.derivative<"var_der", 1>} {
    %0 = modelica.member_create {name = "y"} : !modelica.member<stack, !modelica.real>
    %1 = modelica.neg %arg0 : !modelica.real -> !modelica.real
    modelica.member_store %0, %1 : !modelica.member<stack, !modelica.real>, !modelica.real
    modelica.function_terminator
}

func @test_var() -> () {
    %x = modelica.constant #modelica.real<57.0> : !modelica.real
    %der_x = modelica.constant #modelica.real<2.0> : !modelica.real
    %result = modelica.call @var_der(%x, %der_x) : (!modelica.real, !modelica.real) -> (!modelica.real)
    modelica.print %result : !modelica.real
    return
}

// d/dt (-x) = - d/dt (x)
// CHECK: -2.000000e+00

modelica.function @neg(%arg0 : !modelica.real) -> (!modelica.real) attributes {args_names = ["x"], results_names = ["y"], derivative = #modelica.derivative<"neg_der", 1>} {
    %0 = modelica.member_create {name = "y"} : !modelica.member<stack, !modelica.real>
    %1 = modelica.neg %arg0 : !modelica.real -> !modelica.real
    modelica.member_store %0, %1 : !modelica.member<stack, !modelica.real>, !modelica.real
    modelica.function_terminator
}

func @test_neg() -> () {
    %x = modelica.constant #modelica.real<57.0> : !modelica.real
    %der_x = modelica.constant #modelica.real<2.0> : !modelica.real
    %result = modelica.call @neg_der(%x, %der_x) : (!modelica.real, !modelica.real) -> (!modelica.real)
    modelica.print %result : !modelica.real
    return
}

// d/dt (x + y) = d/dt (x) + d/dt (y)
// CHECK: 5.000000e+00

modelica.function @add(%arg0 : !modelica.real, %arg1 : !modelica.real) -> (!modelica.real) attributes {args_names = ["x", "y"], results_names = ["z"], derivative = #modelica.derivative<"add_der", 1>} {
    %0 = modelica.member_create {name = "z"} : !modelica.member<stack, !modelica.real>
    %1 = modelica.add %arg0, %arg1 : (!modelica.real, !modelica.real) -> !modelica.real
    modelica.member_store %0, %1 : !modelica.member<stack, !modelica.real>, !modelica.real
    modelica.function_terminator
}

func @test_add() -> () {
    %x = modelica.constant #modelica.real<23.0> : !modelica.real
    %y = modelica.constant #modelica.real<57.0> : !modelica.real
    %der_x = modelica.constant #modelica.real<3.0> : !modelica.real
    %der_y = modelica.constant #modelica.real<2.0> : !modelica.real
    %result = modelica.call @add_der(%x, %y, %der_x, %der_y) : (!modelica.real, !modelica.real, !modelica.real, !modelica.real) -> (!modelica.real)
    modelica.print %result : !modelica.real
    return
}

// d/dt (x - y) = d/dt (x) - d/dt (y)
// CHECK: 1.000000e+00

modelica.function @sub(%arg0 : !modelica.real, %arg1 : !modelica.real) -> (!modelica.real) attributes {args_names = ["x", "y"], results_names = ["z"], derivative = #modelica.derivative<"sub_der", 1>} {
    %0 = modelica.member_create {name = "z"} : !modelica.member<stack, !modelica.real>
    %1 = modelica.sub %arg0, %arg1 : (!modelica.real, !modelica.real) -> !modelica.real
    modelica.member_store %0, %1 : !modelica.member<stack, !modelica.real>, !modelica.real
    modelica.function_terminator
}

func @test_sub() -> () {
    %x = modelica.constant #modelica.real<23.0> : !modelica.real
    %y = modelica.constant #modelica.real<57.0> : !modelica.real
    %der_x = modelica.constant #modelica.real<3.0> : !modelica.real
    %der_y = modelica.constant #modelica.real<2.0> : !modelica.real
    %result = modelica.call @sub_der(%x, %y, %der_x, %der_y) : (!modelica.real, !modelica.real, !modelica.real, !modelica.real) -> (!modelica.real)
    modelica.print %result : !modelica.real
    return
}

// d/dt (x * y) = d/dt (x) * y + x * d/dt (y)
// CHECK: 2.170000e+02

modelica.function @mul(%arg0 : !modelica.real, %arg1 : !modelica.real) -> (!modelica.real) attributes {args_names = ["x", "y"], results_names = ["z"], derivative = #modelica.derivative<"mul_der", 1>} {
    %0 = modelica.member_create {name = "z"} : !modelica.member<stack, !modelica.real>
    %1 = modelica.mul %arg0, %arg1 : (!modelica.real, !modelica.real) -> !modelica.real
    modelica.member_store %0, %1 : !modelica.member<stack, !modelica.real>, !modelica.real
    modelica.function_terminator
}

func @test_mul() -> () {
    %x = modelica.constant #modelica.real<23.0> : !modelica.real
    %y = modelica.constant #modelica.real<57.0> : !modelica.real
    %der_x = modelica.constant #modelica.real<3.0> : !modelica.real
    %der_y = modelica.constant #modelica.real<2.0> : !modelica.real
    %result = modelica.call @mul_der(%x, %y, %der_x, %der_y) : (!modelica.real, !modelica.real, !modelica.real, !modelica.real) -> (!modelica.real)
    modelica.print %result : !modelica.real
    return
}

// d/dt (x / y) = (d/dt (x) * y - x * d/dt (y)) / (y^2)
// CHECK: 3.847338e-02

modelica.function @div(%arg0 : !modelica.real, %arg1 : !modelica.real) -> (!modelica.real) attributes {args_names = ["x", "y"], results_names = ["z"], derivative = #modelica.derivative<"div_der", 1>} {
    %0 = modelica.member_create {name = "z"} : !modelica.member<stack, !modelica.real>
    %1 = modelica.div %arg0, %arg1 : (!modelica.real, !modelica.real) -> !modelica.real
    modelica.member_store %0, %1 : !modelica.member<stack, !modelica.real>, !modelica.real
    modelica.function_terminator
}

func @test_div() -> () {
    %x = modelica.constant #modelica.real<23.0> : !modelica.real
    %y = modelica.constant #modelica.real<57.0> : !modelica.real
    %der_x = modelica.constant #modelica.real<3.0> : !modelica.real
    %der_y = modelica.constant #modelica.real<2.0> : !modelica.real
    %result = modelica.call @div_der(%x, %y, %der_x, %der_y) : (!modelica.real, !modelica.real, !modelica.real, !modelica.real) -> (!modelica.real)
    modelica.print %result : !modelica.real
    return
}

func @main() -> () {
    call @test_var() : () -> ()
    call @test_neg() : () -> ()
    call @test_add() : () -> ()
    call @test_sub() : () -> ()
    call @test_mul() : () -> ()
    call @test_div() : () -> ()
    return
}
