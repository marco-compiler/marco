// RUN: modelica-opt %s                             \
// RUN:     --auto-diff                             \
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

// d/dt (x) = d/dt (x)
// CHECK: 2.000000e+00

modelica.function @var : (!modelica.real) -> (!modelica.real) attributes {derivative = #modelica.derivative<"var_der", 1>} {
    %0 = modelica.member_create @x : !modelica.member<!modelica.real, input>
    %1 = modelica.member_create @y : !modelica.member<!modelica.real, output>
    %2 = modelica.member_load %0 : !modelica.member<!modelica.real, input> -> !modelica.real
    modelica.member_store %1, %2 : !modelica.member<!modelica.real, output>, !modelica.real
}

func.func @test_var() -> () {
    %x = modelica.constant #modelica.real<57.0> : !modelica.real
    %der_x = modelica.constant #modelica.real<2.0> : !modelica.real
    %result = modelica.call @var_der(%x, %der_x) : (!modelica.real, !modelica.real) -> (!modelica.real)
    modelica.print %result : !modelica.real
    return
}

// d/dt (-x) = - d/dt (x)
// CHECK: -2.000000e+00

modelica.function @neg : (!modelica.real) -> (!modelica.real) attributes {derivative = #modelica.derivative<"neg_der", 1>} {
    %0 = modelica.member_create @x : !modelica.member<!modelica.real, input>
    %1 = modelica.member_create @y : !modelica.member<!modelica.real, output>
    %2 = modelica.member_load %0 : !modelica.member<!modelica.real, input> -> !modelica.real
    %3 = modelica.neg %2 : !modelica.real -> !modelica.real
    modelica.member_store %1, %3 : !modelica.member<!modelica.real, output>, !modelica.real
}

func.func @test_neg() -> () {
    %x = modelica.constant #modelica.real<57.0>
    %der_x = modelica.constant #modelica.real<2.0>
    %result = modelica.call @neg_der(%x, %der_x) : (!modelica.real, !modelica.real) -> (!modelica.real)
    modelica.print %result : !modelica.real
    return
}

// d/dt (x + y) = d/dt (x) + d/dt (y)
// CHECK: 5.000000e+00

modelica.function @add : (!modelica.real, !modelica.real) -> (!modelica.real) attributes {derivative = #modelica.derivative<"add_der", 1>} {
    %0 = modelica.member_create @x : !modelica.member<!modelica.real, input>
    %1 = modelica.member_create @y : !modelica.member<!modelica.real, input>
    %2 = modelica.member_create @z : !modelica.member<!modelica.real, output>
    %3 = modelica.member_load %0 : !modelica.member<!modelica.real, input> -> !modelica.real
    %4 = modelica.member_load %1 : !modelica.member<!modelica.real, input> -> !modelica.real
    %5 = modelica.add %3, %4 : (!modelica.real, !modelica.real) -> !modelica.real
    modelica.member_store %2, %5 : !modelica.member<!modelica.real, output>, !modelica.real
}

func.func @test_add() -> () {
    %x = modelica.constant #modelica.real<23.0>
    %y = modelica.constant #modelica.real<57.0>
    %der_x = modelica.constant #modelica.real<3.0>
    %der_y = modelica.constant #modelica.real<2.0>
    %result = modelica.call @add_der(%x, %y, %der_x, %der_y) : (!modelica.real, !modelica.real, !modelica.real, !modelica.real) -> (!modelica.real)
    modelica.print %result : !modelica.real
    return
}

// d/dt (x - y) = d/dt (x) - d/dt (y)
// CHECK: 1.000000e+00

modelica.function @sub : (!modelica.real, !modelica.real) -> (!modelica.real) attributes {derivative = #modelica.derivative<"sub_der", 1>} {
    %0 = modelica.member_create @x : !modelica.member<!modelica.real, input>
    %1 = modelica.member_create @y : !modelica.member<!modelica.real, input>
    %2 = modelica.member_create @z : !modelica.member<!modelica.real, output>
    %3 = modelica.member_load %0 : !modelica.member<!modelica.real, input> -> !modelica.real
    %4 = modelica.member_load %1 : !modelica.member<!modelica.real, input> -> !modelica.real
    %5 = modelica.sub %3, %4 : (!modelica.real, !modelica.real) -> !modelica.real
    modelica.member_store %2, %5 : !modelica.member<!modelica.real, output>, !modelica.real
}

func.func @test_sub() -> () {
    %x = modelica.constant #modelica.real<23.0>
    %y = modelica.constant #modelica.real<57.0>
    %der_x = modelica.constant #modelica.real<3.0>
    %der_y = modelica.constant #modelica.real<2.0>
    %result = modelica.call @sub_der(%x, %y, %der_x, %der_y) : (!modelica.real, !modelica.real, !modelica.real, !modelica.real) -> (!modelica.real)
    modelica.print %result : !modelica.real
    return
}

// d/dt (x * y) = d/dt (x) * y + x * d/dt (y)
// CHECK: 2.170000e+02

modelica.function @mul : (!modelica.real, !modelica.real) -> (!modelica.real) attributes {derivative = #modelica.derivative<"mul_der", 1>} {
    %0 = modelica.member_create @x : !modelica.member<!modelica.real, input>
    %1 = modelica.member_create @y : !modelica.member<!modelica.real, input>
    %2 = modelica.member_create @z : !modelica.member<!modelica.real, output>
    %3 = modelica.member_load %0 : !modelica.member<!modelica.real, input> -> !modelica.real
    %4 = modelica.member_load %1 : !modelica.member<!modelica.real, input> -> !modelica.real
    %5 = modelica.mul %3, %4 : (!modelica.real, !modelica.real) -> !modelica.real
    modelica.member_store %2, %5 : !modelica.member<!modelica.real, output>, !modelica.real
}

func.func @test_mul() -> () {
    %x = modelica.constant #modelica.real<23.0>
    %y = modelica.constant #modelica.real<57.0>
    %der_x = modelica.constant #modelica.real<3.0>
    %der_y = modelica.constant #modelica.real<2.0>
    %result = modelica.call @mul_der(%x, %y, %der_x, %der_y) : (!modelica.real, !modelica.real, !modelica.real, !modelica.real) -> (!modelica.real)
    modelica.print %result : !modelica.real
    return
}

// d/dt (x / y) = (d/dt (x) * y - x * d/dt (y)) / (y^2)
// CHECK: 3.847338e-02

modelica.function @div : (!modelica.real, !modelica.real) -> (!modelica.real) attributes {derivative = #modelica.derivative<"div_der", 1>} {
    %0 = modelica.member_create @x : !modelica.member<!modelica.real, input>
    %1 = modelica.member_create @y : !modelica.member<!modelica.real, input>
    %2 = modelica.member_create @z : !modelica.member<!modelica.real, output>
    %3 = modelica.member_load %0 : !modelica.member<!modelica.real, input> -> !modelica.real
    %4 = modelica.member_load %1 : !modelica.member<!modelica.real, input> -> !modelica.real
    %5 = modelica.div %3, %4 : (!modelica.real, !modelica.real) -> !modelica.real
    modelica.member_store %2, %5 : !modelica.member<!modelica.real, output>, !modelica.real
}

func.func @test_div() -> () {
    %x = modelica.constant #modelica.real<23.0>
    %y = modelica.constant #modelica.real<57.0>
    %der_x = modelica.constant #modelica.real<3.0>
    %der_y = modelica.constant #modelica.real<2.0>
    %result = modelica.call @div_der(%x, %y, %der_x, %der_y) : (!modelica.real, !modelica.real, !modelica.real, !modelica.real) -> (!modelica.real)
    modelica.print %result : !modelica.real
    return
}

func.func @main() -> () {
    call @test_var() : () -> ()
    call @test_neg() : () -> ()
    call @test_add() : () -> ()
    call @test_sub() : () -> ()
    call @test_mul() : () -> ()
    call @test_div() : () -> ()
    return
}
