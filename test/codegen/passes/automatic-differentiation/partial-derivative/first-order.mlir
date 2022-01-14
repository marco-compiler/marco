// RUN: modelica-opt %s                             \
// RUN:     --auto-diff                             \
// RUN:     --convert-modelica-functions            \
// RUN:     --convert-modelica                      \
// RUN:     --convert-to-llvm                       \
// RUN: | mlir-cpu-runner                           \
// RUN:     -e main -entry-point-result=void -O0    \
// RUN:     -shared-libs=%runtime_lib               \
// RUN: | FileCheck %s

// d/dx (x) = 1
// CHECK: 1.000000e+00

modelica.function @simpleVar(%arg0 : !modelica.real) -> (!modelica.real) attributes {args_names = ["x"], results_names = ["y"]} {
    %0 = modelica.member_create {name = "y"} : !modelica.member<stack, !modelica.real>
    modelica.member_store %0, %arg0 : !modelica.member<stack, !modelica.real>
    modelica.function_terminator
}

modelica.der_function @simpleVar_x {derived_function = "simpleVar", independent_vars = ["x"]}

func @test_simpleVarDer() -> () {
    %x = modelica.constant #modelica.real<57.0> : !modelica.real
    %result = modelica.call @simpleVar_x(%x) : (!modelica.real) -> (!modelica.real)
    modelica.print %result : !modelica.real
    return
}

// d/dx (constant * x) = constant
// CHECK: 2.300000e+01

modelica.function @mulByScalar(%arg0 : !modelica.real) -> (!modelica.real) attributes {args_names = ["x"], results_names = ["y"]} {
    %0 = modelica.member_create {name = "y"} : !modelica.member<stack, !modelica.real>
    %1 = modelica.constant #modelica.real<23.0> : !modelica.real
    %2 = modelica.mul %arg0, %1 : (!modelica.real, !modelica.real) -> !modelica.real
    modelica.member_store %0, %2: !modelica.member<stack, !modelica.real>
    modelica.function_terminator
}

modelica.der_function @mulByScalar_x {derived_function = "mulByScalar", independent_vars = ["x"]}

func @test_mulByScalar() -> () {
    %x = modelica.constant #modelica.real<57.0> : !modelica.real
    %result = modelica.call @mulByScalar_x(%x) : (!modelica.real) -> (!modelica.real)
    modelica.print %result : !modelica.real
    return
}

// d/dx (x + y) = 1
// CHECK: 1.000000e+00

modelica.function @sumOfVars(%arg0 : !modelica.real, %arg1 : !modelica.real) -> (!modelica.real) attributes {args_names = ["x", "y"], results_names = ["z"]} {
    %0 = modelica.member_create {name = "z"} : !modelica.member<stack, !modelica.real>
    %1 = modelica.add %arg0, %arg1 : (!modelica.real, !modelica.real) -> !modelica.real
    modelica.member_store %0, %1: !modelica.member<stack, !modelica.real>
    modelica.function_terminator
}

modelica.der_function @sumOfVars_x {derived_function = "sumOfVars", independent_vars = ["x"]}

func @test_sumOfVars() -> () {
    %x = modelica.constant #modelica.real<57.0> : !modelica.real
    %y = modelica.constant #modelica.real<23.0> : !modelica.real
    %result = modelica.call @sumOfVars_x(%x, %y) : (!modelica.real, !modelica.real) -> (!modelica.real)
    modelica.print %result : !modelica.real
    return
}

// d/dx (x * y) = y
// CHECK: 2.300000e+01

modelica.function @mulOfVars(%arg0 : !modelica.real, %arg1 : !modelica.real) -> (!modelica.real) attributes {args_names = ["x", "y"], results_names = ["z"]} {
    %0 = modelica.member_create {name = "z"} : !modelica.member<stack, !modelica.real>
    %1 = modelica.mul %arg0, %arg1 : (!modelica.real, !modelica.real) -> !modelica.real
    modelica.member_store %0, %1: !modelica.member<stack, !modelica.real>
    modelica.function_terminator
}

modelica.der_function @mulOfVars_x {derived_function = "mulOfVars", independent_vars = ["x"]}

func @test_mulOfVars() -> () {
    %x = modelica.constant #modelica.real<57.0> : !modelica.real
    %y = modelica.constant #modelica.real<23.0> : !modelica.real
    %result = modelica.call @mulOfVars_x(%x, %y) : (!modelica.real, !modelica.real) -> (!modelica.real)
    modelica.print %result : !modelica.real
    return
}

// foo(x) = x * constant1
// d/dx foo(constant2 * x) = constant2 * constant1
// CHECK: 1.311000e+03

modelica.function @scalarMul(%arg0 : !modelica.real) -> (!modelica.real) attributes {args_names = ["x1"], results_names = ["y1"]} {
    %0 = modelica.member_create {name = "y1"} : !modelica.member<stack, !modelica.real>
    %1 = modelica.constant #modelica.real<23.0> : !modelica.real
    %2 = modelica.mul %arg0, %1 : (!modelica.real, !modelica.real) -> !modelica.real
    modelica.member_store %0, %2: !modelica.member<stack, !modelica.real>
    modelica.function_terminator
}

modelica.function @callOpDer(%arg0 : !modelica.real) -> (!modelica.real) attributes {args_names = ["x2"], results_names = ["y2"]} {
    %0 = modelica.member_create {name = "y2"} : !modelica.member<stack, !modelica.real>
    %1 = modelica.constant #modelica.int<57> : !modelica.int
    %2 = modelica.mul %1, %arg0 : (!modelica.int, !modelica.real) -> !modelica.real
    %3 = modelica.call @scalarMul(%2) : (!modelica.real) -> (!modelica.real)
    modelica.member_store %0, %3: !modelica.member<stack, !modelica.real>
    modelica.function_terminator
}

modelica.der_function @callOpDer_x2 {derived_function = "callOpDer", independent_vars = ["x2"]}

func @test_callOpDer() -> () {
    %x = modelica.constant #modelica.real<2000.0> : !modelica.real
    %result = modelica.call @callOpDer_x2(%x) : (!modelica.real) -> (!modelica.real)
    modelica.print %result : !modelica.real
    return
}

func @main() -> () {
    call @test_simpleVarDer() : () -> ()
    call @test_mulByScalar() : () -> ()
    call @test_sumOfVars() : () -> ()
    call @test_mulOfVars() : () -> ()
    call @test_callOpDer() : () -> ()
    return
}
