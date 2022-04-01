// RUN: modelica-opt %s                             \
// RUN:     --auto-diff                             \
// RUN:     --convert-modelica                      \
// RUN:     --convert-modelica-to-cfg               \
// RUN:     --convert-to-llvm                       \
// RUN: | mlir-cpu-runner                           \
// RUN:     -e main -entry-point-result=void -O0    \
// RUN:     -shared-libs=%runtime_lib               \
// RUN: | FileCheck %s

// d/dx (x) = 1
// CHECK: 1.000000e+00

modelica.function @simpleVar : (!modelica.real) -> (!modelica.real) {
    %0 = modelica.member_create {name = "x"} : !modelica.member<!modelica.real, input>
    %1 = modelica.member_create {name = "y"} : !modelica.member<!modelica.real, output>
    %2 = modelica.member_load %0 : !modelica.member<!modelica.real, input> -> !modelica.real
    modelica.member_store %1, %2 : !modelica.member<!modelica.real, output>, !modelica.real
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

modelica.function @mulByScalar : (!modelica.real) -> (!modelica.real) {
    %0 = modelica.member_create {name = "x"} : !modelica.member<!modelica.real, input>
    %1 = modelica.member_create {name = "y"} : !modelica.member<!modelica.real, output>
    %2 = modelica.constant #modelica.real<23.0> : !modelica.real
    %3 = modelica.member_load %0 : !modelica.member<!modelica.real, input> -> !modelica.real
    %4 = modelica.mul %3, %2 : (!modelica.real, !modelica.real) -> !modelica.real
    modelica.member_store %1, %4: !modelica.member<!modelica.real, output>, !modelica.real
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

modelica.function @sumOfVars : (!modelica.real, !modelica.real) -> (!modelica.real) {
    %0 = modelica.member_create {name = "x"} : !modelica.member<!modelica.real, input>
    %1 = modelica.member_create {name = "y"} : !modelica.member<!modelica.real, input>
    %2 = modelica.member_create {name = "z"} : !modelica.member<!modelica.real, output>
    %3 = modelica.member_load %0 : !modelica.member<!modelica.real, input> -> !modelica.real
    %4 = modelica.member_load %1 : !modelica.member<!modelica.real, input> -> !modelica.real
    %5 = modelica.add %3, %4 : (!modelica.real, !modelica.real) -> !modelica.real
    modelica.member_store %2, %5: !modelica.member<!modelica.real, output>, !modelica.real
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

modelica.function @mulOfVars : (!modelica.real, !modelica.real) -> (!modelica.real) {
    %0 = modelica.member_create {name = "x"} : !modelica.member<!modelica.real, input>
    %1 = modelica.member_create {name = "y"} : !modelica.member<!modelica.real, input>
    %2 = modelica.member_create {name = "z"} : !modelica.member<!modelica.real, output>
    %3 = modelica.member_load %0 : !modelica.member<!modelica.real, input> -> !modelica.real
    %4 = modelica.member_load %1 : !modelica.member<!modelica.real, input> -> !modelica.real
    %5 = modelica.mul %3, %4 : (!modelica.real, !modelica.real) -> !modelica.real
    modelica.member_store %2, %5: !modelica.member<!modelica.real, output>, !modelica.real
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

modelica.function @scalarMul : (!modelica.real) -> (!modelica.real) {
    %0 = modelica.member_create {name = "x1"} : !modelica.member<!modelica.real, input>
    %1 = modelica.member_create {name = "y1"} : !modelica.member<!modelica.real, output>
    %2 = modelica.member_load %0 : !modelica.member<!modelica.real, input> -> !modelica.real
    %3 = modelica.constant #modelica.real<23.0> : !modelica.real
    %4 = modelica.mul %2, %3 : (!modelica.real, !modelica.real) -> !modelica.real
    modelica.member_store %1, %4: !modelica.member<!modelica.real, output>, !modelica.real
}

modelica.function @callOpDer : (!modelica.real) -> (!modelica.real) {
    %0 = modelica.member_create {name = "x2"} : !modelica.member<!modelica.real, input>
    %1 = modelica.member_create {name = "y2"} : !modelica.member<!modelica.real, output>
    %2 = modelica.constant #modelica.int<57> : !modelica.int
    %3 = modelica.member_load %0 : !modelica.member<!modelica.real, input> -> !modelica.real
    %4 = modelica.mul %2, %3 : (!modelica.int, !modelica.real) -> !modelica.real
    %5 = modelica.call @scalarMul(%4) : (!modelica.real) -> (!modelica.real)
    modelica.member_store %1, %5: !modelica.member<!modelica.real, output>, !modelica.real
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
