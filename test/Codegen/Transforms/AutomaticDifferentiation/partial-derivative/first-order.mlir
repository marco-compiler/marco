// RUN: modelica-opt %s                                 \
// RUN:     --auto-diff                                 \
// RUN:     --convert-modelica-to-cf                    \
// RUN:     --convert-modelica-to-arith                 \
// RUN:     --convert-modelica-to-func                  \
// RUN:     --convert-modelica-to-memref                \
// RUN:     --convert-modelica-to-llvm                  \
// RUN:     --convert-arith-to-llvm                     \
// RUN:     --convert-memref-to-llvm                    \
// RUN:     --convert-func-to-llvm                      \
// RUN:     --convert-cf-to-llvm                        \
// RUN:     --reconcile-unrealized-casts                \
// RUN: | mlir-cpu-runner                               \
// RUN:     -e main -entry-point-result=void -O0        \
// RUN:     -shared-libs=%runtime_lib                   \
// RUN: | FileCheck %s

// d/dx (x) = 1
// CHECK: 1.000000e+00

modelica.function @simpleVar {
    modelica.variable @x : !modelica.member<!modelica.real, input>
    modelica.variable @y : !modelica.member<!modelica.real, output>

    modelica.algorithm {
        %0 = modelica.variable_get @x : !modelica.real
        modelica.variable_set @y, %0 : !modelica.real
    }
}

modelica.der_function @simpleVar_x {derived_function = "simpleVar", independent_vars = ["x"]}

func.func @test_simpleVarDer() -> () {
    %x = modelica.constant #modelica.real<57.0>
    %result = modelica.call @simpleVar_x(%x) : (!modelica.real) -> (!modelica.real)
    modelica.print %result : !modelica.real
    return
}

// d/dx (constant * x) = constant
// CHECK: 2.300000e+01

modelica.function @mulByScalar {
    modelica.variable @x : !modelica.member<!modelica.real, input>
    modelica.variable @y : !modelica.member<!modelica.real, output>

    modelica.algorithm {
        %0 = modelica.constant #modelica.real<23.0>
        %1 = modelica.variable_get @x : !modelica.real
        %2 = modelica.mul %0, %1 : (!modelica.real, !modelica.real) -> !modelica.real
        modelica.variable_set @y, %2: !modelica.real
    }
}

modelica.der_function @mulByScalar_x {derived_function = "mulByScalar", independent_vars = ["x"]}

func.func @test_mulByScalar() -> () {
    %x = modelica.constant #modelica.real<57.0>
    %result = modelica.call @mulByScalar_x(%x) : (!modelica.real) -> (!modelica.real)
    modelica.print %result : !modelica.real
    return
}

// d/dx (x + y) = 1
// CHECK: 1.000000e+00

modelica.function @sumOfVars {
    modelica.variable @x : !modelica.member<!modelica.real, input>
    modelica.variable @y : !modelica.member<!modelica.real, input>
    modelica.variable @z : !modelica.member<!modelica.real, output>

    modelica.algorithm {
        %0 = modelica.variable_get @x : !modelica.real
        %1 = modelica.variable_get @y : !modelica.real
        %2 = modelica.add %0, %1 : (!modelica.real, !modelica.real) -> !modelica.real
        modelica.variable_set @z, %2: !modelica.real
    }
}

modelica.der_function @sumOfVars_x {derived_function = "sumOfVars", independent_vars = ["x"]}

func.func @test_sumOfVars() -> () {
    %x = modelica.constant #modelica.real<57.0>
    %y = modelica.constant #modelica.real<23.0>
    %result = modelica.call @sumOfVars_x(%x, %y) : (!modelica.real, !modelica.real) -> (!modelica.real)
    modelica.print %result : !modelica.real
    return
}

// d/dx (x * y) = y
// CHECK: 2.300000e+01

modelica.function @mulOfVars {
    modelica.variable @x : !modelica.member<!modelica.real, input>
    modelica.variable @y : !modelica.member<!modelica.real, input>
    modelica.variable @z : !modelica.member<!modelica.real, output>

    modelica.algorithm {
        %0 = modelica.variable_get @x : !modelica.real
        %1 = modelica.variable_get @y : !modelica.real
        %2 = modelica.mul %0, %1 : (!modelica.real, !modelica.real) -> !modelica.real
        modelica.variable_set @z, %2: !modelica.real
    }
}

modelica.der_function @mulOfVars_x {derived_function = "mulOfVars", independent_vars = ["x"]}

func.func @test_mulOfVars() -> () {
    %x = modelica.constant #modelica.real<57.0>
    %y = modelica.constant #modelica.real<23.0>
    %result = modelica.call @mulOfVars_x(%x, %y) : (!modelica.real, !modelica.real) -> (!modelica.real)
    modelica.print %result : !modelica.real
    return
}

// foo(x) = x * constant1
// d/dx foo(constant2 * x) = constant2 * constant1
// CHECK: 1.311000e+03

modelica.function @scalarMul {
    modelica.variable @x1 : !modelica.member<!modelica.real, input>
    modelica.variable @y1 : !modelica.member<!modelica.real, output>

    modelica.algorithm {
        %0 = modelica.variable_get @x1 : !modelica.real
        %1 = modelica.constant #modelica.real<23.0>
        %2 = modelica.mul %0, %1 : (!modelica.real, !modelica.real) -> !modelica.real
        modelica.variable_set @y1, %2: !modelica.real
    }
}

modelica.function @callOpDer {
    modelica.variable @x2 : !modelica.member<!modelica.real, input>
    modelica.variable @y2 : !modelica.member<!modelica.real, output>

    modelica.algorithm {
        %0 = modelica.constant #modelica.int<57>
        %1 = modelica.variable_get @x2 : !modelica.real
        %2 = modelica.mul %0, %1 : (!modelica.int, !modelica.real) -> !modelica.real
        %3 = modelica.call @scalarMul(%2) : (!modelica.real) -> (!modelica.real)
        modelica.variable_set @y2, %3: !modelica.real
    }
}

modelica.der_function @callOpDer_x2 {derived_function = "callOpDer", independent_vars = ["x2"]}

func.func @test_callOpDer() -> () {
    %x = modelica.constant #modelica.real<2000.0>
    %result = modelica.call @callOpDer_x2(%x) : (!modelica.real) -> (!modelica.real)
    modelica.print %result : !modelica.real
    return
}

func.func @main() -> () {
    call @test_simpleVarDer() : () -> ()
    call @test_mulByScalar() : () -> ()
    call @test_sumOfVars() : () -> ()
    call @test_mulOfVars() : () -> ()
    call @test_callOpDer() : () -> ()
    return
}
