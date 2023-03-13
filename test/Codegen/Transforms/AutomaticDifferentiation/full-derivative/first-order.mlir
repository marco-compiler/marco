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

// d/dt (x) = d/dt (x)
// CHECK: 2.000000e+00

modelica.function @var attributes {derivative = #modelica.derivative<"var_der", 1>} {
    modelica.variable @x : !modelica.variable<!modelica.real, input>
    modelica.variable @y : !modelica.variable<!modelica.real, output>

    modelica.algorithm {
        %0 = modelica.variable_get @x : !modelica.real
        modelica.variable_set @y, %0 : !modelica.real
    }
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

modelica.function @neg attributes {derivative = #modelica.derivative<"neg_der", 1>} {
    modelica.variable @x : !modelica.variable<!modelica.real, input>
    modelica.variable @y : !modelica.variable<!modelica.real, output>

    modelica.algorithm {
        %0 = modelica.variable_get @x : !modelica.real
        %1 = modelica.neg %0 : !modelica.real -> !modelica.real
        modelica.variable_set @y, %1 : !modelica.real
    }
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

modelica.function @add attributes {derivative = #modelica.derivative<"add_der", 1>} {
    modelica.variable @x : !modelica.variable<!modelica.real, input>
    modelica.variable @y : !modelica.variable<!modelica.real, input>
    modelica.variable @z : !modelica.variable<!modelica.real, output>

    modelica.algorithm {
        %0 = modelica.variable_get @x : !modelica.real
        %1 = modelica.variable_get @y : !modelica.real
        %2 = modelica.add %0, %1 : (!modelica.real, !modelica.real) -> !modelica.real
        modelica.variable_set @z, %2 : !modelica.real
    }
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

modelica.function @sub attributes {derivative = #modelica.derivative<"sub_der", 1>} {
    modelica.variable @x : !modelica.variable<!modelica.real, input>
    modelica.variable @y : !modelica.variable<!modelica.real, input>
    modelica.variable @z : !modelica.variable<!modelica.real, output>

    modelica.algorithm {
        %0 = modelica.variable_get @x : !modelica.real
        %1 = modelica.variable_get @y : !modelica.real
        %2 = modelica.sub %0, %1 : (!modelica.real, !modelica.real) -> !modelica.real
        modelica.variable_set @z, %2 : !modelica.real
    }
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

modelica.function @mul attributes {derivative = #modelica.derivative<"mul_der", 1>} {
    modelica.variable @x : !modelica.variable<!modelica.real, input>
    modelica.variable @y : !modelica.variable<!modelica.real, input>
    modelica.variable @z : !modelica.variable<!modelica.real, output>

    modelica.algorithm {
        %0 = modelica.variable_get @x : !modelica.real
        %1 = modelica.variable_get @y : !modelica.real
        %2 = modelica.mul %0, %1 : (!modelica.real, !modelica.real) -> !modelica.real
        modelica.variable_set @z, %2 : !modelica.real
    }
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

modelica.function @div attributes {derivative = #modelica.derivative<"div_der", 1>} {
    modelica.variable @x : !modelica.variable<!modelica.real, input>
    modelica.variable @y : !modelica.variable<!modelica.real, input>
    modelica.variable @z : !modelica.variable<!modelica.real, output>

    modelica.algorithm {
        %0 = modelica.variable_get @x : !modelica.real
        %1 = modelica.variable_get @y : !modelica.real
        %2 = modelica.div %0, %1 : (!modelica.real, !modelica.real) -> !modelica.real
        modelica.variable_set @z, %2 : !modelica.real
    }
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
