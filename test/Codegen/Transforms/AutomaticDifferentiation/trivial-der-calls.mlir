// RUN: modelica-opt %s                                 \
// RUN:     --auto-diff                                 \
// RUN:     --convert-modelica-to-cf                    \
// RUN:     --convert-modelica-to-arith                 \
// RUN:     --convert-modelica-to-func                  \
// RUN:     --convert-modelica-to-memref                \
// RUN:     --convert-modelica-to-llvm                  \
// RUN:     --convert-arith-to-llvm                     \
// RUN:     --convert-memref-to-llvm                    \
// RUN:     --convert-cf-to-llvm                        \
// RUN:     --convert-func-to-llvm                      \
// RUN:     --reconcile-unrealized-casts                \
// RUN: | mlir-cpu-runner                               \
// RUN:     -e main -entry-point-result=void -O0        \
// RUN:     -shared-libs=%runtime_lib                   \
// RUN: | FileCheck %s

// CHECK: -1.979985e+00

modelica.function @sin_der {
    modelica.variable @x : !modelica.variable<!modelica.real, input>
    modelica.variable @der_x : !modelica.variable<!modelica.real, input>
    modelica.variable @y : !modelica.variable<!modelica.real, output>

    modelica.algorithm {
        %0 = modelica.variable_get @x : !modelica.real
        %1 = modelica.sin %0 : !modelica.real -> !modelica.real
        %2 = modelica.der %1 : !modelica.real -> !modelica.real
        modelica.variable_set @y, %2 : !modelica.real
    }
}

func.func @test_sin() -> () {
    %x = modelica.constant #modelica.real<3.0>
    %der_x = modelica.constant #modelica.real<2.0>
    %result = modelica.call @sin_der(%x, %der_x) : (!modelica.real, !modelica.real) -> (!modelica.real)
    modelica.print %result : !modelica.real
    return
}

// CHECK: 2.822400e-01

modelica.function @cos_der {
    modelica.variable @x : !modelica.variable<!modelica.real, input>
    modelica.variable @der_x : !modelica.variable<!modelica.real, input>
    modelica.variable @y : !modelica.variable<!modelica.real, output>

    modelica.algorithm {
        %0 = modelica.variable_get @x : !modelica.real
        %1 = modelica.cos %0 : !modelica.real -> !modelica.real
        %2 = modelica.der %1 : !modelica.real -> !modelica.real
        modelica.variable_set @y, %2 : !modelica.real
    }
}

func.func @test_cos() -> () {
    %x = modelica.constant #modelica.real<3.0>
    %der_x = modelica.constant #modelica.real<2.0>
    %result = modelica.call @cos_der(%x, %der_x) : (!modelica.real, !modelica.real) -> (!modelica.real)
    modelica.print %result : !modelica.real
    return
}

func.func @main() -> () {
    call @test_sin() : () -> ()
    call @test_cos() : () -> ()
    return
}
