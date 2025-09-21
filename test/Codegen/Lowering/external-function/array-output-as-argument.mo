// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @arrayOutputAsArgument
// CHECK:   bmodelica.algorithm {
// CHECK:       %[[x:.*]] = bmodelica.variable_get @x
// CHECK:       bmodelica.external_call @foo(%[[x]]) : (tensor<2x!bmodelica.int>) -> ()
// CHECK:   }

function arrayOutputAsArgument
    output Integer[2] x;
external "C" foo(x);
end arrayOutputAsArgument;
