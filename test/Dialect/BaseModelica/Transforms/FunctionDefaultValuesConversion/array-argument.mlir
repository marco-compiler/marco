// RUN: modelica-opt %s --split-input-file --function-default-values-conversion | FileCheck %s

// CHECK-LABEL: @caller
// CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica.dense_int<[0, 0, 0]> : tensor<3x!bmodelica.int>
// CHECK: bmodelica.call @arrayDefaultValue(%[[cst]]) : (tensor<3x!bmodelica.int>) -> ()

bmodelica.function @arrayDefaultValue {
    bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.int, input>

    bmodelica.default @x {
        %0 = bmodelica.constant #bmodelica.dense_int<[0, 0, 0]> : tensor<3x!bmodelica.int>
        bmodelica.yield %0 : tensor<3x!bmodelica.int>
    }

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x : tensor<3x!bmodelica.int>
        bmodelica.print %0 : tensor<3x!bmodelica.int>
    }
}

func.func @caller() {
    bmodelica.call @arrayDefaultValue() : () -> ()
    func.return
}
